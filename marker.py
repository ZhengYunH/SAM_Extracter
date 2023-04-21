import tkinter as tk
from tkinter import *
from tkinter import simpledialog, filedialog
from PIL import Image, ImageTk
import numpy as np
from tkinter import ttk
from scipy.ndimage import binary_dilation
import windnd
import torch
import numpy as np
import cv2
from segment_anything import sam_model_registry, SamPredictor
import os
import glob
import sys


sam_root = "X:\segment-anything"
sam_model_root = os.path.join(sam_root, "models")
enable_CUDA = torch.cuda.is_available()
sys.path.append(sam_root)
mask_padding = 1 # pixel padding for mask

def get_checkpoints_dict():
    global sam_model_root
    ckpts = glob.glob(sam_model_root + "/*.pth")
    ckpts_dict = {}
    for ckpt in ckpts:
        ckpt_name = os.path.basename(ckpt)
        vit_index = ckpt_name.find("vit")
        if vit_index != -1:
            model_type = ckpt_name[vit_index: vit_index + 5]
            ckpts_dict[model_type] = ckpt
    return ckpts_dict

def get_predictor(model_type, ckpt_path):
    sam = sam_model_registry[model_type](checkpoint=ckpt_path)
    if enable_CUDA:
        sam.to(device='cuda')
    predictor = SamPredictor(sam)
    return predictor

def get_embedding(predictor, image_path):
    image = cv2.imread(image_path)
    predictor.set_image(image)
    image_embedding = predictor.get_image_embedding().cpu().numpy()
    return image_embedding


def predict(predictor, input_point, input_label, bbox):
    # image_embedding = torch.from_numpy(image_embedding).unsqueeze(0)
    # if enable_CUDA:
        # image_embedding = image_embedding.to(device='cuda')
    input_bbox = np.array(bbox) if bbox else None
    input_point = np.array(input_point) if len(input_point) > 0 else None
    input_label = np.array(input_label) if len(input_label) > 0 else None
    return predictor.predict(point_coords=input_point, point_labels=input_label, box=input_bbox) # Choose the model's best mask


class SaveDialog(simpledialog.Dialog):
    def __init__(self, parent, width=-1, height=-1, title=None):
        self.width = width
        self.height = height
        super().__init__(parent, title)
       

    def body(self, master):
        tk.Label(master, text="width：").grid(row=0)
        tk.Label(master, text="height：").grid(row=1)
        tk.Button(master, text="512*512", command=self.set_512).grid(row=2, column=0)

        self.entry1 = tk.Entry(master)
        self.entry2 = tk.Entry(master)

        self.entry1.insert(0, str(self.width))
        self.entry2.insert(0, str(self.height))

        self.entry1.grid(row=0, column=1)
        self.entry2.grid(row=1, column=1)

        return self.entry1  # 将光标定位在第一个输入框

    def set_512(self):
        self.entry1.delete(0, END)
        self.entry1.insert(0, "512")
        self.entry2.delete(0, END)
        self.entry2.insert(0, "512")

    def buttonbox(self):
        box = tk.Frame(self)

        # 定义确定按钮
        ok_button = tk.Button(box, text="确定", width=10, command=self.ok, default=tk.ACTIVE)
        ok_button.pack(side=tk.LEFT, padx=5, pady=5)

        # 定义取消按钮
        cancel_button = tk.Button(box, text="取消", width=10, command=self.cancel)
        cancel_button.pack(side=tk.LEFT, padx=5, pady=5)

        self.bind("<Return>", self.ok)
        self.bind("<Escape>", self.cancel)

        box.pack()

    def ok(self, event=None):
        self.apply()
        self.result = "ok"  # 标记结果为"ok"
        self.withdraw()
        self.update_idletasks()
        self.parent.focus_set()
        self.destroy()

    def cancel(self, event=None):
        self.result = "cancel"  # 标记结果为"cancel"
        self.parent.focus_set()
        self.destroy()

    def apply(self):
        self.entry_values = (self.entry1.get(), self.entry2.get())

class App:
    def __init__(self, master):
        # 创建主窗口
        self.master = master
        self.master.title("标记图片上的点")
        
        # 创建菜单
        self.menu = Menu(self.master)
        self.master.config(menu=self.menu)
        
        # self.file_menu = Menu(self.menu)
        # self.menu.add_cascade(label="文件", menu=self.file_menu)
        # self.file_menu.add_command(label="导入图片", command=self.import_image)
        # self.file_menu.add_command(label="保存点坐标", command=self.save_points)
        # self.file_menu.add_command(label="加载点坐标", command=self.load_points)
        
        # 创建画布
        self.canvas = Canvas(self.master, bg="white")
        self.canvas.pack(fill=BOTH, expand=YES)
        
        # 初始化变量
        self.image = None
        self.image_tk = None
        self.points = []
        self.foreColor = "red"
        self.save_counter = 0
        self.last_save_path = ""
        self.filename = ""
        self.predictor = None
        
        self.ckpt_dict = get_checkpoints_dict()
        option_list = [f"{key}" for key, value in self.ckpt_dict.items()]
        if not len(option_list):
            tk.messagebox.showinfo("提示", f"无可用模型, 确保{sam_model_root}内有模型文件")
            return

        # 添加导入图片按钮
        self.import_button = Button(self.master, text="导入图片", command=self.import_image)
        self.import_button.pack(side=LEFT, padx=10, pady=10)

        self.clear_button = Button(self.master, text="清除点", command=self.clear_points)
        self.clear_button.pack(side=LEFT, padx=10, pady=10)

      
        self.list_model_combo = ttk.Combobox(self.master, values=option_list)
        self.list_model_combo.pack(side=LEFT, padx=10, pady=10)     
        self.list_model_combo.current(0)   
        
        self.import_model_button = Button(self.master, text="导入模型", command=self.import_model)
        self.import_model_button.pack(side=LEFT, padx=10, pady=10)
        self.predictor = None

        self.box_mode_check = Checkbutton(self.master, text="框选模式", command=self.box_mode)
        self.box_mode_check.pack(side=LEFT, padx=10, pady=10)
        self.enable_box_mode = False
        self.draw_box = False
        self.box_handle = None

        self.mask_handles = []

    def _on_drop(self, files):
        if not self.predictor:
            tk.messagebox.showerror("错误", "请先导入模型")
            return
        if files:
            self.import_image_impl(files[0].decode("utf-8"))

    def box_mode(self):
        self.enable_box_mode = not self.enable_box_mode
        # self.clear_points()
        if self.enable_box_mode:
            self.canvas.bind("<Button-1>", self.box_start)
            self.canvas.bind("<B1-Motion>", self.box_move)
            self.canvas.bind("<ButtonRelease-1>", self.box_end)
        else:
            self.canvas.bind("<Button-1>", lambda event: self.add_point(event, "red"))
            self.canvas.unbind("<B1-Motion>")
            self.canvas.unbind("<ButtonRelease-1>")

    def box_start(self, event):
        if self.box_handle:
            self.canvas.delete(self.box_handle)
            self.box_handle = None
        self.box_start_x = event.x
        self.box_start_y = event.y
        self.box_handle = self.canvas.create_rectangle(self.box_start_x, self.box_start_y, self.box_start_x, self.box_start_y, outline="red", width=2)

    def box_move(self, event):
        self.canvas.coords(self.box_handle, self.box_start_x, self.box_start_y, event.x, event.y)

    def box_end(self, event):
        self.canvas.coords(self.box_handle, self.box_start_x, self.box_start_y, event.x, event.y)

        self.box_start_x, self.box_end_x = min(self.box_start_x, event.x) / self.canvas_scale, max(self.box_start_x, event.x) / self.canvas_scale
        self.box_start_y, self.box_end_y = min(self.box_start_y, event.y) / self.canvas_scale, max(self.box_start_y, event.y) / self.canvas_scale
        if abs(self.box_start_x - self.box_end_x) < 1 or abs(self.box_start_y - self.box_end_y) < 1:
            self.canvas.delete(self.box_handle)
            self.box_handle = None
        self.do_predict()

    def import_model(self):
        model_type = self.list_model_combo.get()
        ckpt_path = self.ckpt_dict[model_type]
        self.predictor = get_predictor(model_type, ckpt_path)
        # re-import image
        if self.filename: 
            self.import_image_impl(self.filename)

    def import_image(self):
        if not self.predictor:
            tk.messagebox.showerror("错误", "请先导入模型")
            return

        # 弹出文件选择对话框
        filetypes = (("JPEG 文件", "*.jpg"), ("PNG 文件", "*.png"), ("所有文件", "*.*"))
        filename = filedialog.askopenfilename(filetypes=filetypes)
        self.import_image_impl(filename)

    def import_image_impl(self, filename=None):
        if self.filename and self.filename != filename:
            self.save_counter = 0

        # 加载图片并调整窗口大小
        if filename:
            self.filename = filename
            self.image = Image.open(filename)
            self.canvas_scale = 1.0
            if self.image.width > 1920 or self.image.height > 1080:
                self.canvas_scale = min(1920 / self.image.width, 1080 / self.image.height)

            self.master.geometry("{}x{}".format(int(self.image.width * self.canvas_scale), int(self.image.height * self.canvas_scale)))
            self.image_resize = self.image.resize((int(self.image.width * self.canvas_scale), int(self.image.height * self.canvas_scale)), Image.LANCZOS)
            self.image_embedding = get_embedding(self.predictor, filename)
            self.image_tk = ImageTk.PhotoImage(self.image_resize)
            self.canvas.create_image(0, 0, anchor=NW, image=self.image_tk)

            self.clear_points()
            if self.enable_box_mode:
                self.canvas.bind("<Button-1>", self.box_start)
                self.canvas.bind("<B1-Motion>", self.box_move)
                self.canvas.bind("<ButtonRelease-1>", self.box_end)
            else:
                self.canvas.bind("<Button-1>", lambda event: self.add_point(event, "red"))
                self.canvas.unbind("<B1-Motion>")
                self.canvas.unbind("<ButtonRelease-1>")

            # 隐藏导入图片按钮并解除绑定事件
            # self.import_button.pack_forget()
            self.canvas.bind("<Button-2>", self.delete_point)
            self.canvas.bind("<Button-3>", lambda event: self.add_point(event, "blue"))
            self.canvas.bind("<Control-Button-1>", self.save_file)
            # self.canvas.bind("<Control-s>", self.save_file)

    def clear_points(self):
        self.points = []
        self.canvas.delete("all")
        self.canvas.create_image(0, 0, anchor=NW, image=self.image_tk)

    def save_file(self, event):
        x, y = int(event.x/ self.canvas_scale), int(event.y/self.canvas_scale)

        mi = None
        mip = None
        for mask_pack in self.mask_handles:
            image_id, mask_image, mask_image_tk, reize_tk, mask_image_pil = mask_pack
            if any(mask_image[y][x]):
                mi = mask_image
                mip = mask_image_pil
                break

        if mi is None:
            return
        
       # 获取非零元素的坐标
        nonzero_coords = np.where(mi != 0)

        # 获取非零元素的包围盒
        bbox = (np.min(nonzero_coords[1]), np.min(nonzero_coords[0]), np.max(nonzero_coords[1]),  np.max(nonzero_coords[0])) # (left, top, right, bottom)

        width = bbox[2] - bbox[0]
        height= bbox[3] - bbox[1]

        dialog = SaveDialog(self.master, width, height, title="图片分辨率")
        if dialog.result == "cancel":
            return
        out_width = int(dialog.entry_values[0])
        out_height = int(dialog.entry_values[1])
        image = Image.new("RGB", (out_width, out_height))

        raw_image = self.image.crop(bbox)
        mask = mip.crop(bbox)
        mask_array = np.array(mask)
        mask_array = binary_dilation(mask_array[..., 0]!=0, iterations=mask_padding)
        for x in range(width):
            for y in range(height):
                mask_pixel = mask_array[y][x]
                if not mask_pixel:
                    raw_image.putpixel((x, y), (0, 0, 0))

        # scale
        if out_width != width or out_height != height:
            scale = min( out_width / width, out_height / height)
            width = int(width * scale)
            height = int(height * scale)
            raw_image = raw_image.resize((width, height), Image.LANCZOS)

        # move to center
        image.paste(raw_image, (int((out_width - width) / 2), int((out_height - height) / 2)))

        n_filename = self.filename.replace("\\", "/")
        initialfile = n_filename.split("/")[-1].split(".")[0] + '_' + str(self.save_counter)
        initial_dir = ""
        if self.last_save_path:
            initial_dir = '/'.join(self.last_save_path.split("/")[:-1])
        file_path = filedialog.asksaveasfilename(defaultextension=".png",  initialfile=initialfile, initialdir=initial_dir, filetypes=[("PNG 图片", "*.png"), ("所有文件", "*.*")])
        print(file_path, initialfile, self.last_save_path)
        if file_path:
            image.save(file_path)
            print(f"图片已保存为：{file_path}")
            self.save_counter += 1
            self.last_save_path = file_path
        else:
            print("未选择文件名，图片未保存。")

    def show_mask(self, mask, score, random_color=False):
        if random_color:
            color = np.concatenate([np.random.random(3), np.array([0.5])], axis=0) * 255
        else:
            src_color = np.array([30, 144, 255, 127])
            dst_color = np.array([255, 0, 0, 127])
            color = src_color * score + dst_color * (1 - score)
        h, w = mask.shape[-2:]
        mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
        mask_image = mask_image.astype(np.uint8)
        mask_image_pil = Image.fromarray(mask_image)
        mask_image_pil_resize = mask_image_pil.resize((int(self.image.width * self.canvas_scale), int(self.image.height * self.canvas_scale)), Image.LANCZOS)
        mask_image_tk = ImageTk.PhotoImage(mask_image_pil_resize)
        image_id = self.canvas.create_image(0, 0, anchor=NW, image=mask_image_tk)
        self.mask_handles.append([image_id, mask_image, mask_image_tk, mask_image_pil_resize, mask_image_pil])

    def add_point(self, event, color=None):
        # 在画布上添加一个点
        if self.image:
            if not color:
                color = self.foreColor
            x, y = event.x, event.y
            oval = self.canvas.create_oval(x-5, y-5, x+5, y+5, outline=color, width=2)
            self.points.append((x / self.canvas_scale, y / self.canvas_scale, color, oval))
            self.do_predict()
            
    def do_predict(self):
        masks = None
        scores = None
        show_highest = True
       
        input_points = [point[:2] for point in self.points]
        input_labels = [1 if point[2] == self.foreColor else 0 for point in self.points]
        for handle in self.mask_handles:
            self.canvas.delete(handle[0])
        self.mask_handles.clear()
        bbox =  (self.box_start_x, self.box_start_y, self.box_end_x, self.box_end_y) if self.box_handle else None
        if len(input_points) == 0 and not bbox:
            self.clear_points()
            return
        masks, scores, logits = predict(self.predictor, input_points, input_labels, bbox)

        if show_highest:
            self.show_mask(masks[np.argmax(scores)], 1.0)
        else:
            for i, mask in enumerate(masks):
                self.show_mask(mask, scores[i], random_color=False)

    def delete_point(self, event):
        # 在画布上删除一个点
        if self.image:
            x, y = event.x / self.canvas_scale, event.y / self.canvas_scale
            for point in self.points:
                if abs(point[0]-x) < 5 and abs(point[1]-y) < 5:
                    self.canvas.delete(point[3])
                    self.points.remove(point)
                    self.do_predict()
                    break
                
        
    def resize_canvas(self, event):
        # 当窗口大小改变时，调整画布大小
        if self.image:
            width, height = self.master.winfo_width(), self.master.winfo_height()
            scale = min(width/self.image.width, height/self.image.height)
            new_width, new_height = int(self.image.width*scale), int(self.image.height*scale)
            self.canvas.config(width=new_width, height=new_height)
            self.canvas.scale("all", 0, 0, scale, scale)
                                     
    def save_points(self):
        # 保存点坐标到文件中
        if self.image:
            filetypes = (("文本文件", "*.txt"), ("所有文件", "*.*"))
            filename = filedialog.asksaveasfilename(filetypes=filetypes)
            if filename:
                with open(filename, "w") as f:
                    for point in self.points:
                        f.write("{},{},{}\n".format(point[0], point[1], point[2]))
    
    def load_points(self):
        # 从文件中加载点坐标
        if self.image:
            filetypes = (("文本文件", "*.txt"), ("所有文件", "*.*"))
            filename = filedialog.askopenfilename(filetypes=filetypes)
            if filename:
                with open(filename, "r") as f:
                    for line in f:
                        x, y, color = line.strip().split(",")
                        x, y = int(x), int(y)
                        if color == "red":
                            oval = self.canvas.create_oval(x-5, y-5, x+5, y+5, fill="red")
                            self.points.append((x, y, color, oval))
                        elif color == "blue":
                            oval = self.canvas.create_oval(x-5, y-5, x+5, y+5, fill="blue")
                            self.points.append((x, y, color, oval))
    
if __name__ == '__main__':
    root = Tk()
    app = App(root)
    windnd.hook_dropfiles(root, func=app._on_drop)
    root.mainloop()
