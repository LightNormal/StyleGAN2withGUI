
#!/usr/bin/python3
import time
import tkinter as tk
from tkinter import ttk
from tkinter.filedialog import *
import tkinter.messagebox #弹窗库
import re
import run_generator
import cv2
from PIL import Image, ImageTk
import numpy as np
import align_images
import project_images
import modify_and_show

# Create instance
win = tk.Tk()   

win.maxsize(1000,800)
# Add a title       
win.title("人脸生成程序")

tabControl = ttk.Notebook(win)          # Create Tab Control
tab1 = ttk.Frame(tabControl)            # Create a tab 
tabControl.add(tab1, text='人脸生成')      # Add the tab
tab2 = ttk.Frame(tabControl)            # Add a second tab
tabControl.add(tab2, text='人脸属性编辑')      # Make second tab visible

tab3 = ttk.Frame(tabControl)            # Add a second tab
tabControl.add(tab3, text='人脸融合')      # Make second tab visible

tabControl.pack(expand=1, fill="both")  # Pack to make visible

# LabelFrame using tab1 as the parent
mighty = ttk.LabelFrame(tab1, text=' 参数值输入 ')
mighty.grid(column=0, row=0, padx=8, pady=4)

# Modify adding a Label using mighty as the parent instead of win
a_label = ttk.Label(mighty, text="输入随机种子（正整数）:")
a_label.grid(column=0, row=0, sticky='W')

# Modified Button Click Function
def checkValue():
    print(seeds.get())
    if seeds.get().isdigit():
        if int(seeds.get())<0:
            tkinter.messagebox.showerror('错误','seeds值小于0')
            seeds.set('')
            return False
        else:
            return True
    else:
        tkinter.messagebox.showerror('错误', 'seeds不是数字')
        seeds.set('')
        return False

# Adding a Textbox Entry widget
seeds = tk.StringVar()
seeds_entered = ttk.Entry(mighty,textvariable=seeds ,width=12)
seeds_entered.grid(column=1, row=0, sticky='W',padx=10,pady=10)               # align left/West

def checknum():
    if number.get().isdigit():
        if int(number.get())<1 and int(number.get())>100:
            tkinter.messagebox.showerror('错误','请输入1-100内的整数')
            number.set('1')
            return False
        else:
            return True
    else:
        tkinter.messagebox.showerror('错误', '数量不是数字')
        number.set('1')
        return False

# Creating three checkbuttons
num_label=ttk.Label(mighty, text="生成图像数量(1-100):").grid(column=0, row=4,sticky='W')
number = tk.StringVar()
number_chosen = ttk.Entry(mighty, width=12, textvariable=number)
number_chosen.grid(column=1, row=4,sticky='W',padx=10,pady=10)

def checkfloat():
    pattern = re.compile(r'^[-+]?[-0-9]\d*\.\d*|[-+]?\.?[0-9]\d*$')
    result = pattern.match(psi.get())
    if result:
        return True
    else:
        tkinter.messagebox.showerror('错误', '截断指数不是正小数')
        psi.set('')
        return False


psi_label=ttk.Label(mighty, text="截断指数(-2.0,2.0):").grid(column=0, row=8,sticky='W')
psi = tk.StringVar()
psi_entry = ttk.Entry(mighty, width=12, textvariable=psi)
psi_entry.grid(column=1, row=8,sticky='W',padx=10,pady=10)


def check():
    if checkValue() and checknum() and checkfloat():
        run_generator.run_generate_pic(int(seeds.get()),int(number.get()),float(psi.get()))

run_button=ttk.Button(mighty, text='生成图片',command=check).grid(column=1, row=14,sticky='W',padx=10,pady=10)

#seeds_entered.focus()      # Place cursor into seeds Entry
#======================
# Start GUI
#======================
# ---------------Tab2控件介绍------------------#
# We are creating a container tab3 to hold all other widgets -- Tab2
monty2 = ttk.LabelFrame(tab2,text='文件操作')
monty2.grid(column=0, row=0, padx=8, pady=4)
# Creating three checkbuttons
viewhigh = 256
viewwide = 256

def open_file():
    global imgtk
    pic_path = askopenfilename(title='open file', filetypes=[('图像文件','.png'),('图像文件','.jpg')])
    if pic_path:
        img_bgr = imreadex(pic_path) # 读取图片
        imgtk = get_imgtk(img_bgr)
        global path
        path=pic_path
        image_ctl.configure(image=imgtk)
a_label = ttk.Button(monty2, text="打开图片文件",command=open_file)
a_label.grid(column=0, row=1, sticky='W')
image_ctl = ttk.Label(monty2)
image_ctl.grid(column=0, row=9, sticky='W')

textLabel=ttk.Label(monty2,text='图像的长宽要相等').grid(column=1, row=1, sticky='W')

def generate_letent():
    dst_dir=askdirectory(title='选择图像向量生成目录')
    if(dst_dir):
        project_images.project_file(path,dst_dir,1)
        global exist#是否生成了向量
        global filename
        temp=path.split('/')[-1]
        filename=temp.split('.')[0]
        exist=dst_dir+filename+'.npy'
        submit_button.configure(state='normal')
latent=ttk.Button(monty2, text="生成图像向量",command=generate_letent)
latent.grid(column=2, row=1, sticky='E')

ttk.Label(monty2,text='选择要修改的属性').grid(column=0,columnspan=3, row=2, sticky='W')
# Radiobutton list
values = ["年龄", "角度", "性别", "眼睛大小", "眼镜", "微笑","白皮肤","黄皮肤","黑皮肤"]

# create three Radiobuttons using one variable
radVar = tk.IntVar()

# Selecting a non-existing index value for radVar
radVar.set(99)

# Creating all three Radiobutton widgets within one loop
for col in range(3):
    # curRad = 'rad' + str(col)
    curRad = tk.Radiobutton(monty2, text=values[col], variable=radVar, value=col)
    curRad.grid(column=col, row=3, sticky=tk.W, columnspan=3)
for col in range(3, 6):
    # curRad = 'rad' + str(col)
    curRad = tk.Radiobutton(monty2, text=values[col], variable=radVar, value=col)
    curRad.grid(column=col - 3, row=4, sticky=tk.W, columnspan=3)
for col in range(6, 9):
    # curRad = 'rad' + str(col)
    curRad = tk.Radiobutton(monty2, text=values[col], variable=radVar, value=col)
    curRad.grid(column=col - 6, row=5, sticky=tk.W, columnspan=3)

def load_npy_file():
    npy_file= askopenfilename(title='open file', filetypes=[('图像向量文件', '.npy')])
    global exist
    exist=npy_file
    global filename
    str_tmp=exist.split('/')[-1]
    filename=str_tmp.split('.')[0]
    tkinter.messagebox.showinfo("载入图像隐向量",'载入成功')
    submit_button.configure(state='normal')
load_npy=ttk.Button(monty2,text="载入向量文件",command=load_npy_file)
load_npy.grid(column=0,columnspan=3, row=6, sticky='W')
#确认修改，设置参数
def to_modify():
    if radVar.get()!=99:
        global exist
        if os.path.exists(exist):
            modify_and_show.choice(radVar.get(),exist,filename)
            in_button.configure(state='normal')
        else:
            tkinter.messagebox.showerror('错误', '图像向量文件不存在！请重新生成或打开')
    else:
        tkinter.messagebox.showerror('错误', '请选择要修改的属性！')
#确定修改属性
submit_button=ttk.Button(monty2,text="修改",state='disabled',command=to_modify)
submit_button.grid(column=2,columnspan=3, row=6, sticky='W')
#设置滑动条
ttk.Label(monty2,text='选择喜欢的值').grid(column=0, row=7, sticky='W')

scale_val=tk.IntVar()
scale_in=tk.Scale(monty2,from_=-40, to=40,length=150, variable=scale_val,resolution=2,orient=tk.HORIZONTAL)
scale_in.grid(column=0,columnspan=2, row=8, sticky='W')
#scale_in.set(0)

def save():
    global exist
    if os.path.exists(exist):
        result=modify_and_show.save_img(scale_val.get())
        global img
        img_bgr= imreadex(result) # 读取图片
        img = get_imgtk(img_bgr)
        res_img.configure(image=img)
    else:
        tkinter.messagebox.showerror('错误', '图像向量文件不存在！请重新生成或打开')
in_button=ttk.Button(monty2,text="确定",state='disabled',command=save)
in_button.grid(column=2, row=8, sticky='W')
res_img=ttk.Label(monty2)
res_img.grid(column=2,row=9,sticky='W')

#读取图片文件
def imreadex(filename):
	return cv2.imdecode(np.fromfile(filename, dtype=np.uint8), cv2.IMREAD_COLOR)



def get_imgtk(img_bgr):  # 参数为背景色
    img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)  # 转换颜色
    im = Image.fromarray(img)  # 读取图像
    imgtk = ImageTk.PhotoImage(image=im)
    wide = imgtk.width()  # 宽度等于图像的宽度
    high = imgtk.height()  # 高度等于图像的高度
    if wide > viewwide or high > viewhigh:  # 如果大于窗口的大小
        wide_factor = viewwide / wide  # 获得宽度的缩放比
        high_factor = viewhigh / high  # 获得高度的缩放比
        factor = min(wide_factor, high_factor)  # 取最小值
        wide = int(wide * factor)
        if wide <= 0: wide = 1
        high = int(high * factor)
        if high <= 0: high = 1
        im = im.resize((wide, high), Image.ANTIALIAS)  # 调整大小
        imgtk = ImageTk.PhotoImage(image=im)
    return imgtk

# ---------------Tab3控件介绍------------------#
monty3 = ttk.LabelFrame(tab3,text='请选择两个图像向量作为输入')
monty3.grid(column=0, row=0, padx=8, pady=4)

def load_npy1():
    npy_file= askopenfilename(title='open file', filetypes=[('图像向量文件', '.npy')])
    global npy1,src_img1
    npy1=npy_file.split('/')[-1]
    tkinter.messagebox.showinfo("载入图像隐向量",'载入成功')
    src_img1=modify_and_show.ImageFromVec(npy1)
    src1.configure(image=src_img1)

ttk.Button(monty3,text="载入向量文件1",command=load_npy1).grid(column=0, row=1)

def load_npy2():
    npy_file= askopenfilename(title='open file', filetypes=[('图像向量文件', '.npy')])
    global npy2,src_img2
    npy2=npy_file.split('/')[-1]
    tkinter.messagebox.showinfo("载入图像隐向量",'载入成功')
    src_img2=modify_and_show.ImageFromVec(npy2)
    src2.configure(image=src_img2)

ttk.Button(monty3,text="载入向量文件2",command=load_npy2).grid(column=1, row=1)

src1=ttk.Label(monty3)
src1.grid(column=0,row=5,sticky='W')
src2=ttk.Label(monty3)
src2.grid(column=1,row=5,sticky='W')

def mix_img():
    global npy1,npy2,mix_image
    if npy1 and npy2:
        mix_image=modify_and_show.mix_pic(npy1,npy2,psi.get())
    else:
        tkinter.messagebox.showerror('错误', '请选择两个图像向量作为生成源')
    '''
    img_bgr = imreadex(result)  # 读取图片
    mix_image = get_imgtk(img_bgr)
    '''
    mix_img.configure(image=mix_image)

ttk.Label(monty3,text="psi(0-1):").grid(column=0,row=4)
psi=tk.DoubleVar()
scale_psi=tk.Scale(monty3,from_=0, to=1,length=150, variable=psi,resolution=0.05,orient=tk.HORIZONTAL)
scale_psi.grid(column=1,columnspan=2, row=4, sticky='W')
ttk.Button(monty3,text="生成混合文件",command=mix_img).grid(column=2, row=1)

mix_img=ttk.Label(monty3)
mix_img.grid(column=2,row=5,sticky='W')
win.mainloop()