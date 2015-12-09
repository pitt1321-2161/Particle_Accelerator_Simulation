from tkinter import *
from PIL import Image, ImageTk
root=Tk()
imagePath=r"one.png"
photo=PhotoImage(file=imagePath)
label=Label(root,image=photo)
label.grid(row=1,column=1)

root.mainloop()