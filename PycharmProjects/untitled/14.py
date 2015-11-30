from tkinter import *

root=Tk()

photo=PhotoImage(file="big.jpg")
label = Label(root, image=photo)
label.pack()

root.mainloop()