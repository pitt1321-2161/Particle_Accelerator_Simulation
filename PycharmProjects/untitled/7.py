from tkinter import *

root = Tk()

def leftClick(event):
    print("left")

frame= Frame(root,width=300, height=250)
frame.bind("<Button>",leftClick)
frame.pack()

root.mainloop()