from tkinter import *

def doNothing():
    print("I won't ...")

root = Tk()

# *** Main Menu ***

menu = Menu(root)
root.config(menu=menu)

subMenu = Menu(menu)
menu.add_cascade(label="File", menu=subMenu)
subMenu.add_command(label="Now Project...",command=doNothing)
subMenu.add_command(label="Now",command=doNothing)
subMenu.add_separator()
subMenu.add_command(label="Exit", command=quit)

editMenu = Menu(menu)
menu.add_cascade(label="Edit", menu=editMenu)
editMenu.add_command(label="Redo", command=doNothing)

# *** the Toolbar ***

toolbar = Frame(root,bg="blue")
insertButt = Button(toolbar, text="Insert Image", command=doNothing)
insertButt.pack(side=LEFT, padx=2, pady=2)
printButt = Button(toolbar, text="print", command=doNothing)
printButt.pack(side=LEFT, padx=2, pady=2)

toolbar.pack(side=TOP, fill=X)

# *** Status Bar ***

status = Label(root, text="Do nothing...", bd=1, relief = SUNKEN, anchor=W)
status.pack(side=BOTTOM, fill=X)


root.mainloop()