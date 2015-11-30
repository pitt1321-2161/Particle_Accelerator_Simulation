from tkinter import *

root = Tk()

# *** Entry and Label ***

Label_1= Label(root,text="Mass")
Label_2= Label(root,text="Charge")
entry_1= Entry(root)
entry_2= Entry(root)
T1= Text(root,height=1,width=60)



T1.grid(row=6,column=1)
Label_1.grid(row=0,sticky=W)
Label_2.grid(row=1,sticky=W)

entry_1.grid(row=0,column=1)
entry_2.grid(row=1,column=1)



# *** Choice for the type of decay ***

#c = Checkbutton(root, text="D0->(k+)+(pi)")
#c.grid(row=2)

#d = Checkbutton(root, text="D0->(D-)+(D+)")
#d.grid(row=3)

#e = Checkbutton(root, text="random decay")
#e.grid(row=4)

# *** Button ***

def decay1(event):
    print(entry_1.get())
    T1.delete("1.0",END)
    T1.insert(END,entry_1.get())

def decay2(event):
    print("D0->(D-)+(D+) happens")



button_1 = Button(root, text="Run")
button_1.bind("<Button-1>",decay1)
button_1.grid(row=5)

quitButton = Button(root,text="Quit",command=lambda: controller.show_frame(PageOne))
quitButton.grid(row=5,column=1)

class PageOne(tk.frame)


root.mainloop()