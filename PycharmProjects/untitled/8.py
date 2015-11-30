from tkinter import *

class BanButtons:
    def __init__(self,master):
        frame = Frame(master)
        frame.pack()

        self.printButton = Button(frame,text="Print message", command=self.printMessage)
        self.printButton.pack(side=LEFT)
        self.quitButton = Button(frame,text="Quit",command=frame.quit)
        self.quitButton.pack(side=LEFT)

    def printMessage(self):
        print("Wow, this actually worked!")


root = Tk()
b= BanButtons(root)
root.mainloop()