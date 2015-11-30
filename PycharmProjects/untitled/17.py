from tkinter import *

LARGE_FONT = ("Verdana" , 12)

class SeaofBTCapp(Tk):
    def __init__(self, *args, **kwargs):
        Tk.__init__(self, *args, **kwargs)
        frame1 = Frame(self)

        frame1.pack(side="top", fill="both", expand= True)

        frame1.grid_rowconfigure(0,weight=1)
        frame1.grid_columnconfigure(0,weight=1)

        self.frames= {}

        for F in (StartPage, Page1,Page2):
            frame = F(frame1,self)
            self.frames[F]= frame
            frame.grid(row=0,column=0,sticky="nswe")

        self.show_frame(StartPage)

    def show_frame(self,cont):

        frame = self.frames[cont]
        frame.tkraise()

class StartPage(Frame):
    def __init__(self, parent, controller):
        Frame.__init__(self,parent)
        label=Label(self, text="StartPage", font=LARGE_FONT)
        label.pack(pady=10, padx=10)

        button = Button(self, text="Visit Page 1", command=lambda:controller.show_frame(Page1))
        button.pack()

        button2 = Button(self, text="Visit Page 2", command=lambda:controller.show_frame(Page2))
        button2.pack()

        button3 = Button(self, text="Visit Page 1", command=lambda:controller.show_frame(Page1))
        button3.pack()

class Page1(Frame):
    def __init__(self, parent, controller):
        Frame.__init__(self,parent)
        label=Label(self, text="Page1", font=LARGE_FONT)
        label.pack(pady=10, padx=10)

        button1 = Button(self, text="Back to Home", command=lambda:controller.show_frame(StartPage))
        button1.pack()

class Page2(Frame):
    def __init__(self, parent, controller):
        Frame.__init__(self,parent)
        label=Label(self, text="Page2", font=LARGE_FONT)
        label.pack(pady=10, padx=10)

        button4 = Button(self, text="Back to Home", command=lambda:controller.show_frame(StartPage))
        button4.pack()

app=SeaofBTCapp()
app.mainloop()