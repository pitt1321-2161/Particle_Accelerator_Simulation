from tkinter import *
import numpy as np
import matplotlib.pyplot as plt

root = Tk()

canvas= Canvas(root, width=200, height=100)
canvas.pack()
#t=np.linspace(0,100,1000)
#y=np.sin(t)
#plt.plot(t,y)

blackline = canvas.create_line(0,0,200,50)
redline = canvas.create_line(0,100,200,50,fill = "red")
green = canvas.create_rectangle(25,25,200,50,fill="green")

root.mainloop()