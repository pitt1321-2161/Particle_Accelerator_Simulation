import numpy as np
import random
import numpy.testing as test
import scipy
from scipy import integrate
import matplotlib.pyplot as plt
class Particle:

    def __init__(self,name,x,y,z,vx,vy,vz,m,q,t):
        self.__name = name
        self.__x = x 
        self.__y = y
        self.__z = z
        self.__vx = vx
        self.__vy = vy
        self.__vz = vz
        self.__v = np.sqrt(vx[0]**2+vy[0]**2+vz[0]**2)
        self.__m = float(m)
        self.__q = q 
        self.__t = float(t)
        self.__gamma = float(1/np.sqrt(1-(self.__v)**2))
        self.__px = []
        self.__py = []
        self.__pz = []
        for i in range(len(vx)):
            self.__px.append(self.__gamma * self.__m * self.__vx[i])
            self.__py.append(self.__gamma * self.__m * self.__vy[i])
            self.__pz.append(self.__gamma * self.__m * self.__vz[i])
        self.__p = np.sqrt(self.__px[0]**2+self.__py[0]**2+self.__pz[0]**2)
        
    def name(self):
        return self.__name
        
    def x(self):
        return self.__x
        
    def y(self):
        return self.__y
        
    def z(self):
        return self.__z
    
    
    def vx(self):
        return self.__vx
        
    def vy(self):
        return self.__vy
        
    def vz(self):
        return self.__vz   
        
    def v(self):
        return self.__v
        
        
    def gamma(self):
        return self.__gamma
    
    def mass(self):
        return self.__m
        
    def charge(self):
        return self.__q
        
    def lifetime(self):
        return self.__t
        
    def px(self):
        return self.__px
        
    def py(self):
        return self.__py
        
    def pz(self):
        return self.__pz
        
    def p(self):
        return self.__p

    def set_p4(self, Px, Py, Pz, E):
        pass
    
    def e(self):
        return np.sqrt(self.__p**2+self.__m**2)
    
    def array(self):
        return [self.__x[0], self.__vx[0], self.__y[0], self.__vy[0], self.__z[0], self.__vz[0], self.__m, self.__q]
    
    def __str__(self):
        return 'Particle: {}, Mass: {} GeV/c**2, Charge: {}, Lifetime: {} s, Energy: {} GeV, Momentum: {} GeV/c'.format(
            self.__name, self.__m, self.__q, self.__t, self.e(), self.__p)


from tkinter import *

# define a class for multiple pages
class ParticleAccelaration(Tk):
    def __init__(self, *args, **kwargs):
        Tk.__init__(self, *args, **kwargs)
        frame1 = Frame(self)

        frame1.pack(side="top", fill="both", expand= True)

        frame1.grid_rowconfigure(0,weight=1)
        frame1.grid_columnconfigure(0,weight=1)

        self.frames= {}

        for F in (Page1,Page2,Page3,Page4):
            frame = F(frame1,self)
            self.frames[F]= frame
            frame.grid(row=0,column=0,sticky="nswe")

        self.show_frame(Page1)

    def show_frame(self,cont):

        frame = self.frames[cont]
        frame.tkraise()

#first Page-Single Particle appendx
class Page1(Frame):
    def __init__(self, parent, controller):
        Frame.__init__(self,parent)
        # *** Entry and Label ***
        Label_0= Label(self,text="Single Particle Appendx",font=(12))
        Label_1= Label(self,text="Mass")
        Label_2= Label(self,text="Charge")
        Label_3= Label(self,text="Life Time")
        Label_4= Label(self,text="Px:")
        Label_5= Label(self,text="P:")
        Label_6= Label(self,text="V:")
        Label_7= Label(self,text="D0:")

        entry_1= Entry(self)
        entry_2= Entry(self)
        entry_3= Entry(self)

        T1= Text(self,height=1,width=60)
        T2= Text(self,height=1,width=60)
        T3= Text(self,height=1,width=60)
        T4= Text(self,height=4,width=60)

        Label_0.grid(row=1,columnspan=4,stick=W+E)
        Label_1.grid(row=2,sticky=W)
        Label_2.grid(row=3,sticky=W)
        Label_3.grid(row=4,sticky=W)
        Label_4.grid(row=7,sticky=W)
        Label_5.grid(row=8,sticky=W)
        Label_6.grid(row=9,sticky=W)
        Label_7.grid(row=10,sticky=W)

        entry_1.grid(row=2,column=1,columnspan=2,sticky=W+E)
        entry_2.grid(row=3,column=1,columnspan=2,sticky=W+E)
        entry_3.grid(row=4,column=1,columnspan=2,sticky=W+E)

        T1.grid(row=7,column=1,columnspan=3,sticky=W+E+N+S)
        T2.grid(row=8,column=1,columnspan=3,sticky=W+E+N+S)
        T3.grid(row=9,column=1,columnspan=3,sticky=W+E+N+S)
        T4.grid(row=10,column=1,columnspan=3,sticky=W+E+N+S)

        #define initial value of text
        T1.insert(END,0)
        T2.insert(END,0)
        T3.insert(END,0)
        T4.insert(END,0)

        def decay1(event):
            input_m=entry_1.get()
            input_q=entry_2.get()
            input_t=entry_3.get()
            D0 = Particle('D0',[0],[0],[0],[.1],[.1],[.1],input_m,input_q,input_t)
            T1.delete("1.0",END)
            T2.delete("1.0",END)
            T3.delete("1.0",END)
            T4.delete("1.0",END)
            T1.insert(END,D0.px())
            T2.insert(END,D0.p())
            T3.insert(END,D0.v())
            T4.insert(END,D0)
            print(D0.px())
            print(D0.p())
            print(D0.v())
            print(D0)

        button_1 = Button(self, text="Run",font=("Verdana" , 14))
        button_1.bind("<Button-1>",decay1)
        button_1.grid(row=2,column=3,columnspan=2,rowspan=3)

        button1_1 = Button(self, text="Single Particle Appendx", command=lambda:controller.show_frame(Page1))
        button1_1.grid(row=0,column=0,sticky=W)

        Button1_2 = Button(self, text="Decay Chain", command=lambda:controller.show_frame(Page2))
        Button1_2.grid(row=0,column=1,sticky=E)

        Button1_3 = Button(self, text="Two Body", command=lambda:controller.show_frame(Page3))
        Button1_3.grid(row=0,column=2,sticky=E)

        Button1_4 = Button(self, text="Pathway", command=lambda:controller.show_frame(Page4))
        Button1_4.grid(row=0,column=3,sticky=E)


import numpy as np
import random

# This function should randomly generate the chain of decays that will occur and print them in a nested list
# The two elements are both lists who's first element is the particles created in the collision of e+ e-
# ie if D0 and D0 bar were produced in the e+ e- collision and decayed to K- pi+ and K+ pi- respectively it would print
# [['D0', ['K-', 'pi+']], ['D0_bar', ['K+', 'pi-']]]


class Page2(Frame):
    def __init__(self, parent, controller):
        Frame.__init__(self,parent)
        label=Label(self, text="Decay Chain",font=(12))
        label.grid(row=1,column=0,columnspan=4,sticky=W+E)
        Label_2_1=Label(self,text="This function should randomly generate the chain of decays that will occur and print them in a nested list")
        Label_2_2=Label(self,text= "The two elements are both lists who's first element is the particles created in the collision of e+ e-")
        Label_2_3=Label(self,text="ie if D0 and D0 bar were produced in the e+ e- collision and decayed to K- pi+ and K+ pi- respectively it would print")
        Label_2_4=Label(self,text="[['D0', ['K-', 'pi+']], ['D0_bar', ['K+', 'pi-']]]")

        T_2_1= Text(self,height=1, width=60)

        Label_2_1.grid(row=2,column=1,columnspan=4,sticky=W)
        Label_2_2.grid(row=3,column=1,columnspan=4,sticky=W)
        Label_2_3.grid(row=4,column=1,columnspan=4,sticky=W)
        Label_2_4.grid(row=5,column=1,columnspan=4,sticky=W)
        T_2_1.grid(row=6,column=1,columnspan=4,sticky=W+E+N+S)

        button2_1 = Button(self, text="Single Particle Appendx", command=lambda:controller.show_frame(Page1))
        button2_1.grid(row=0,column=0,sticky=W)

        Button2_2 = Button(self, text="Decay Chain", command=lambda:controller.show_frame(Page2))
        Button2_2.grid(row=0,column=1,sticky=E)

        Button2_3 = Button(self, text="Two Body", command=lambda:controller.show_frame(Page3))
        Button2_3.grid(row=0,column=2,sticky=E)

        Button2_4 = Button(self, text="Pathway", command=lambda:controller.show_frame(Page4))
        Button2_4.grid(row=0,column=3,sticky=E)

        T_2_1.insert(END,"0")

        def decaychain(event):
            particles = []
            if random.random() < .5:
                particles.append(['D0'])
                particles.append(['D0_bar'])
                if random.random()<.75:
                    particles[0].append(['K+','pi-'])
                else:
                    particles[0].append(['K-','pi+'])
                if random.random()<.75:
                    particles[1].append(['K+','pi-'])
                else:
                    particles[1].append(['Ks','pi0'])
            else:
                particles.append(['D+'])
                particles.append(['D-'])
                if random.random()<.75:
                    particles[0].append(['Ks','pi+'])
                else:
                    particles[0].append(['K+','pi0'])
                    if random.random()<.75:
                        particles[1].append(['Ks','pi-'])
                    else:
                        particles[1].append(['K-','pi0'])

            T_2_1.delete("1.0",END)
            T_2_1.insert(END,particles)


        button_2_2 = Button(self, text="Run",font=(14))
        button_2_2.bind("<Button-1>",decaychain)
        button_2_2.grid(row=6,column=0,sticky=W+E)


#quitButton = Button(root,text="Quit",command=root.quit)
#quitButton.grid(row=5,column=1,sticky=W)

class Page3(Frame):
    def __init__(self, parent, controller):
        Frame.__init__(self,parent)
        Label_1_1= Label(self,text="Two Body decay")
        Label_1_2= Label(self,text="Pi")
        Label_1_3= Label(self,text="Km")
        Label_1_4= Label(self,text="Test result")

        Label_1_1.grid(row=1,columnspan=4)
        Label_1_2.grid(row=2)
        Label_1_3.grid(row=3)
        Label_1_4.grid(row=4)

        T_1_1= Text(self,height=2,width=80)
        T_1_2= Text(self,height=2,width=80)
        T_1_3= Text(self,height=2,width=80)

        T_1_1.grid(row=2,column=1,columnspan=4)
        T_1_2.grid(row=3,column=1,columnspan=4)
        T_1_3.grid(row=4,column=1,columnspan=4)

        T_1_1.insert(END,"0")
        T_1_2.insert(END,"0")
        T_1_3.insert(END,"0")


        def two_body_decay(parent,daughter1,daughter2):

            e_init = parent.e()
            p_init = parent.e()
            m = parent.mass()
            m1 = daughter1.mass()
            m2 = daughter2.mass()

            if m < (m1+m2):
                print('Daughter particles have greater mass than parent')
                return

            e1 = (m*m + m1*m1 - m2*m2) / (2.0*m)
            e2 = (m*m - m1*m1 + m2*m2) / (2.0*m)
            P  = np.sqrt(e1*e1 - m1*m1)

            theta = np.arccos( 2.0*random.random() - 1.0 )
            phi   = 2.0 * np.pi * random.random()

            pX = P*np.sin(theta)*np.cos(phi)
            pY = P*np.sin(theta)*np.sin(phi)
            pZ = P*np.cos(theta)

            betax1 = pX / e1
            betay1 = pY / e1
            betaz1 = pZ / e1
            beta12 = betax1*betax1 + betay1*betay1 + betaz1*betaz1
            gamma1 = 1.0/np.sqrt(1.0-beta12)

            betax2 = pX / e2
            betay2 = pY / e2
            betaz2 = pZ / e2
            beta22 = betax2*betax2 + betay2*betay2 + betaz2*betaz2
            gamma2 = 1.0/np.sqrt(1.0-beta22)

            vX1 = [pX / (m1*gamma1)]
            vY1 = [pY / (m1*gamma1)]
            vZ1 = [pZ / (m1*gamma1)]

            vX2 = [pX / (m2*gamma2)]
            vY2 = [pY / (m2*gamma2)]
            vZ2 = [pZ / (m2*gamma2)]

            X = [parent.x()[-1]]
            Y = [parent.y()[-1]]
            Z = [parent.z()[-1]]

            name1 = daughter1.name()
            name2 = daughter2.name()

            q1 = daughter1.charge()
            q2 = daughter2.charge()

            t1 = daughter1.lifetime()
            t2 = daughter2.lifetime()

            daughter1 = Particle(name1,X,Y,Z,vX1,vY1,vZ1,m1,q1,t1)
            daughter2 = Particle(name2,X,Y,Z,vX2,vY2,vZ2,m2,q2,t2)
            return parent, daughter1, daughter2


        def two_body_decay_test():
            D0 = Particle('D0',[1,5],[1,4],[1,8],[.01],[.01],[.01],3.0,0,1.1)
            pi_plus = Particle('pip',[0],[0],[0],[.01],[.01],[.01],.135,1,2.0)
            K_minus = Particle('Km',[0],[0],[0],[.01],[.01],[.01],1.3,-1,3.0)
            D0, pi_plus, K_minus = two_body_decay(D0,pi_plus,K_minus)

            test.assert_almost_equal(pi_plus.p(), K_minus.p())
            test.assert_almost_equal(pi_plus.x()[0], K_minus.x()[0], D0.x()[-1])
            test.assert_almost_equal(pi_plus.y()[0], K_minus.y()[0], D0.y()[-1])
            test.assert_almost_equal(pi_plus.z()[0], K_minus.z()[0], D0.z()[-1])

            test.assert_almost_equal(pi_plus.e(), 1.2213708333)
            test.assert_almost_equal(K_minus.e(), 1.7786291667)

            T_1_1.delete("1.0",END)
            T_1_2.delete("1.0",END)
            T_1_3.delete("1.0",END)

            T_1_1.insert(END,pi_plus)
            T_1_2.insert(END,K_minus)
            T_1_3.insert(END,"Two Body Decay Tests Passed")

            print(pi_plus)
            print(K_minus)
            print('Two Body Decay Tests Passed')

        button_3_1 = Button(self, text="decay test",command=two_body_decay_test)
        button_3_1.grid(row=5)

        button3_1 = Button(self, text="Single Particle Appendx", command=lambda:controller.show_frame(Page1))
        button3_1.grid(row=0,column=0,sticky=W)

        Button3_2 = Button(self, text="Decay Chain", command=lambda:controller.show_frame(Page2))
        Button3_2.grid(row=0,column=1,sticky=E)

        Button3_3 = Button(self, text="Two Body", command=lambda:controller.show_frame(Page3))
        Button3_3.grid(row=0,column=2,sticky=E)

        Button3_4 = Button(self, text="Pathway", command=lambda:controller.show_frame(Page4))
        Button3_4.grid(row=0,column=3,sticky=E)

class Page4(Frame):
    def __init__(self, parent, controller):
        Frame.__init__(self,parent)
        label=Label(self, text="Path Test for particle", font=(12))
        label.grid(row=1,column=1,columnspan=3,sticky=E+W)

        button4_1 = Button(self, text="Single Particle Appendx", command=lambda:controller.show_frame(Page1))
        button4_1.grid(row=0,column=0,sticky=W)

        Button4_2 = Button(self, text="Decay Chain", command=lambda:controller.show_frame(Page2))
        Button4_2.grid(row=0,column=1,sticky=E)

        Button4_3 = Button(self, text="Two Body", command=lambda:controller.show_frame(Page3))
        Button4_3.grid(row=0,column=2,sticky=E)

        Button4_4 = Button(self, text="Pathway", command=lambda:controller.show_frame(Page4))
        Button4_4.grid(row=0,column=3,sticky=E)

        def path(particle,t,B=1):
            x,vx,y,vy,z,vz,m,q = particle[0],particle[1],particle[2],particle[3],particle[4],particle[5],particle[6],particle[7]
            a = q*np.sqrt(vy**2+vz**2)*B/m
            r = np.sqrt(vy**2+vz**2)
            return np.array([vx, 0, vy, vz*q*B/m, vz,-vy*q*B/m])

        D_plus = Particle('D+',[0],[0],[0],[.01],[.01],[.01],3.0,.001,1.1)
        t_array = np.linspace(0,100000,1000000)
        D_plus_path = scipy.integrate.odeint(path, D_plus.array(), t_array)


        def plotpy():
            plt.plot(D_plus_path[:,0],D_plus_path[:,2])
            plt.show()

        def plotpz():
            plt.plot(D_plus_path[:,0],D_plus_path[:,4])
            plt.show()

        def plottot():
            plt.plot(D_plus_path[:,5],D_plus_path[:,3])
            plt.show()

        button_4_2 = Button(self, text="py",command=plotpy,font=(14))
        button_4_2.grid(row=2,column=1,sticky=E+W)

        button_4_3 = Button(self, text="px",command=plotpz,font=(14))
        button_4_3.grid(row=2,column=2,sticky=E+W)

        #Label_2_3 = Label(self, text="    ")
        #Label_2_3.grid(row=2,column=2)

        button_4_4 = Button(self, text="tot",command=plottot,font=(14))
        button_4_4.grid(row=2,column=3,sticky=W+E)

app=ParticleAccelaration()
app.mainloop()

