import numpy as np
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
class Two_Body(Tk):
    def __init__(self, *args, **kwargs):
        Tk.__init__(self, *args, **kwargs)
        frame1 = Frame(self)

        frame1.pack(side="top", fill="both", expand= True)

        frame1.grid_rowconfigure(0,weight=1)
        frame1.grid_columnconfigure(0,weight=1)

        self.frames= {}

        for F in (Page1,Page2):
            frame = F(frame1,self)
            self.frames[F]= frame
            frame.grid(row=0,column=0,sticky="nswe")

        self.show_frame(Page1)

    def show_frame(self,cont):

        frame = self.frames[cont]
        frame.tkraise()

import numpy as np
import random
import numpy.testing as test
#first Page-Single Particle appendx
class Page1(Frame):
    def __init__(self, parent, controller):
        Frame.__init__(self,parent)
        Label_1_1= Label(self,text="Two Body decay")
        Label_1_2= Label(self,text="Pi")
        Label_1_3= Label(self,text="Km")
        Label_1_4= Label(self,text="Test result")

        Label_1_1.grid(row=1,columnspan=3)
        Label_1_2.grid(row=2)
        Label_1_3.grid(row=3)
        Label_1_4.grid(row=4)

        T_1_1= Text(self,height=2,width=80)
        T_1_2= Text(self,height=2,width=80)
        T_1_3= Text(self,height=2,width=80)

        T_1_1.grid(row=2,column=1)
        T_1_2.grid(row=3,column=1)
        T_1_3.grid(row=4,column=1)

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

        button_1_1 = Button(self, text="decay test",command=two_body_decay_test)
        button_1_1.grid(row=5)

        Button_1_2 = Button(self, text="path", command=lambda:controller.show_frame(Page2))
        Button_1_2.grid(row=0,column=0,sticky=W+E)



class Page2(Frame):
    def __init__(self, parent, controller):
        Frame.__init__(self,parent)
        label=Label(self, text="Path Test for particle", font=(12))
        label.grid(row=1,column=1,columnspan=3,sticky=E+W)

        button4 = Button(self, text="Two body test", command=lambda:controller.show_frame(Page1))
        button4.grid(row=0)

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

        button_2_2 = Button(self, text="py",command=plotpy,font=(14))
        button_2_2.grid(row=2,column=1,sticky=E+W)

        button_2_3 = Button(self, text="px",command=plotpz,font=(14))
        button_2_3.grid(row=2,column=2,sticky=E+W)

        #Label_2_3 = Label(self, text="    ")
        #Label_2_3.grid(row=2,column=2)

        button_2_4 = Button(self, text="tot",command=plottot,font=(14))
        button_2_4.grid(row=2,column=3,sticky=W+E)

app=Two_Body()
app.mainloop()
