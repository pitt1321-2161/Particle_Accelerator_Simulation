from tkinter import *
import matplotlib.pyplot as plt
import numpy as np
import numpy.testing as test
import random
import scipy
from scipy import integrate
import numpy as np
import random
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class Particle:
    def __init__(self, name, m, q, t):
        self.__name = name
        self.__x = None
        self.__y = None
        self.__z = None
        self.__vx = None
        self.__vy = None
        self.__vz = None
        self.__v = None  # np.sqrt(vx[0]**2+vy[0]**2+vz[0]**2)
        self.__m = float(m)
        self.__q = q
        self.__t = float(t)
        self.__gamma = None  # float(1/np.sqrt(1-(self.__v)**2))
        self.__px = None
        self.__py = None
        self.__pz = None
        '''
        for i in range(len(vx)):
            self.__px.append(self.__gamma * self.__m * self.__vx[i])
            self.__py.append(self.__gamma * self.__m * self.__vy[i])
            self.__pz.append(self.__gamma * self.__m * self.__vz[i])
        '''
        self.__p = None  # np.sqrt(self.__px[0]**2+self.__py[0]**2+self.__pz[0]**2)
        self.__e = None  # np.sqrt(self.__p**2+self.__m**2)
        self.__daughters = []
        self.__parent = None

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

    def set_pos(self, x, y, z):
        self.__x, self.__y, self.__z = x, y, z

    '''
    def set_y(self, y):
        self.__y = y

    def set_z(self, z):
        self.__z = z
    '''

    def set_v(self, vx, vy, vz):
        self.__vx = vx
        self.__vy = vy
        self.__vz = vz
        self.__v = np.sqrt(vx[0] ** 2 + vy[0] ** 2 + vz[0] ** 2)
        self.__gamma = 1 / np.sqrt(1 - (self.__v) ** 2)
        self.__px = []
        self.__py = []
        self.__pz = []
        for i in range(len(vx)):
            self.__px.append(self.__gamma * self.__m * self.__vx[i])
            self.__py.append(self.__gamma * self.__m * self.__vy[i])
            self.__pz.append(self.__gamma * self.__m * self.__vz[i])
        self.__p = np.sqrt(self.__px[0] ** 2 + self.__py[0] ** 2 + self.__pz[0] ** 2)
        self.__e = np.sqrt(self.__p ** 2 + self.__m ** 2)

    '''
    def set_vy(self, vy):
        self.__vy = vy

    def set_vz(self, vz):
        self.__vz = vz
    '''

    def e(self):
        if self.__p or self.__p == 0:
            return np.sqrt(self.__p ** 2 + self.__m ** 2)

    def array(self):
        return [self.__x[-1], self.__vx[-1], self.__y[-1], self.__vy[-1], self.__z[-1], self.__vz[-1], self.__m,
                self.__q]

    def append(self, daughter):
        self.__daughters.append(daughter)
        daughter.append_parent(self)

    def daughter(self, num):
        if num < len(self.__daughters):
            return self.__daughters[num]

    def append_parent(self, parent):
        self.__parent = parent

    def parent(self):
        return self.__parent

    def boost(self, parent):
        '''
        Performs a Lorentz Boost into the rest frame of a particle. Use on daughters to get from C.o.M. frame to Lab frame

        Inputs
        ------
        parent  -- Particle whose rest frame we want to boost to.  [Particle Object]
        self -- Particle who is being boosted

        Outputs
        -------


        Notes
        -----

        '''

        name = self.__name
        m = self.__m
        q = self.__q
        t = self.__t

        betax = parent.px()[-1] / parent.e()
        betay = parent.py()[-1] / parent.e()
        betaz = parent.pz()[-1] / parent.e()
        gamma = parent.gamma()
        dot = betax * self.__px[-1] + betay * self.__py[-1] + betaz * self.__pz[-1]
        prod = gamma * (gamma * dot / (1.0 + gamma) + self.__e)

        pX = self.__px[-1] + betax * prod
        pY = self.__py[-1] + betay * prod
        pZ = self.__pz[-1] + betaz * prod
        e = gamma * (self.__e + dot)

        betax = pX / e
        betay = pY / e
        betaz = pZ / e
        beta2 = betax * betax + betay * betay + betaz * betaz
        self.__gamma = 1.0 / np.sqrt(1.0 - beta2)

        self.__px = [pX]
        self.__py = [pY]
        self.__pz = [pZ]

        self.__vx = [pX / (m * self.__gamma)]
        self.__vy = [pY / (m * self.__gamma)]
        self.__vz = [pZ / (m * self.__gamma)]

        self.__x = [parent.x()[-1]]
        self.__y = [parent.y()[-1]]
        self.__z = [parent.z()[-1]]

        self.__p = np.sqrt(self.__px[0] ** 2 + self.__py[0] ** 2 + self.__pz[0] ** 2)
        self.__e = e

    def __str__(self):
        if self.e() and self.__p:
            return 'Particle: {}, Mass: {} GeV/c**2, Charge: {}, Lifetime: {} fs, Energy: {} MeV, Momentum: {} MeV/c'.format(
                self.__name, self.__m, self.__q, self.__t, self.e(), self.__p)
        else:
            return 'Particle: {}, Mass: {} GeV/c**2, Charge: {}, Lifetime: {} fs'.format(
                self.__name, self.__m, self.__q, self.__t)


mass_dict = {'D0': 1864.84, 'D0_bar': 1864.84, 'D+': 1869.61, 'D-': 1869.61,
             'Ks': 497.611, 'Kl': 497.611, 'K+': 493.677, 'K-': 493.677,
             'pi0': 134.9766, 'pi+': 139.57018, 'pi-': 139.57018,
             'rho': 775.26, 'rho+': 775.26, 'rho-': 775.26,
             'e+': .510999, 'e-': .510999, 'e+e-': 5000, 'gamma': 1*10**-5,
             'mu': 105.6584, 'mu+': 105.6584, 'mu-': 105.6584}

# In 10^-15 s
life_dict = {'D0': 410.1, 'D0_bar': 410.1, 'D+': 1040, 'D-': 1040,
             'Ks': 89540, 'Kl': 5.116*10**7 , 'K+': 1.238*10**7, 'K-': 1.238*10**7,
             'pi0': .0852, 'pi+': 2.6033*10**7, 'pi-': 2.6033*10**7,
             'rho': 4.5*10**-9, 'rho+': 4.5*10**-9, 'rho-': 4.5*10**-9,
             'e+': 1*10**100 , 'e-': 1*10**100, 'e+e-': 1*10**-5, 'gamma': 10**100,
             'mu': 2.197*10**9, 'mu+': 2.197*10**9, 'mu-': 2.197*10**9}

decay_dict = {'e+e-': [[0,.5,1], ['D0','D0_bar','D+','D-']],
              'D0': [[0,.25,.5,.75,1], ['K-','rho+','K-','pi+','Ks','pi0','rho+','pi-']],
              'D0_bar': [[0,.25,.5,.75,1], ['K+','rho-','K+','pi-','Ks','pi0','rho-','pi+']],
              'D+': [[0,.25,.5,.75,1], ['Ks','pi+','Kl','pi+','Ks','rho+','K+','pi0']],
              'D-': [[0,.25,.5,.75,1], ['Ks','pi-','Kl','pi-','Ks','rho-','K-','pi0']],
              'Ks': [[0,.5,1], ['pi+','pi-','pi0','pi0']],
              'Kl': [[0,.5,1], ['pi+','pi-','pi0','pi0']],
              'pi0':[[0,1], ['gamma', 'gamma']],
              'rho': [[0,1], ['pi+','pi-']],
              'rho+': [[0,1], ['pi0', 'pi+']],
              'rho-': [[0,1], ['pi0', 'pi+']]}


def mass_charge_lifetime(particle_name):
    h_bar = 6.58 * 10**-7
    average_mass = mass_dict[str(particle_name)]
    average_life = life_dict[str(particle_name)]
    err_mass = h_bar/average_life
    err_life = h_bar/average_mass
    rand_mass = np.random.standard_cauchy()
    rand_life = np.random.standard_cauchy()
    mass = average_mass + err_mass * rand_mass
    life = average_life + err_life * rand_life
    if '-' in particle_name:
        q = -1
    elif '+' in particle_name:
        q = 1
    else:
        q = 0
    return mass, q, life

def make_particle(p):
    mass, charge, life = mass_charge_lifetime(p)
    P = Particle(p,mass,charge,life)
    return P

def make_decay(p, particles, pos):
    rand = np.random.rand()
    for i in range(len(decay_dict[p][0])):
        if decay_dict[p][0][i] < rand <= decay_dict[p][0][i+1]:
            particles.append(make_particle(decay_dict[p][1][i*2]))
            particles.append(make_particle(decay_dict[p][1][i*2+1]))
            particles[pos].append(particles[-2])
            particles[pos].append(particles[-1])




def two_body_decay(parent,daughter1,daughter2):

    import numpy as np
    import random

    m = parent.mass()
    m1 = daughter1.mass()
    m2 = daughter2.mass()

    # Check if decay is possible
    if m < (m1+m2):
        print('Daughter particles have greater mass than parent')
        return

    # C.o.M. Frame energies and momenta
    e1 = (m*m + m1*m1 - m2*m2) / (2.0*m)
    e2 = (m*m - m1*m1 + m2*m2) / (2.0*m)
    P  = np.sqrt(e1*e1 - m1*m1)

    # Get angles
    theta = np.arccos( 2.0*random.random() - 1.0 )
    phi   = 2.0 * np.pi * random.random()

    # Calculate Momenta
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

    # Calculate Velocity from momentum
    vX1 = [pX / (m1*gamma1)]
    vY1 = [pY / (m1*gamma1)]
    vZ1 = [pZ / (m1*gamma1)]

    vX2 = [-pX / (m2*gamma2)]
    vY2 = [-pY / (m2*gamma2)]
    vZ2 = [-pZ / (m2*gamma2)]

    X = [parent.x()[-1]]
    Y = [parent.y()[-1]]
    Z = [parent.z()[-1]]

    daughter1.set_v(vX1,vY1,vZ1)
    daughter2.set_v(vX2,vY2,vZ2)

    daughter1.set_pos(X,Y,Z)
    daughter2.set_pos(X,Y,Z)

    daughter1.boost(parent)
    daughter2.boost(parent)

def path(particle,t,B=1):
    x,vx,y,vy,z,vz,m,q = particle[0],particle[1],particle[2],particle[3],particle[4],particle[5],particle[6],particle[7]
    return np.array([vx, 0, vy, vz*q*B/m, vz, -vy*q*B/m])

def travel(particle):
    t_array = np.linspace(0,np.min([particle.gamma() * particle.lifetime(), 2000]),10000)
    particle_path = scipy.integrate.odeint(path, particle.array(), t_array)
    particle.set_pos(particle_path[:,0], particle_path[:,2], particle_path[:,4])
    particle.set_v(particle_path[:,1], particle_path[:,3], particle_path[:,5])




D_plus = Particle('D+',1864.84,1,410.1)
D_plus.set_pos([0],[0],[0])
D_plus.set_v([.01],[.01],[.01])
D_minus = Particle('D-',1864.84,-1,410.1)
D_minus.set_pos([0],[0],[0])
D_minus.set_v([.01],[.01],[.01])
K_plus = Particle('K+',493.677,1,1.238*10**7)
K_plus.set_pos([0],[0],[0])
K_plus.set_v([.01],[.01],[.01])
K_minus = Particle('K-',493.677,1,-1.238*10**7)
K_minus.set_pos([0],[0],[0])
K_minus.set_v([.01],[.01],[.01])
pi_plus = Particle('Pi+',139.57,1,2.6033*10**7)
pi_plus.set_pos([0],[0],[0])
pi_plus.set_v([.01],[.01],[.01])
pi_minus = Particle('Pi-',139.57,-1,2.6033*10**7)
pi_minus.set_pos([0],[0],[0])
pi_minus.set_v([.01],[.01],[.01])
rho_plus = Particle('rho+',775.26,1,4.5*10**-9)
rho_plus.set_pos([0],[0],[0])
rho_plus.set_v([.01],[.01],[.01])
rho_minus = Particle('rho-',775.26,-1,4.5*10**-9)
rho_minus.set_pos([0],[0],[0])
rho_minus.set_v([.01],[.01],[.01])
mu_plus = Particle('mu+',105.6584,1,2.197*10**9)
mu_plus.set_pos([0],[0],[0])
mu_plus.set_v([.01],[.01],[.01])
mu_minus = Particle('mu-',105.6584,-1,2.197*10**9)
mu_minus.set_pos([0],[0],[0])
mu_minus.set_v([.01],[.01],[.01])






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
        Label_0= Label(self,text="Particle Accelerator Simulation",font=(12))
        Label_0.grid(row=0,columnspan=6,stick=W+E)

        Label_space_0=Label(self,text="    ")
        Label_space_0.grid(row=1)

        w_0=Canvas(self,width=900,height=10)
        w_0.grid(row=2,column=0,columnspan=7,sticky=W+E)

        w_0.create_line(0,10,900,10,dash=(4,4))

        Label_0_1=Label(self,text="Decay Chain")
        Label_0_1.grid(row=2,column=0)

        photo=PhotoImage(file="one.gif")
        label_image=Label(self, image=photo)
        label_image.image=photo
        label_image.grid(row=3,column=5,columnspan=2,rowspan=4)

        Label_1=Label(self,text="This simulator should randomly generate the chain of decays that will occur and print them in a nested list")
        Label_2=Label(self,text= "The two elements are both lists who's first element is the particles created in the collision of e+ e-")
        Label_3=Label(self,text="ie if D0 and D0 bar were produced in the e+ e- collision and decayed to K- pi+ and K+ pi- respectively it would print")
        Label_4=Label(self,text="[initial electron][parent1][parent2][daughter1_1][daughter1_2][daughter2_1][daughter2_2][gammaN..]")

        Label_1.grid(row=3,column=0,columnspan=5,sticky=W)
        Label_2.grid(row=4,column=0,columnspan=5,sticky=W)
        Label_3.grid(row=5,column=0,columnspan=5,sticky=W)
        Label_4.grid(row=6,column=0,columnspan=5,sticky=W)

        Label_space_1=Label(self,text="     ")
        Label_space_1.grid(row=7,column=0,columnspan=4,sticky=W)

        T_2_1= Text(self,height=2, width=60)
        T_2_1.grid(row=8,column=1,columnspan=4,rowspan=3,sticky=W+E+N+S)
        T_2_1.insert(END,"0")

        def decaychain():
            particles = []
            rand = np.random.rand(10)
            particles.append(make_particle('e+e-'))
            particles[0].set_pos([0],[0],[0])
            particles[0].set_v([0],[0],[0])
            i = 0
            while True:
                if particles[i].lifetime() < 2000:
                    make_decay(particles[i].name(), particles, i)
                i += 1
                if i > len(particles) - 1:
                    break
            return particles


        def run_decaychain():
            T_2_1.delete("1.0",END)
            A=decaychain()
            for i in A:
                T_2_1.insert(END,[i.name()],)

        button_1 = Button(self, text="Decay Chain",command=run_decaychain,font=("Verdana" ,12))
        button_1.grid(row=9,column=0,rowspan=2)

        w_1=Canvas(self,width=900,height=10)
        w_1.grid(row=11,column=0,columnspan=6,sticky=W+E)

        w_1.create_line(0,10,900,10,dash=(4,4))

        Label_0_2=Label(self,text="2-D Plot")
        Label_0_2.grid(row=11,column=0)

        Label_9=Label(self,text="     ")
        Label_9.grid(row=15,column=0,columnspan=4,sticky=W)

        Label_6=Label(self,text="Particles",font=(11))
        Label_6.grid(row=16,column=0)


        T_2_2= Text(self,height=2, width=60)
        T_2_2.grid(row=16,column=1,columnspan=4,rowspan=2,sticky=W+E+N+S)
        T_2_2.insert(END,"0")

        def plotxy(a):
            travel(a)
            T_2_2.delete("1.0",END)
            T_2_2.insert(END,a)
            plt.plot(a.x(),a.y())
            plt.title("x-y")
            plt.show()

        def plotxz(a):
            plt.plot(a.x(),a.z())
            plt.title("x-z")
            plt.show()

        def plotyz(a):
            plt.plot(a.y(),a.z())
            plt.title("y-z")
            plt.show()

        def D_plus_plot():
            plotxy(D_plus)
            plotxz(D_plus)
            plotyz(D_plus)

        def D_minus_plot():
            plotxy(D_minus)
            plotxz(D_minus)
            plotyz(D_minus)

        def K_plus_plot():
            plotxy(K_plus)
            plotxz(K_plus)
            plotyz(K_plus)

        def K_minus_plot():
            plotxy(K_minus)
            plotxz(K_minus)
            plotyz(K_minus)

        def pi_plus_plot():
            plotxy(pi_plus)
            plotxz(pi_plus)
            plotyz(pi_plus)

        def pi_minus_plot():
            plotxy(pi_minus)
            plotxz(pi_minus)
            plotyz(pi_minus)

        def rho_plus_plot():
            plotxy(rho_plus)
            plotxz(rho_plus)
            plotyz(rho_plus)

        def rho_minus_plot():
            plotxy(rho_minus)
            plotxz(rho_minus)
            plotyz(rho_minus)

        def mu_plus_plot():
            plotxy(mu_plus)
            plotxz(mu_plus)
            plotyz(mu_plus)

        def mu_minus_plot():
            plotxy(mu_minus)
            plotxz(mu_minus)
            plotyz(mu_minus)

        button_4_2 = Button(self, text="D+",command=D_plus_plot,font=("Verdana" ,14))
        button_4_2.grid(row=12,column=1,sticky=W)

        button_4_3 = Button(self, text="K+",command=K_plus_plot,font=("Verdana" ,14))
        button_4_3.grid(row=12,column=2,sticky=W)

        button_4_4 = Button(self, text="pi+",command=pi_plus_plot,font=("Verdana" ,14))
        button_4_4.grid(row=12,column=3,sticky=W)

        button_4_5 = Button(self, text="rho+",command=rho_plus_plot,font=("Verdana" ,14))
        button_4_5.grid(row=12,column=4,sticky=W)

        button_4_6 = Button(self, text="mu+",command=mu_plus_plot,font=("Verdana" ,14))
        button_4_6.grid(row=12,column=5,sticky=W)

        Label_9=Label(self,text="     ")
        Label_9.grid(row=13,column=0,columnspan=4,sticky=W)

        button_4_2 = Button(self, text="D-",command=D_minus_plot,font=("Verdana" ,14))
        button_4_2.grid(row=14,column=1,sticky=W)

        button_4_3 = Button(self, text="K-",command=K_minus_plot,font=("Verdana" ,14))
        button_4_3.grid(row=14,column=2,sticky=W)

        button_4_4 = Button(self, text="pi-",command=pi_minus_plot,font=("Verdana" ,14))
        button_4_4.grid(row=14,column=3,sticky=W)

        button_4_5 = Button(self, text="rho-",command=rho_minus_plot,font=("Verdana" ,14))
        button_4_5.grid(row=14,column=4,sticky=W)

        button_4_6 = Button(self, text="mu-",command=mu_minus_plot,font=("Verdana" ,14))
        button_4_6.grid(row=14,column=5,sticky=W)

        Label_9=Label(self,text="     ")
        Label_9.grid(row=15,column=0,columnspan=4,sticky=W)

        w_2=Canvas(self,width=900,height=10)
        w_2.grid(row=18,column=0,columnspan=6,sticky=W+E)

        w_2.create_line(0,10,900,10,dash=(4,4))

        Label_0_3=Label(self,text="3-D Plot")
        Label_0_3.grid(row=18,column=0)

        T_2_3= Text(self,height=16, width=60)
        T_2_3.grid(row=19,column=1,columnspan=5,rowspan=8,sticky=W+E+N+S)
        T_2_3.insert(END,"0")

        def main():
            chain = decaychain()
            T_2_3.delete("1.0",END)
            for P in chain:
                if not P.parent():
                    two_body_decay(P,P.daughter(0),P.daughter(1))

                else:
                    travel(P)
                    if P.daughter(0):
                        two_body_decay(P,P.daughter(0),P.daughter(1))
                    print(P.name(), P.v(), P.px()[0], P.py()[0], P.pz()[0])

            fig = plt.figure()
            ax = fig.gca(projection='3d')
            for P in chain:
                ax.plot([P.x()[0]],[P.y()[0]],[P.z()[0]],'o')
                ax.plot(P.x(), P.y(), P.z(), label=P.name())
            ax.plot([200, -200], [0, 0], [0, 0])
            plt.legend(bbox_to_anchor=(1.05, 1), loc=1, borderaxespad=0.)
            for P in chain[1:]:
                print(P)
                T_2_3.insert(END,P)
                T_2_3.insert(END,"\n")
            plt.show()


        button_1 = Button(self, text="Plot",command=main,font=("Verdana" , 14))
        button_1.grid(row=19,column=0,columnspan=1,rowspan=1)

class Page2(Frame):
    def __init__(self, parent, controller):
        Frame.__init__(self,parent)


        button2_1 = Button(self, text="Single Particle Appendx", command=lambda:controller.show_frame(Page1))
        button2_1.grid(row=0,column=0,sticky=W)

        Button2_2 = Button(self, text="Decay Chain", command=lambda:controller.show_frame(Page2))
        Button2_2.grid(row=0,column=1,sticky=E)

        Button2_3 = Button(self, text="Two Body", command=lambda:controller.show_frame(Page3))
        Button2_3.grid(row=0,column=2,sticky=E)

        Button2_4 = Button(self, text="Pathway", command=lambda:controller.show_frame(Page4))
        Button2_4.grid(row=0,column=3,sticky=E)




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

        #button_3_1 = Button(self, text="decay test",command=two_body_decay_test)
        #button_3_1.grid(row=5)

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


app=ParticleAccelaration()
app.mainloop()

