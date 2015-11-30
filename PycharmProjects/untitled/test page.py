import scipy
from scipy import integrate
import matplotlib.pyplot as plt
import numpy as np

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

D0 = Particle('D0',[0],[0],[0],[.1],[.1],[.1],3.0,0,1.1)
print(D0.px())
print(D0.p())
print(D0.v())
print(D0)

def path(particle,t,B=1):
    x,vx,y,vy,z,vz,m,q = particle[0],particle[1],particle[2],particle[3],particle[4],particle[5],particle[6],particle[7]
    # force on charged particle = q*vperp*B
    # Our B field is in the x direction
    a = q*np.sqrt(vy**2+vz**2)*B/m
    # th = np.arctan(vz/vy)
    r = np.sqrt(vy**2+vz**2)
    return np.array([vx, 0, vy, vz*q*B/m, vz,-vy*q*B/m])

D_plus = Particle('D+',[0],[0],[0],[.01],[.01],[.01],3.0,.001,1.1)
#print(D0.x())
t_array = np.linspace(0,10000,100000)

D_plus_path = scipy.integrate.odeint(path, D_plus.array(), t_array)

#plt.plot(t_array, D0_path[:,0])

plt.plot(D_plus_path[:,0],D_plus_path[:,2])
#plt.show()
plt.plot(D_plus_path[:,0],D_plus_path[:,4])
#plt.show()
plt.plot(D_plus_path[:,5],D_plus_path[:,3])
plt.show()
