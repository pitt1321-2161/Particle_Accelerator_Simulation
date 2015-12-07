from __future__ import division
import numpy as np


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


