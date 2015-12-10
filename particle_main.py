from __future__ import division
import numpy as np
import numpy.testing as test
import random
import scipy
from scipy import integrate
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from particle_class import Particle as Particle
from particle_functions import *

def main():
    chain = decaychain()
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
    ax.plot([500, -500], [0, 0], [0, 0])
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    print('Particle \tMass (MeV/c**2) \tCharge (e) \tLifetime (fs) \tEnergy (MeV) \tMomentum (MeV/c)')
    for P in chain[1:]:
        print('{}  \t\t{:.6g}  \t\t\t{} \t\t\t{:.6g} \t {:.6g} \t{:.6g}'.format(
                P.name(), P.mass(), P.charge(), P.lifetime(), P.e(), P.p()))
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()
    
    
main()

