from __future__ import division
import numpy as np
import numpy.testing as test
import random
import scipy
from scipy import integrate
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from particle_dicts import *
from particle_class import *

'''
# This function should randomly generate the chain of decays that will occur and print them in a nested list
# The two elements are both lists who's first element is the particles created in the collision of e+ e-
# ie if D0 and D0 bar were produced in the e+ e- collision and decayed to K- pi+ and K+ pi- respectively it would print
# [['D0', ['K-', 'pi+']], ['D0_bar', ['K+', 'pi-']]]

def decaychain():
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

    return particles
'''

def make_particle(p):
    mass, charge, life = mass_charge_lifetime(p)
    P = Particle(p,mass,charge,life)
    return P
    
def decaychain():
    particles = []
    if random.random() < .5:
        particles.append(make_particle('D0'))
        particles.append(make_particle('D0_bar'))
        if random.random()<.75:
            particles.append(make_particle('K+'))
            particles.append(make_particle('pi-'))
            particles[0].append(particles[2])
            particles[0].append(particles[3])
        else:
            particles.append(make_particle('Ks'))
            particles.append(make_particle('pi0'))
            particles[0].append(particles[2])
            particles[0].append(particles[3])
        if random.random()<.75:
            particles.append(make_particle('K+'))
            particles.append(make_particle('pi-'))
            particles[1].append(particles[4])
            particles[1].append(particles[5])
        else:
            particles.append(make_particle('Ks'))
            particles.append(make_particle('pi0'))
            particles[1].append(particles[4])
            particles[1].append(particles[5])
    else:
        particles.append(make_particle('D+'))
        particles.append(make_particle('D-'))
        if random.random()<.75:
            particles.append(make_particle('Ks'))
            particles.append(make_particle('pi+'))
            particles[0].append(particles[2])
            particles[0].append(particles[3])
        else:
            particles.append(make_particle('K+'))
            particles.append(make_particle('pi0'))
            particles[0].append(particles[2])
            particles[0].append(particles[3])
        if random.random()<.75:
            particles.append(make_particle('Ks'))
            particles.append(make_particle('pi-'))
            particles[1].append(particles[4])
            particles[1].append(particles[5])
        else:
            particles.append(make_particle('K-'))
            particles.append(make_particle('pi0'))
            particles[1].append(particles[4])
            particles[1].append(particles[5])
        
    return particles


def two_body_decay(parent,daughter1,daughter2):
    """
    This function will take a particle and make it decay to two daughter particles
    while conserving energy and momentum relativistically.
    
    Inputs
    ------
    particle  -- Initial particle we want to decay.  [Particle Object]
    duaghter1 -- First daughter particle.  [Same as above, but information like its position, velocity, etc. will be unknown]
    duaghter2 -- Second daughter particle.  [Same as above]
    
    Outputs
    -------
    duaghter1 -- First daughter particle.  [Same as above, but information like its position, velocity, etc. will now be known]
    duaghter2 -- Second daughter particle.  [Same as above]
    
    Notes
    -----

    """
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
    
    
# dx/dt = vx
# dvx/dt = 0
# dy/dt = vy
# dvy/dt = vy*q*B/m
# dz/dt = vz
# dvz/dt = vz*q*B/m

def path(particle,t,B=1):
    x,vx,y,vy,z,vz,m,q = particle[0],particle[1],particle[2],particle[3],particle[4],particle[5],particle[6],particle[7]
    return np.array([vx, 0, vy, vz*q*B/m, vz, -vy*q*B/m])

def travel(particle):
    t_array = np.linspace(0,np.min([particle.lifetime(), 1000]),10000)
    particle_path = scipy.integrate.odeint(path, particle.array(), t_array)
    particle.set_pos(particle_path[:,0], particle_path[:,2], particle_path[:,4])
    particle.set_v(particle_path[:,1], particle_path[:,3], particle_path[:,5])

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
