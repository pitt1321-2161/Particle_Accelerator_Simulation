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

test_particle_1 = Particle('test_1', 2000, 0, 100)
test_particle_1.set_pos([0],[0],[0])
test_particle_1.set_v([0],[0],[0])
test_particle_2 = Particle('test_2', 1000, 0, 1000)
test_particle_3 = Particle('test_3', 500, 0, 1000)

two_body_decay(test_particle_1, test_particle_2, test_particle_3)
travel(test_particle_2)
travel(test_particle_3)
chain = [test_particle_1, test_particle_2, test_particle_3]

fig = plt.figure()
ax = fig.gca(projection='3d')
for P in chain:
    print(P.v())
    ax.plot([P.x()[0]],[P.y()[0]],'o')
    ax.plot(P.x(), P.y(), P.z(), label=P.name())
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
plt.xlabel('x')
plt.ylabel('y')
plt.show()