# from manipulator_environment import *
# import numpy as np
# env = Planar_Environment()
# env.reset()
# for i in range(5):
#     env.step(np.array([0.1, 0.1]))
#     fig, axs = plt.subplots(2,1)


#     env.viz_arm()


import numpy as np
import matplotlib.pyplot as plt
from manipulator_environment import *


env = Planar_Environment()
env.reset()
vals = []
plt.ion()
for i in range(20):
    vals.append(i)
    env.step(np.array([0.1, -0.1]))

    full_fig, axs = plt.subplots(2,1)

    axs[0].plot(vals)
    env.viz_arm(axs[1])

    inp = input()
    if inp.lower() == 'exit':
        break
    
    plt.close()

plt.ioff()
plt.close('all')
