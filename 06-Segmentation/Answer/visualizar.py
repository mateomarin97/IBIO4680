import matplotlib.pyplot as plt
import numpy as np
a=np.loadtxt("Segmentacion1.dat")

plt.figure()
plt.imshow(a,cmap='inferno')
plt.savefig("Segmentacion1.jpg")
