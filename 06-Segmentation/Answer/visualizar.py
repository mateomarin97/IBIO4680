import matplotlib.pyplot as plt
import numpy as np
valoresk=[1]
for i in valoresk:
	a=np.loadtxt("Segmentacionwatershed.dat")

	plt.figure()
	plt.imshow(a,cmap='inferno')
	plt.savefig("Segmentacionwatershed.jpg")
