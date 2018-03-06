import matplotlib.pyplot as plt
import numpy as np
valoresk=[2,3]
for i in valoresk:
	a=np.loadtxt("Segmentacion"+str(i)+".dat")

	plt.figure()
	plt.imshow(a,cmap='inferno')
	plt.savefig("Segmentacion"+str(i)+".jpg")
