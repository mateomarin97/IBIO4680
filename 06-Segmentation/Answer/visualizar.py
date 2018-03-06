import matplotlib.pyplot as plt
import numpy as np
valoresk=[2,4,5,6,7,8,9,10]
for i in valoresk:
	a=np.loadtxt("Segmentacion"+str(i)+".dat")

	plt.figure()
	plt.imshow(a,cmap='inferno')
	plt.savefig("Segmentacion"+str(i)+".jpg")
