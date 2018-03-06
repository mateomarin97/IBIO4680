import matplotlib.pyplot as plt
import numpy as np
valoresk=[2,3,4,5,6,7,8,9]
for i in valoresk:
	a=np.loadtxt("Segmentacionlabkmeans"+str(i)+".dat")

	plt.figure()
	plt.imshow(a,cmap='inferno')
	plt.savefig("Segmentacionlabgmm"+str(i)+".jpg")
	
	
a=np.loadtxt("Segmentacionlabwatershed.dat")
plt.figure()
plt.imshow(a,cmap='inferno')
plt.savefig("Segmentacionlabwatershed.jpg")
