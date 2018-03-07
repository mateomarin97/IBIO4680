import matplotlib.pyplot as plt
import numpy as np
valoresk=[2,3,4,5,6,7,8,9]

for i in valoresk:
	a=np.loadtxt("Segmentacionhsvjerarquia"+str(i)+".dat")

	plt.figure()
	plt.imshow(a,cmap='inferno')
	plt.savefig("Segmentacionhsvjerarquia"+str(i)+".jpg")
	
	a=np.loadtxt("Segmentacionhsvgmm"+str(i)+".dat")

	plt.figure()
	plt.imshow(a,cmap='inferno')
	plt.savefig("Segmentacionhsvgmm"+str(i)+".jpg")

	a=np.loadtxt("Segmentacionhsvkmeans"+str(i)+".dat")

	plt.figure()
	plt.imshow(a,cmap='inferno')
	plt.savefig("Segmentacionhsvkmeans"+str(i)+".jpg")

	
a=np.loadtxt("Segmentacionhsvwatershed.dat")
plt.figure()
plt.imshow(a,cmap='inferno')
plt.savefig("Segmentacionhsvwatershed.jpg")
