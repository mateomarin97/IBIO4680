import matplotlib.pyplot as plt
import numpy as np
valoresk=[2,3,4,5,6,7,8,9]

for i in valoresk:
	a=np.loadtxt("Segmentacionhsvxyjerarquia"+str(i)+".dat")

	plt.figure()
	plt.imshow(a,cmap='inferno')
	plt.savefig("Segmentacionhsvxyjerarquia"+str(i)+".jpg")
	plt.close()
	
	a=np.loadtxt("Segmentacionhsvxygmm"+str(i)+".dat")

	plt.figure()
	plt.imshow(a,cmap='inferno')
	plt.savefig("Segmentacionhsvxygmm"+str(i)+".jpg")
	plt.close()

	a=np.loadtxt("Segmentacionhsvxykmeans"+str(i)+".dat")

	plt.figure()
	plt.imshow(a,cmap='inferno')
	plt.savefig("Segmentacionhsvxykmeans"+str(i)+".jpg")
	plt.close()





	a=np.loadtxt("Segmentacionrgbxyjerarquia"+str(i)+".dat")

	plt.figure()
	plt.imshow(a,cmap='inferno')
	plt.savefig("Segmentacionrgbxyjerarquia"+str(i)+".jpg")
	plt.close()
	a=np.loadtxt("Segmentacionrgbxygmm"+str(i)+".dat")

	plt.figure()
	plt.imshow(a,cmap='inferno')
	plt.savefig("Segmentacionrgbxygmm"+str(i)+".jpg")
	plt.close()
	a=np.loadtxt("Segmentacionrgbxykmeans"+str(i)+".dat")

	plt.figure()
	plt.imshow(a,cmap='inferno')
	plt.savefig("Segmentacionrgbxykmeans"+str(i)+".jpg")
	plt.close()




	a=np.loadtxt("Segmentacionlabxyjerarquia"+str(i)+".dat")

	plt.figure()
	plt.imshow(a,cmap='inferno')
	plt.savefig("Segmentacionlabxyjerarquia"+str(i)+".jpg")
	plt.close()
	a=np.loadtxt("Segmentacionlabxygmm"+str(i)+".dat")

	plt.figure()
	plt.imshow(a,cmap='inferno')
	plt.savefig("Segmentacionlabxygmm"+str(i)+".jpg")
	plt.close()
	a=np.loadtxt("Segmentacionlabxykmeans"+str(i)+".dat")

	plt.figure()
	plt.imshow(a,cmap='inferno')
	plt.savefig("Segmentacionlabxykmeans"+str(i)+".jpg")
	plt.close()


	
a=np.loadtxt("Segmentacionhsvxywatershed.dat")
plt.figure()
plt.imshow(a,cmap='inferno')
plt.savefig("Segmentacionhsvxywatershed.jpg")
plt.close()



a=np.loadtxt("Segmentacionrgbxywatershed.dat")
plt.figure()
plt.imshow(a,cmap='inferno')
plt.savefig("Segmentacionrgbxywatershed.jpg")
plt.close()

a=np.loadtxt("Segmentacionlabxywatershed.dat")
plt.figure()
plt.imshow(a,cmap='inferno')
plt.savefig("Segmentacionlabxywatershed.jpg")
plt.close()
