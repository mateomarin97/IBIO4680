import matplotlib.pyplot as plt
import numpy as np
a=np.array([1.0,1.0,1.0])
a=a/np.sum(a)
b=np.array([2.0,2.0,2.0])
hist=np.loadtxt("Histograma.dat")
plt.figure()
plt.hist(hist,bins=125)
plt.savefig("Histograma.jpg")
c=np.zeros((25,25))
for i in range(25):
    for j in range(25):
        c[i][j]=i
print(c[:,0])
