import matplotlib.pyplot as plt
import numpy as np
a=np.array([1.0,1.0,1.0])
a=a/np.sum(a)
b=np.array([2.0,2.0,2.0])
hist=np.loadtxt("Histograma.dat")
plt.figure()
plt.hist(hist)
plt.savefig("Histograma.jpg")

print(np.sum(hist))
print(a)
print(a+b)
