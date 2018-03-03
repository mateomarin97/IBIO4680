import matplotlib.pyplot as plt
import numpy as np
#ACA de los bosques
a=[4.919999999999998264e-01,5.280000000000000249e-01,5.639999999999999458e-01,5.839999999999999636e-01,5.879999999999999671e-01,5.799999999999999600e-01,5.760000000000000675e-01,5.639999999999999458e-01,5.879999999999999671e-01,5.800000000000000711e-01,5.999999999999999778e-01,6.480000000000000204e-01,6.360000000000000098e-01]
#ACA de los vecinos
c=[2.920000000000000373e-01,3.200000000000000067e-01,3.559999999999999276e-01,3.720000000000001084e-01,3.920000000000001261e-01,3.720000000000001084e-01,3.760000000000000009e-01,3.920000000000000151e-01,3.880000000000001226e-01,3.960000000000000187e-01,3.840000000000000635e-01,3.840000000000000080e-01,3.880000000000000671e-01,]
b=[25,50,75,90,100,125,140,145,150,155,160,175,200]
plt.figure()
plt.scatter(b,a,label="Bosque")
plt.scatter(b,c,label="Vecino",color="k")
plt.xlabel("# textones")
plt.ylabel("ACA")
plt.legend()
plt.savefig("GraficoarbolesACAVsk.jpg")

#ACA de los bosques con 80 pixeles
a.pop()
d=a
#ACA de los bosques con 90 pixeles
e=[4.719999999999999751e-01,5.760000000000000675e-01,5.679999999999999494e-01,6.119999999999999885e-01,5.799999999999999600e-01,6.239999999999998881e-01,6.279999999999997806e-01,6.039999999999999813e-01,5.880000000000000782e-01,6.079999999999999849e-01,6.359999999999998987e-01,6.159999999999999920e-01]
#Como no hay para 200 debemos quitarlo de b
b.pop()

plt.figure()
plt.scatter(b,d,label="80 X 80")
plt.scatter(b,e,label="90 X 90",color="k")
plt.xlabel("# textones")
plt.ylabel("ACA")
plt.legend(loc='upper left')
plt.savefig("Graficoarbolescomparacionpixeles_Arboles=100_Pro=100.jpg")

#ACA K=155,Pi=90,Pro=100 numero de arboles cambia
f=[6.319999999999998952e-01,6.079999999999998739e-01,6.079999999999999849e-01,6.239999999999998881e-01]
#Valores del numero de aroles
nar=[50,75,100,125]

plt.figure()
plt.scatter(nar,f,label="Datos Bosque")
plt.xlabel("# arboles")
plt.ylabel("ACA")
plt.legend(loc='best')
plt.savefig("Graficoarbolescomparanumeroarboles_k=155_Pi=90_Pro=100.jpg")


#ACA k=155,Pi=90,Arb=100 profundidad cambia
g=[6.159999999999999920e-01,6.039999999999998703e-01,6.079999999999999849e-01,5.959999999999998632e-01]

plt.figure()
plt.scatter(nar,g,label="Datos Bosque")
plt.xlabel("Profundidad")
plt.ylabel("ACA")
plt.legend(loc='best')
plt.savefig("Graficoarbolescomparaprofundidad_k=155_Pi=90_Ar=100.jpg")
