#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
import os
import time
from sys import stdin,stdout
from random import shuffle
from PIL import Image

#Empiezo a contar el tiempo
tiempoi=time.time()

#Definimos el numero N de imagenes a mostrar
N=9
#Como este script siempre crea un directorio en el cual guarda las imagenes es mejor borrar dicho diretorio antes de volver a empezar
os.system(' ls -l|grep Imagenesrecortadas|wc -l > numero.dat')
w=np.loadtxt("numero.dat")
if(w!=0):
    os.system('rm -rf Imagenesrecortadas')
#Creamos el directorio en el cual vamos a guardar las imagenes recortadas
os.system('mkdir Imagenesrecortadas ')
#Guardamos en numero.dat el numero de archivos con el nombre deseado
os.system(' ls -l|grep BSR_bsds500.tgz|wc -l > numero.dat')
#Leemos este numero
x=np.loadtxt("numero.dat")
#Verificamos si el archivo no ha sido descargado
if(x==0):
   os.system(' wget www.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/BSR/BSR_bsds500.tgz')
#Guardamos en numero.dat el numero de archivos llamado BSR
os.system('ls -l|grep -w "BSR"|wc -l > numero.dat')
#Leemos este numero
y=np.loadtxt("numero.dat")
#Verificamos que el archivo no haya sido descomprimido
if(y==0):
    os.system(' tar -xzvf BSR_bsds500.tgz')
#Ahora que ya tenemos los datos vamos a guardar los nombres de las imagenes del test en una lista de la cual luego elegimos algunos al azar.
os.system('ls ./BSR/BSDS500/data/images/test > test.dat')
#Ahora cargamos dicha lista
with open('test.dat', 'r') as myfile:
    data=myfile.readlines()
#Estons nombres tienen un /n que me molesta asi que lo voy a quitar, ademas el ultimo elemento no es una imagen
nombres=[s.replace('\n','') for s in data]
nombres.pop()
#Ahora reorganizamos aleatoriamente esta lista de nombres
shuffle(nombres)
#Ahora agarramos solo los primeros N elementos de nombres
nom=[]
for i in range(N):
    nom.append(nombres[i])
#Vamos a hacer una lista que contega las iamgenes seleccionadas
imagenes=[]
for i in nom:
    imagenes.append(Image.open(r'./BSR/BSDS500/data/images/test/'+i))

#Ahora cargamos las imagenes las cortamos y guardamos
for i in range(len(imagenes)):
    imagenn=imagenes[i].resize((256,256))
    imagenn.save(r'./Imagenesrecortadas/'+nom[i])




    
#Ahora vamos a mostrar en una misma imagen las N imagenes originales y de label les voy a poner el nombre de la imagen, porque no tengo las
#imagenes segmentadas, como para ponerlas debajo de las nos segementadas.
plt.figure()
plt.subplots_adjust(hspace=0.0)
for i in range(len(imagenes)):
    plt.subplot(3,3,i+1)
    plt.imshow(np.asarray(imagenes[i]),cmap='gray')
    plt.axis('off')
    plt.text(len(np.asarray(imagenes[i])[0,:])/4.0,len(np.asarray(imagenes[i])[:,0])/2.0,nom[i],bbox=dict(facecolor='red',alpha=0.5))
plt.savefig("Collage.jpg")

#Borramos las cosas inecesarias
os.system('rm test.dat numero.dat')

#Imprimo el tiempo de ejecucion
print(u"El tiempo de ejecucion es: " + str(time.time()-tiempoi) + " segundos")
