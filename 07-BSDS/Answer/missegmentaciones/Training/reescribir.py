#!/usr/bin/env python3
import numpy as np
import scipy.io as sio
import os





def reescribir(filename,filenamedimensiones):

    segmentations=np.zeros((22,), dtype=np.object)
    Archivo=np.loadtxt(filename)
    dimensiones=np.loadtxt(filenamedimensiones)

    segmentations[0]=Archivo[0].reshape(int(dimensiones[1]),int(dimensiones[2])).astype('uint16')
    segmentations[1]=Archivo[1].reshape(int(dimensiones[1]),int(dimensiones[2])).astype('uint16')
    segmentations[2]=Archivo[2].reshape(int(dimensiones[1]),int(dimensiones[2])).astype('uint16')
    segmentations[3]=Archivo[3].reshape(int(dimensiones[1]),int(dimensiones[2])).astype('uint16')
    segmentations[4]=Archivo[4].reshape(int(dimensiones[1]),int(dimensiones[2])).astype('uint16')
    segmentations[5]=Archivo[5].reshape(int(dimensiones[1]),int(dimensiones[2])).astype('uint16')
    segmentations[6]=Archivo[6].reshape(int(dimensiones[1]),int(dimensiones[2])).astype('uint16')
    segmentations[7]=Archivo[7].reshape(int(dimensiones[1]),int(dimensiones[2])).astype('uint16')
    segmentations[8]=Archivo[8].reshape(int(dimensiones[1]),int(dimensiones[2])).astype('uint16')
    segmentations[9]=Archivo[9].reshape(int(dimensiones[1]),int(dimensiones[2])).astype('uint16')
    segmentations[10]=Archivo[10].reshape(int(dimensiones[1]),int(dimensiones[2])).astype('uint16')
    segmentations[11]=Archivo[11].reshape(int(dimensiones[1]),int(dimensiones[2])).astype('uint16')
    segmentations[12]=Archivo[12].reshape(int(dimensiones[1]),int(dimensiones[2])).astype('uint16')
    segmentations[13]=Archivo[13].reshape(int(dimensiones[1]),int(dimensiones[2])).astype('uint16')
    segmentations[14]=Archivo[14].reshape(int(dimensiones[1]),int(dimensiones[2])).astype('uint16')
    segmentations[15]=Archivo[15].reshape(int(dimensiones[1]),int(dimensiones[2])).astype('uint16')
    segmentations[16]=Archivo[16].reshape(int(dimensiones[1]),int(dimensiones[2])).astype('uint16')
    segmentations[17]=Archivo[17].reshape(int(dimensiones[1]),int(dimensiones[2])).astype('uint16')
    segmentations[18]=Archivo[18].reshape(int(dimensiones[1]),int(dimensiones[2])).astype('uint16')
    segmentations[19]=Archivo[19].reshape(int(dimensiones[1]),int(dimensiones[2])).astype('uint16')
    segmentations[20]=Archivo[20].reshape(int(dimensiones[1]),int(dimensiones[2])).astype('uint16')
    segmentations[21]=Archivo[21].reshape(int(dimensiones[1]),int(dimensiones[2])).astype('uint16')

    sio.savemat(filename, {'segs':segmentations})



os.system('ls ./gmm > nombres1.dat')
#Ahora cargamos dicha lista
with open('nombres1.dat', 'r') as myfile:
    data=myfile.readlines()
#Estons nombres tienen un /n que me molesta asi que lo voy a quitar, ademas el ultimo elemento no es una imagen
nombres=[s.replace('\n','') for s in data]
#Ahora hacemos la lista de direcciones de validation gmm
direccionesgmm=[]
for i in nombres:
    direccionesgmm.append('./gmm/'+str(i))
    

#Ahora hacemos la lista de direcciones de kmeans
direccioneskmeans=[]
for i in nombres:
    direccioneskmeans.append('./kmeans/'+str(i))
    
    
#Ahora hacemos la lista de direcciones de kmeans
direccionesdimensiones=[]
for i in nombres:
    direccionesdimensiones.append('./dimensiones/'+str(i))

cont=0.0
total=2.0*len(nombres)
for i in range(len(nombres)):
    reescribir(direccionesgmm[i],direccionesdimensiones[i])
    cont=cont+1.0
    print(cont*100.0/total)


for i in range(len(nombres)):
    reescribir(direccioneskmeans[i],direccionesdimensiones[i])
    cont=cont+1.0
    print(cont*100.0/total)
    
    
