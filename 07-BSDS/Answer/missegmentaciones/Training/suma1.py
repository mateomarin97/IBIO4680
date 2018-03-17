#!/usr/bin/env python3
import numpy as np
import scipy.io as sio
import os

def suma1(filename):
    dic=sio.loadmat(filename)
    seg = dic['segs']
    nseg=seg.shape[1]
    nfilas=seg[0][0].shape[0]
    ncolumnas=seg[0][0].shape[1]
    for k in range(nseg):
        mini=100
        for i in range(nfilas):
            for j in range(ncolumnas):
                mini=min(seg[0][k][i][j],mini)
        for i in range(nfilas):
            for j in range(ncolumnas):
                seg[0][k][i][j]=seg[0][k][i][j]-mini+1
                
    
    segmentations=np.zeros((22,), dtype=np.object)
    for k in range(nseg):
        segmentations[k]=seg[0][k].astype('uint16')
    sio.savemat(filename,{'segs':segmentations})


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
    

cont=0.0
total=2.0*len(nombres)
for i in range(len(nombres)):
    suma1(direccionesgmm[i])
    cont=cont+1.0
    print(cont*100.0/total)


for i in range(len(nombres)):
    suma1(direccioneskmeans[i])
    cont=cont+1.0
    print(cont*100.0/total)
