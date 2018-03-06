#!/usr/bin/env python3

from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
import numpy as np
import matplotlib.pyplot as plt
from skimage import io, color

#Funcion para pasar de rgb a escala de grises tomada de
#https://stackoverflow.com/questions/12201577/how-can-i-convert-an-rgb-image-into-grayscale-in-python
def rgb2gray(rgb):
    r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
    return gray



#Definimos la funcion de segmentacion
#featureSpace : 'rgb', 'lab', 'hsv', 'rgb+xy', 'lab+xy' or 'hsv+xy'
#clusteringMethod = 'kmeans', 'gmm', 'hierarchical' or 'watershed'.
#numberOfClusters positvie integer (larger than 2)
#rgbIMage es la imagen
def segmentByClustering( rgbImage, featureSpace, clusteringMethod, numberOfClusters):
    k=numberOfClusters
    #Empezamos con rgb y kmeans
    if featureSpace=='rgb'and clusteringMethod=='kmeans':
        Imagen=rgbImage
        #Tambien obtenemos la imagen a blanco y negro
        Imagenbn=rgb2gray(Imagen)
        #Ahora vamos a representar cada pixel en el espacio rgb mas intensidad, para ello tenemos la lista vectores con todos los vectores
        vectores=[]
        for i in range(Imagen.shape[0]):
            for j in range(Imagen.shape[1]):
                #aux es el vector de representacion del pixel,va asi, r,g,b,Intensidad
                aux=[Imagen[i][j][0],Imagen[i][j][1],Imagen[i][j][2],Imagenbn[i][j]]
                vectores.append(aux)
        #Ahora que tenemos los puntos pasamos a hacer kmeans:
        kmeans = KMeans(n_clusters=k, n_init=1, max_iter=100).fit(np.array(vectores))
        #Tenemos el mapa de segmentacion,hay que dejarlo como una matriz de nuevo
        map = kmeans.labels_
        segmentation = map.reshape(Imagen.shape[0],Imagen.shape[1])



        
    #Ahora rgb con gmm
    if featureSpace=='rgb'and clusteringMethod=='gmm':
        Imagen=rgbImage
        #Tambien obtenemos la imagen a blanco y negro
        Imagenbn=rgb2gray(Imagen)
        #Ahora vamos a representar cada pixel en el espacio rgb mas intensidad, para ello tenemos la lista vectores con todos los vectores
        vectores=[]
        for i in range(Imagen.shape[0]):
            for j in range(Imagen.shape[1]):
                #aux es el vector de representacion del pixel,va asi, r,g,b,Intensidad
                aux=[Imagen[i][j][0],Imagen[i][j][1],Imagen[i][j][2],Imagenbn[i][j]]
                vectores.append(aux)
        #Ahora que tenemos los puntos pasamos a hacer gmm
        gmm=GaussianMixture(n_components=k).fit(np.array(vectores))
        #Ahora obtenemos el mapa de la imagen
        map=gmm.predict(np.array(vectores))
        segmentation = map.reshape(Imagen.shape[0],Imagen.shape[1])
        



    return segmentation
                
        
#Probemos una imagen
filename = "./BSDS_tiny/24063.jpg"
Imagen1 = io.imread(filename)
#Hagamos varios k
valoresk=[3]
for w in valoresk:
    Segmentacion1=segmentByClustering( Imagen1, 'rgb', 'gmm', w)
    np.savetxt("Segmentacion"+str(w)+".dat",Segmentacion1)


