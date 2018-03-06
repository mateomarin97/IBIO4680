#!/usr/bin/env python3

from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
import numpy as np
import matplotlib.pyplot as plt
from skimage import io, color
import cv2
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.cluster.hierarchy import fcluster
from PIL import Image


#Funcion para pasar de rgb a escala de grises tomada de
#https://stackoverflow.com/questions/12201577/how-can-i-convert-an-rgb-image-into-grayscale-in-python
def rgb2gray(rgb):
    r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
    for i in range(gray.shape[0]):
        for j in range(gray.shape[1]):
            gray[i][j]=min(255,int(gray[i][j]))
    return gray


#Funcion que expresa una imagen(de un solo canal) en una matriz de minimos locales. Devuelve otra matriz de las dimensiones de la imagen
#con 0 donde no hay minimos locales y 1 donde si
def minimolocal(A):
    matrizmi=np.zeros((A.shape[0],A.shape[1]),int)
    #Empezamos mirando la esquina superior izquierda
    if A[0][0]<A[0][1] and A[0][0]<A[1][0] and A[0][0]<A[1][1]:
        matrizmi[0][0]=1
    #Ahora vemos la esquina superior derecha
    if A[0][A.shape[1]-1]<A[0][A.shape[1]-2] and A[0][A.shape[1]-1]<A[1][A.shape[1]-1] and A[0][A.shape[1]-1]<A[1][A.shape[1]-2]:
        matrizmi[0][A.shape[1]-1]=1
    #Ahora vemos la esquina inferior izquierda
    if A[A.shape[0]-1][0]<A[A.shape[0]-1][1] and A[A.shape[0]-1][0]<A[A.shape[0]-2][0] and A[A.shape[0]-1][0]<A[A.shape[0]-2][1]:
        matrizmi[A.shape[0]-1][0]=1
    #Ahora vemos la esquina inferior derecha
    if A[A.shape[0]-1][A.shape[1]-1]<A[A.shape[0]-1][A.shape[1]-2] and A[A.shape[0]-1][A.shape[1]-1]<A[A.shape[0]-2][A.shape[1]-1] and A[A.shape[0]-1][A.shape[1]-1]<A[A.shape[0]-2][A.shape[1]-2]:
        matrizmi[A.shape[0]-1][A.shape[1]-1]=1
    #Ahora vemos la fila superior
    for j in range(1,A.shape[1]-1):
        if A[0][j]<A[0][j-1] and A[0][j]<A[0][j+1] and A[0][j]<A[1][j] and A[0][j]<A[1][j+1] and A[0][j]<A[1][j-1]:
            matrizmi[0][j]=1
    #Ahora vemos la fila inferior
    for j in range(1,A.shape[1]-1):
        if A[A.shape[0]-1][j]<A[A.shape[0]-1][j-1] and A[A.shape[0]-1][j]<A[A.shape[0]-1][j+1] and A[A.shape[0]-1][j]<A[A.shape[0]-2][j] and A[A.shape[0]-1][j]<A[A.shape[0]-2][j+1] and A[A.shape[0]-1][j]<A[A.shape[0]-2][j-1]:
            matrizmi[A.shape[0]-1][j]=1
    #Ahora la columna izquierda
    for i in range(1,A.shape[0]-1):
        if A[i][0]<A[i][1] and A[i][0]<A[i-1][0] and A[i][0]<A[i+1][0] and A[i][0]<A[i-1][1] and A[i][0]<A[i+1][1]:
            matrizmi[i][0]=1
    #Ahora la columna derecha
    for i in range(1,A.shape[0]-1):
        if A[i][A.shape[1]-1]<A[i][A.shape[1]-2] and A[i][A.shape[1]-1]<A[i-1][A.shape[1]-1] and A[i][A.shape[1]-1]<A[i+1][A.shape[1]-1] and A[i][A.shape[1]-1]<A[i-1][A.shape[1]-2] and A[i][A.shape[1]-1]<A[i+1][A.shape[1]-2]:
            matrizmi[i][A.shape[1]-1]=1
    #Ahora vemos los cuadros del medio
    for i in range(1,A.shape[0]-1):
        for j in range(1,A.shape[1]-1):
            if A[i][j]<A[i][j+1] and A[i][j]<A[i][j-1] and A[i][j]<A[i+1][j] and A[i][j]<A[i-1][j] and A[i][j]<A[i-1][j+1] and A[i][j]<A[i-1][j-1] and A[i][j]<A[i+1][j+1] and A[i][j]<A[i+1][j-1]:
                matrizmi[i][j]=1

    return np.array(matrizmi)
    
    


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


    #Ahora rgb con watershed
    if featureSpace=='rgb'and clusteringMethod=='watershed':
        #Para usar watershed importe la imagen con cv2, no io
        Imagen=rgbImage
        #Tambien obtenemos la imagen a blanco y negro
        Imagenbn= cv2.cvtColor(Imagen,cv2.COLOR_BGR2GRAY)
        #Watershed es ciego al color (en este caso), asi que solo voy a trabajar con la imagen a blanco y negro.
        #Voy a hacer el watershed tipico en el cual cada minimo local es un punto por donde empiezo a inundar, intente poner un
        #umbral para no marcar lineas divisoras de agua bajas pero no lo puede hacer en la funcion de cv2.
        #Lo primero es calcular la matriz de minimos locales de la Imagen a blanco y negro.
        Matrizminimos=minimolocal(Imagenbn)
        #Ahora creamos la mascara de marcadores
        markers=np.zeros((Imagenbn.shape[0],Imagenbn.shape[1]),np.int32)
        #Ahora alteramos la matriz de marcadores de forma tal que cada minimo local tenga un entero diferente representandolo en ella, y
        #aquellos que no son minimos locales sean ceros
        entero=1
        for i in range(Matrizminimos.shape[0]):
            for j in range(Matrizminimos.shape[1]):
                if Matrizminimos[i][j]==1:
                    markers[i][j]=entero
                    entero=entero+1
        #Ahora que tenemos la matriz de marcadores hacemos watershed
        segmentation = cv2.watershed(Imagen,markers)


    #Ahora rgb con hierarchical
    if featureSpace=='rgb'and clusteringMethod=='hierarchical':
        Imagen=rgbImage
        ancho=Imagen.shape[0]
        largo=Imagen.shape[1]
        #Debemos cortar la imagen y para ello determinamos de que dimensiones la queremos
        Npix=10
        csi0=int(ancho*0.5 +1.0 - Npix*0.5)
        csi1=int(largo*0.5 +1.0 - Npix*0.5)
        cid0=int(ancho*0.5 -1.0 +Npix*0.5)
        cid1=int(largo*0.5 -1.0 +Npix*0.5)
        Imagen=np.asarray(Image.open(filename).crop((csi0,csi1,cid0,cid1)))
        #Tambien obtenemos la imagen a blanco y negro
        Imagenbn=rgb2gray(Imagen)
        #Ahora vamos a representar cada pixel en el espacio rgb mas intensidad, para ello tenemos la lista vectores con todos los vectores
        vectores=[]
        for i in range(Imagen.shape[0]):
            for j in range(Imagen.shape[1]):
                #aux es el vector de representacion del pixel,va asi, r,g,b,Intensidad
                aux=[Imagen[i][j][0],Imagen[i][j][1],Imagen[i][j][2],Imagenbn[i][j]]
                vectores.append(aux)
        #Ahora que tenemos los datos podemos hacer la jerarquia
        jerar = linkage(vectores, 'ward')
        map=fcluster(jerar, k, criterion='maxclust')
        segmentation = map.reshape(Imagen.shape[0],Imagen.shape[1])
                
            
        
        

       
        



    return segmentation
                
        
#Probemos una imagen
filename = "./BSDS_tiny/24063.jpg"
Imagen1 = io.imread(filename)
#Imagen1 = cv2.imread(filename)
#Hagamos varios k
valoresk=[2,3]
for w in valoresk:
    Segmentacion1=segmentByClustering( Imagen1, 'rgb', 'hierarchical', w)
    np.savetxt("Segmentacion"+str(w)+".dat",Segmentacion1)
#Segmentacion1=segmentByClustering( Imagen1,'rgb' ,'watershed', 1)
#np.savetxt("Segmentacionwatershed.dat",Segmentacion1)


