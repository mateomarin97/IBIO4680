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
import os


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
        return segmentation


        
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
        return segmentation

    #Ahora rgb con hierarchical
    if featureSpace=='rgb'and clusteringMethod=='hierarchical':
        Imagen=rgbImage
        ancho=Imagen.shape[0]
        largo=Imagen.shape[1]
        #Debemos cortar la imagen y para ello determinamos de que dimensiones la queremos
        Npix=300
        csi0=int(0)
        csi1=int(0)
        cid0=int(min(Npix,ancho))
        cid1=int(min(Npix,largo))
        Imagen=np.asarray(Image.fromarray(Imagen).crop((csi0,csi1,cid0,cid1)))
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

	#Me parece apropiado que el metodo de jerarquia tambien devuelva la jerarquia
        return jerar, segmentation
    
                
            

    #Ahora kmeans con lab

    if featureSpace=='lab'and clusteringMethod=='kmeans':
        Imagen=color.rgb2lab(rgbImage)
        #El primer canal viene siendo la imagen a blanco y negro, va de 0 a 100 asi que cuando la vaya a cargar en la lista de vectores
        #la multiplico por 255/100=2.55 para que vaya de cero a 255
        #El segundo canal es el a que va de -128 a 127, asi que cuando lo ponga en la lista de vectores le sumo 127 para que vaya de 0 a 255
        #El tercero es el b tambien va de -128 a 127.
        #Ahora vamos a representar cada pixel en el espacio rgb mas intensidad, para ello tenemos la lista vectores con todos los vectores
        vectores=[]
        for i in range(Imagen.shape[0]):
            for j in range(Imagen.shape[1]):
                #aux es el vector de representacion del pixel,va asi, r,g,b,Intensidad
                aux=[int(Imagen[i][j][0]*2.55),Imagen[i][j][1]+128,Imagen[i][j][2]+128]
                vectores.append(aux)
        #Ahora que tenemos los puntos pasamos a hacer kmeans:
        kmeans = KMeans(n_clusters=k, n_init=1, max_iter=100).fit(np.array(vectores))
        #Tenemos el mapa de segmentacion,hay que dejarlo como una matriz de nuevo
        map = kmeans.labels_
        segmentation = map.reshape(Imagen.shape[0],Imagen.shape[1])
        return segmentation


    #Ahora lab con gmm
    if featureSpace=='lab'and clusteringMethod=='gmm':
        Imagen=color.rgb2lab(rgbImage)
        #El primer canal viene siendo la imagen a blanco y negro, va de 0 a 100 asi que cuando la vaya a cargar en la lista de vectores
        #la multiplico por 255/100=2.55 para que vaya de cero a 255
        #El segundo canal es el a que va de -128 a 127, asi que cuando lo ponga en la lista de vectores le sumo 127 para que vaya de 0 a 255
        #El tercero es el b tambien va de -128 a 127.
        #Ahora vamos a representar cada pixel en el espacio rgb mas intensidad, para ello tenemos la lista vectores con todos los vectores
        vectores=[]
        for i in range(Imagen.shape[0]):
            for j in range(Imagen.shape[1]):
                #aux es el vector de representacion del pixel,va asi, r,g,b,Intensidad
                aux=[int(Imagen[i][j][0]*2.55),Imagen[i][j][1]+128,Imagen[i][j][2]+128]
                vectores.append(aux)
        #Ahora que tenemos los puntos pasamos a hacer gmm
        gmm=GaussianMixture(n_components=k).fit(np.array(vectores))
        #Ahora obtenemos el mapa de la imagen
        map=gmm.predict(np.array(vectores))
        segmentation = map.reshape(Imagen.shape[0],Imagen.shape[1])
        return segmentation


    #Ahora lab con watershed
    if featureSpace=='lab'and clusteringMethod=='watershed':
        #Para usar watershed importe la imagen con cv2, no io
        Imagen=cv2.cvtColor(rgbImage, cv2.COLOR_BGR2LAB)
        #En este caso cv2 ya escala los canales L,A,B para ir de 0 a 255
        #Hacemos la imagen de luminosidad
        Imagenbn=Imagen[:,:,0]
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
        return segmentation
        

       


    #Ahora lab con hierarchical
    if featureSpace=='lab'and clusteringMethod=='hierarchical':
        Imagen=rgbImage
        #El primer canal viene siendo la imagen a blanco y negro, va de 0 a 100 asi que cuando la vaya a cargar en la lista de vectores
        #la multiplico por 255/100=2.55 para que vaya de cero a 255
        #El segundo canal es el a que va de -128 a 127, asi que cuando lo ponga en la lista de vectores le sumo 127 para que vaya de 0 a 255
        #El tercero es el b tambien va de -128 a 127.
        ancho=Imagen.shape[0]
        largo=Imagen.shape[1]
        #Debemos cortar la imagen y para ello determinamos de que dimensiones la queremos
        Npix=300
        csi0=int(0)
        csi1=int(0)
        cid0=int(min(Npix,ancho))
        cid1=int(min(Npix,largo))
        Imagen=np.asarray(Image.fromarray(Imagen).crop((csi0,csi1,cid0,cid1)))
        Imagen=color.rgb2lab(Imagen)
        #Ahora vamos a representar cada pixel en el espacio lab , para ello tenemos la lista vectores con todos los vectores
        vectores=[]
        for i in range(Imagen.shape[0]):
            for j in range(Imagen.shape[1]):
                #aux es el vector de representacion del pixel,va asi, r,g,b,Intensidad
                aux=[int(Imagen[i][j][0]*2.55),Imagen[i][j][1]+128,Imagen[i][j][2]+128]
                vectores.append(aux)
        #Ahora que tenemos los datos podemos hacer la jerarquia
        jerar = linkage(vectores, 'ward')
        map=fcluster(jerar, k, criterion='maxclust')
        segmentation = map.reshape(Imagen.shape[0],Imagen.shape[1])

	#Me parece apropiado que el metodo de jerarquia tambien devuelva la jerarquia
        return jerar, segmentation

    #Empezamos con hsv y kmeans
    if featureSpace=='hsv'and clusteringMethod=='kmeans':
        Imagen=color.rgb2hsv(rgbImage)
        #Los canales HSV, van de 0 a 1, los multiplico por 255 para pasar de 0 a 255
        #Ahora vamos a representar cada pixel en el espacio hsv , para ello tenemos la lista vectores con todos los vectores, ademas
        #quiero que H y V sean mas importantes que S, para eso hago las distancias en S mas grandes para que no las vea de a mucho
        factorescalaS=30
        vectores=[]
        for i in range(Imagen.shape[0]):
            for j in range(Imagen.shape[1]):
                #aux es el vector de representacion del pixel,va asi, r,g,b,Intensidad
                aux=[Imagen[i][j][0]*255,Imagen[i][j][1]*255*factorescalaS,Imagen[i][j][2]*255]
                vectores.append(aux)
        #Ahora que tenemos los puntos pasamos a hacer kmeans:
        kmeans = KMeans(n_clusters=k, n_init=1, max_iter=100).fit(np.array(vectores))
        #Tenemos el mapa de segmentacion,hay que dejarlo como una matriz de nuevo
        map = kmeans.labels_
        segmentation = map.reshape(Imagen.shape[0],Imagen.shape[1])
        return segmentation



    #Ahora hsv con gmm
    if featureSpace=='hsv'and clusteringMethod=='gmm':
        Imagen=color.rgb2hsv(rgbImage)
        #Los canales HSV, van de 0 a 1, los multiplico por 255 para pasar de 0 a 255
        #Ahora vamos a representar cada pixel en el espacio hsv , para ello tenemos la lista vectores con todos los vectores, ademas
        #quiero que H y V sean mas importantes que S, para eso hago las distancias en S mas grandes para que no las vea de a mucho
        factorescalaS=30
        vectores=[]
        for i in range(Imagen.shape[0]):
            for j in range(Imagen.shape[1]):
                #aux es el vector de representacion del pixel,va asi, r,g,b,Intensidad
                aux=[Imagen[i][j][0]*255,Imagen[i][j][1]*255*factorescalaS,Imagen[i][j][2]*255]
                vectores.append(aux)
        #Ahora que tenemos los puntos pasamos a hacer gmm
        gmm=GaussianMixture(n_components=k).fit(np.array(vectores))
        #Ahora obtenemos el mapa de la imagen
        map=gmm.predict(np.array(vectores))
        segmentation = map.reshape(Imagen.shape[0],Imagen.shape[1])
        return segmentation


    #Ahora hsv con watershed
    if featureSpace=='hsv'and clusteringMethod=='watershed':
        #Para usar watershed importe la imagen con cv2, no io
        Imagen=cv2.cvtColor(rgbImage, cv2.COLOR_BGR2HSV)
        #En este caso cv2 hace que H vaya de 0 a 179, S y V de 0 a 255
        #Hacemos la imagen de luminosidad
        Imagenbn=Imagen[:,:,2]
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
        return segmentation


    #Ahora hsv con hierarchical
    if featureSpace=='hsv'and clusteringMethod=='hierarchical':
        Imagen=rgbImage
        #Los canales HSV, van de 0 a 1, los multiplico por 255 para pasar de 0 a 255
        ancho=Imagen.shape[0]
        largo=Imagen.shape[1]
        #Debemos cortar la imagen y para ello determinamos de que dimensiones la queremos
        Npix=300
        csi0=int(0)
        csi1=int(0)
        cid0=int(min(Npix,ancho))
        cid1=int(min(Npix,largo))
        Imagen=np.asarray(Image.fromarray(Imagen).crop((csi0,csi1,cid0,cid1)))
        Imagen=color.rgb2hsv(Imagen)
        #Ahora vamos a representar cada pixel en el espacio hsv , para ello tenemos la lista vectores con todos los vectores
        factorescalaS=30
        vectores=[]
        for i in range(Imagen.shape[0]):
            for j in range(Imagen.shape[1]):
                #aux es el vector de representacion del pixel,va asi, r,g,b,Intensidad
                aux=[Imagen[i][j][0]*255,Imagen[i][j][1]*255*factorescalaS,Imagen[i][j][2]*255]
                vectores.append(aux)
        #Ahora que tenemos los datos podemos hacer la jerarquia
        jerar = linkage(vectores, 'ward')
        map=fcluster(jerar, k, criterion='maxclust')
        segmentation = map.reshape(Imagen.shape[0],Imagen.shape[1])

	#Me parece apropiado que el metodo de jerarquia tambien devuelva la jerarquia
        return jerar, segmentation


    ######################################################################################################################################
    #####################################################################################################################################
    ####################################################################################################################################
    ###################################################################################################################################
    ##################################################################################################################################
    #################################################################################################################################
    ################################################################################################################################
    ###############################################################################################################################
    ##############################################################################################################################
    #############################################################################################################################
    ############################################################################################################################
    #Empezamos con rgb+xy y kmeans
    if featureSpace=='rgb+xy'and clusteringMethod=='kmeans':
        Imagen=rgbImage
        #Tambien obtenemos la imagen a blanco y negro
        Imagenbn=rgb2gray(Imagen)
        #Ahora vamos a representar cada pixel en el espacio rgb mas intensidad, para ello tenemos la lista vectores con todos los vectores
        vectores=[]
        #En el siguiente for la i es la coordenada x que va de 0 a Imagen.shape[0]-1, algo similar pasa con la componente y y j. Quiero que
        #vayan de 0 a 255, entonces la componente x la multiplico por 255/Imagen.shape[0]-1, mientras que a y la multiplico por
        #255/Imagen.shape[1]-1. Ahora, quiero darle un poco mas de peso a las coordenadas x,y que a las demas entonces defino unos factores
        #de escala que haga que se junten un poco para que los metodos tenga un bias hacia ellas.
        factorcartesiano=0.7
        for i in range(Imagen.shape[0]):
            for j in range(Imagen.shape[1]):
                #aux es el vector de representacion del pixel,va asi, r,g,b,Intensidad,x,y
                aux=[Imagen[i][j][0],Imagen[i][j][1],Imagen[i][j][2],Imagenbn[i][j],i*(255/(Imagen.shape[0]-1))*factorcartesiano,j*(255/(Imagen.shape[1]-1))*factorcartesiano]
                vectores.append(aux)
        #Ahora que tenemos los puntos pasamos a hacer kmeans:
        kmeans = KMeans(n_clusters=k, n_init=1, max_iter=100).fit(np.array(vectores))
        #Tenemos el mapa de segmentacion,hay que dejarlo como una matriz de nuevo
        map = kmeans.labels_
        segmentation = map.reshape(Imagen.shape[0],Imagen.shape[1])
        return segmentation


        
    #Ahora rgb con gmm
    if featureSpace=='rgb+xy'and clusteringMethod=='gmm':
        Imagen=rgbImage
        #Tambien obtenemos la imagen a blanco y negro
        Imagenbn=rgb2gray(Imagen)
        #Ahora vamos a representar cada pixel en el espacio rgb mas intensidad, para ello tenemos la lista vectores con todos los vectores
        vectores=[]
        factorcartesiano=0.7
        for i in range(Imagen.shape[0]):
            for j in range(Imagen.shape[1]):
                #aux es el vector de representacion del pixel,va asi, r,g,b,Intensidad
                aux=[Imagen[i][j][0],Imagen[i][j][1],Imagen[i][j][2],Imagenbn[i][j],i*(255/(Imagen.shape[0]-1))*factorcartesiano,j*(255/(Imagen.shape[1]-1))*factorcartesiano]
                vectores.append(aux)
        #Ahora que tenemos los puntos pasamos a hacer gmm
        gmm=GaussianMixture(n_components=k).fit(np.array(vectores))
        #Ahora obtenemos el mapa de la imagen
        map=gmm.predict(np.array(vectores))
        segmentation = map.reshape(Imagen.shape[0],Imagen.shape[1])
        return segmentation

    #Ahora rgb con watershed
    if featureSpace=='rgb+xy'and clusteringMethod=='watershed':
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
        return segmentation

    #Ahora rgb con hierarchical
    if featureSpace=='rgb+xy'and clusteringMethod=='hierarchical':
        Imagen=rgbImage
        ancho=Imagen.shape[0]
        largo=Imagen.shape[1]
        #Debemos cortar la imagen y para ello determinamos de que dimensiones la queremos
        Npix=300
        csi0=int(0)
        csi1=int(0)
        cid0=int(min(Npix,ancho))
        cid1=int(min(Npix,largo))
        Imagen=np.asarray(Image.fromarray(Imagen).crop((csi0,csi1,cid0,cid1)))
        #Tambien obtenemos la imagen a blanco y negro
        Imagenbn=rgb2gray(Imagen)
        #Ahora vamos a representar cada pixel en el espacio rgb mas intensidad, para ello tenemos la lista vectores con todos los vectores
        vectores=[]
        factorcartesiano=0.7
        for i in range(Imagen.shape[0]):
            for j in range(Imagen.shape[1]):
                #aux es el vector de representacion del pixel,va asi, r,g,b,Intensidad
                aux=[Imagen[i][j][0],Imagen[i][j][1],Imagen[i][j][2],Imagenbn[i][j],i*(255/(Imagen.shape[0]-1))*factorcartesiano,j*(255/(Imagen.shape[1]-1))*factorcartesiano]
                vectores.append(aux)
        #Ahora que tenemos los datos podemos hacer la jerarquia
        jerar = linkage(vectores, 'ward')
        map=fcluster(jerar, k, criterion='maxclust')
        segmentation = map.reshape(Imagen.shape[0],Imagen.shape[1])

	#Me parece apropiado que el metodo de jerarquia tambien devuelva la jerarquia
        return jerar, segmentation
    
                
            

    #Ahora kmeans con lab

    if featureSpace=='lab+xy'and clusteringMethod=='kmeans':
        Imagen=color.rgb2lab(rgbImage)
        #El primer canal viene siendo la imagen a blanco y negro, va de 0 a 100 asi que cuando la vaya a cargar en la lista de vectores
        #la multiplico por 255/100=2.55 para que vaya de cero a 255
        #El segundo canal es el a que va de -128 a 127, asi que cuando lo ponga en la lista de vectores le sumo 127 para que vaya de 0 a 255
        #El tercero es el b tambien va de -128 a 127.
        #Ahora vamos a representar cada pixel en el espacio rgb mas intensidad, para ello tenemos la lista vectores con todos los vectores
        vectores=[]
        factorcartesiano=0.7
        for i in range(Imagen.shape[0]):
            for j in range(Imagen.shape[1]):
                #aux es el vector de representacion del pixel,va asi, r,g,b,Intensidad
                aux=[int(Imagen[i][j][0]*2.55),Imagen[i][j][1]+128,Imagen[i][j][2]+128,i*(255/(Imagen.shape[0]-1))*factorcartesiano,j*(255/(Imagen.shape[1]-1))*factorcartesiano]
                vectores.append(aux)
        #Ahora que tenemos los puntos pasamos a hacer kmeans:
        kmeans = KMeans(n_clusters=k, n_init=1, max_iter=100).fit(np.array(vectores))
        #Tenemos el mapa de segmentacion,hay que dejarlo como una matriz de nuevo
        map = kmeans.labels_
        segmentation = map.reshape(Imagen.shape[0],Imagen.shape[1])
        return segmentation


    #Ahora lab con gmm
    if featureSpace=='lab+xy'and clusteringMethod=='gmm':
        Imagen=color.rgb2lab(rgbImage)
        #El primer canal viene siendo la imagen a blanco y negro, va de 0 a 100 asi que cuando la vaya a cargar en la lista de vectores
        #la multiplico por 255/100=2.55 para que vaya de cero a 255
        #El segundo canal es el a que va de -128 a 127, asi que cuando lo ponga en la lista de vectores le sumo 127 para que vaya de 0 a 255
        #El tercero es el b tambien va de -128 a 127.
        #Ahora vamos a representar cada pixel en el espacio rgb mas intensidad, para ello tenemos la lista vectores con todos los vectores
        vectores=[]
        factorcartesiano=0.7
        for i in range(Imagen.shape[0]):
            for j in range(Imagen.shape[1]):
                #aux es el vector de representacion del pixel,va asi, r,g,b,Intensidad
                aux=[int(Imagen[i][j][0]*2.55),Imagen[i][j][1]+128,Imagen[i][j][2]+128,i*(255/(Imagen.shape[0]-1))*factorcartesiano,j*(255/(Imagen.shape[1]-1))*factorcartesiano]
                vectores.append(aux)
        #Ahora que tenemos los puntos pasamos a hacer gmm
        gmm=GaussianMixture(n_components=k).fit(np.array(vectores))
        #Ahora obtenemos el mapa de la imagen
        map=gmm.predict(np.array(vectores))
        segmentation = map.reshape(Imagen.shape[0],Imagen.shape[1])
        return segmentation


    #Ahora lab con watershed
    if featureSpace=='lab+xy'and clusteringMethod=='watershed':
        #Para usar watershed importe la imagen con cv2, no io
        Imagen=cv2.cvtColor(rgbImage, cv2.COLOR_BGR2LAB)
        #En este caso cv2 ya escala los canales L,A,B para ir de 0 a 255
        #Hacemos la imagen de luminosidad
        Imagenbn=Imagen[:,:,0]
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
        return segmentation
        

       


    #Ahora lab con hierarchical
    if featureSpace=='lab+xy'and clusteringMethod=='hierarchical':
        Imagen=rgbImage
        #El primer canal viene siendo la imagen a blanco y negro, va de 0 a 100 asi que cuando la vaya a cargar en la lista de vectores
        #la multiplico por 255/100=2.55 para que vaya de cero a 255
        #El segundo canal es el a que va de -128 a 127, asi que cuando lo ponga en la lista de vectores le sumo 127 para que vaya de 0 a 255
        #El tercero es el b tambien va de -128 a 127.
        ancho=Imagen.shape[0]
        largo=Imagen.shape[1]
        #Debemos cortar la imagen y para ello determinamos de que dimensiones la queremos
        Npix=300
        csi0=int(0)
        csi1=int(0)
        cid0=int(min(Npix,ancho))
        cid1=int(min(Npix,largo))
        Imagen=np.asarray(Image.fromarray(Imagen).crop((csi0,csi1,cid0,cid1)))
        Imagen=color.rgb2lab(Imagen)
        #Ahora vamos a representar cada pixel en el espacio lab , para ello tenemos la lista vectores con todos los vectores
        vectores=[]
        factorcartesiano=0.7
        for i in range(Imagen.shape[0]):
            for j in range(Imagen.shape[1]):
                #aux es el vector de representacion del pixel,va asi, r,g,b,Intensidad
                aux=[int(Imagen[i][j][0]*2.55),Imagen[i][j][1]+128,Imagen[i][j][2]+128,i*(255/(Imagen.shape[0]-1))*factorcartesiano,j*(255/(Imagen.shape[1]-1))*factorcartesiano]
                vectores.append(aux)
        #Ahora que tenemos los datos podemos hacer la jerarquia
        jerar = linkage(vectores, 'ward')
        map=fcluster(jerar, k, criterion='maxclust')
        segmentation = map.reshape(Imagen.shape[0],Imagen.shape[1])

	#Me parece apropiado que el metodo de jerarquia tambien devuelva la jerarquia
        return jerar, segmentation

    #Empezamos con hsv y kmeans
    if featureSpace=='hsv+xy'and clusteringMethod=='kmeans':
        Imagen=color.rgb2hsv(rgbImage)
        #Los canales HSV, van de 0 a 1, los multiplico por 255 para pasar de 0 a 255
        #Ahora vamos a representar cada pixel en el espacio hsv , para ello tenemos la lista vectores con todos los vectores, ademas
        #quiero que H y V sean mas importantes que S, para eso hago las distancias en S mas grandes para que no las vea de a mucho
        factorescalaS=30
        vectores=[]
        factorcartesiano=0.7
        for i in range(Imagen.shape[0]):
            for j in range(Imagen.shape[1]):
                #aux es el vector de representacion del pixel,va asi, r,g,b,Intensidad
                aux=[Imagen[i][j][0]*255,Imagen[i][j][1]*255*factorescalaS,Imagen[i][j][2]*255,i*(255/(Imagen.shape[0]-1))*factorcartesiano,j*(255/(Imagen.shape[1]-1))*factorcartesiano]
                vectores.append(aux)
        #Ahora que tenemos los puntos pasamos a hacer kmeans:
        kmeans = KMeans(n_clusters=k, n_init=1, max_iter=100).fit(np.array(vectores))
        #Tenemos el mapa de segmentacion,hay que dejarlo como una matriz de nuevo
        map = kmeans.labels_
        segmentation = map.reshape(Imagen.shape[0],Imagen.shape[1])
        return segmentation



    #Ahora hsv con gmm
    if featureSpace=='hsv+xy'and clusteringMethod=='gmm':
        Imagen=color.rgb2hsv(rgbImage)
        #Los canales HSV, van de 0 a 1, los multiplico por 255 para pasar de 0 a 255
        #Ahora vamos a representar cada pixel en el espacio hsv , para ello tenemos la lista vectores con todos los vectores, ademas
        #quiero que H y V sean mas importantes que S, para eso hago las distancias en S mas grandes para que no las vea de a mucho
        factorescalaS=30
        vectores=[]
        factorcartesiano=0.7
        for i in range(Imagen.shape[0]):
            for j in range(Imagen.shape[1]):
                #aux es el vector de representacion del pixel,va asi, r,g,b,Intensidad
                aux=[Imagen[i][j][0]*255,Imagen[i][j][1]*255*factorescalaS,Imagen[i][j][2]*255,i*(255/(Imagen.shape[0]-1))*factorcartesiano,j*(255/(Imagen.shape[1]-1))*factorcartesiano]
                vectores.append(aux)
        #Ahora que tenemos los puntos pasamos a hacer gmm
        gmm=GaussianMixture(n_components=k).fit(np.array(vectores))
        #Ahora obtenemos el mapa de la imagen
        map=gmm.predict(np.array(vectores))
        segmentation = map.reshape(Imagen.shape[0],Imagen.shape[1])
        return segmentation


    #Ahora hsv con watershed
    if featureSpace=='hsv+xy'and clusteringMethod=='watershed':
        #Para usar watershed importe la imagen con cv2, no io
        Imagen=cv2.cvtColor(rgbImage, cv2.COLOR_BGR2HSV)
        #En este caso cv2 hace que H vaya de 0 a 179, S y V de 0 a 255
        #Hacemos la imagen de luminosidad
        Imagenbn=Imagen[:,:,2]
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
        return segmentation


    #Ahora hsv con hierarchical
    if featureSpace=='hsv+xy'and clusteringMethod=='hierarchical':
        Imagen=rgbImage
        #Los canales HSV, van de 0 a 1, los multiplico por 255 para pasar de 0 a 255
        ancho=Imagen.shape[0]
        largo=Imagen.shape[1]
        #Debemos cortar la imagen y para ello determinamos de que dimensiones la queremos
        Npix=300
        csi0=int(0)
        csi1=int(0)
        cid0=int(min(Npix,ancho))
        cid1=int(min(Npix,largo))
        Imagen=np.asarray(Image.fromarray(Imagen).crop((csi0,csi1,cid0,cid1)))
        Imagen=color.rgb2hsv(Imagen)
        #Ahora vamos a representar cada pixel en el espacio hsv , para ello tenemos la lista vectores con todos los vectores
        factorescalaS=30
        vectores=[]
        factorcartesiano=0.7
        for i in range(Imagen.shape[0]):
            for j in range(Imagen.shape[1]):
                #aux es el vector de representacion del pixel,va asi, r,g,b,Intensidad
                aux=[Imagen[i][j][0]*255,Imagen[i][j][1]*255*factorescalaS,Imagen[i][j][2]*255,i*(255/(Imagen.shape[0]-1))*factorcartesiano,j*(255/(Imagen.shape[1]-1))*factorcartesiano]
                vectores.append(aux)
        #Ahora que tenemos los datos podemos hacer la jerarquia
        jerar = linkage(vectores, 'ward')
        map=fcluster(jerar, k, criterion='maxclust')
        segmentation = map.reshape(Imagen.shape[0],Imagen.shape[1])

	#Me parece apropiado que el metodo de jerarquia tambien devuelva la jerarquia
        return jerar, segmentation
    






##############################################################################
##############################################################################
#Vamos a cargar las imagenes del Training
#os.system('ls ./images/train > train.dat')
#Ahora cargamos dicha lista
with open('train.dat', 'r') as myfile:
    data=myfile.readlines()
#Estos nombres tienen un /n que me molesta asi que lo voy a quitar, ademas el ultimo elemento no es una imagen
nombrestrain=[s.replace('\n','') for s in data]
nombrestrain=[s.replace('.jpg','') for s in nombrestrain]
#Ademas el ultimo nombre es algo que nada que ver asi que lo quito
nombrestrain.pop()


#Vamos a cargar las imagenes del test
#os.system('ls ./images/test > test.dat')
#Ahora cargamos dicha lista
with open('test.dat', 'r') as myfile:
    data=myfile.readlines()
#Estos nombres tienen un /n que me molesta asi que lo voy a quitar, ademas el ultimo elemento no es una imagen
nombrestest=[s.replace('\n','') for s in data]
nombrestest=[s.replace('.jpg','') for s in nombrestest]
#Ademas el ultimo nombre es algo que nada que ver asi que lo quito
nombrestest.pop()


#Vamos a cargar las imagenes de val
#os.system('ls ./images/val > val.dat')
#Ahora cargamos dicha lista
with open('val.dat', 'r') as myfile:
    data=myfile.readlines()
#Estos nombres tienen un /n que me molesta asi que lo voy a quitar, ademas el ultimo elemento no es una imagen
nombresval=[s.replace('\n','') for s in data]
nombresval=[s.replace('.jpg','') for s in nombresval]
#Ademas el ultimo nombre es algo que nada que ver asi que lo quito
nombresval.pop()


#Definimos el numero total de imagenes a segmentar
nimagenesseg=float(len(nombresval)+len(nombrestest)+len(nombrestrain))
#Definimos un contador de cuantas llevamos
contadorima=0

valoresk=range(2,200)
               
        
#Probemos las imagenes del train
for o in nombrestrain:
    filename = "./images/train/"+str(o)+".jpg"
    Imagen1 = io.imread(filename)
    listakmeans=[]
    listagmm=[]
    for w in valoresk:
        listakmeans.append(segmentByClustering( Imagen1, 'rgb', 'kmeans', w))
        listagmm.append(segmentByClustering( Imagen1, 'rgb', 'gmm', w))
    np.savetxt("./missegmentaciones/Training/kmeans/"+str(o)+".mat",np.array(listakmeans))
    np.savetxt("./missegmentaciones/Training/gmm/"+str(o)+".mat",np.array(listagmm))
    contadorima=contadorima+1
    print(float(contadorima/nimagenesseg)*100.0)




#Probemos las imagenes del test
for o in nombrestest:
    filename = "./images/test/"+str(o)+".jpg"
    Imagen1 = io.imread(filename)
    listakmeans=[]
    listagmm=[]
    for w in valoresk:
        listakmeans.append(segmentByClustering( Imagen1, 'rgb', 'kmeans', w))
        listagmm.append(segmentByClustering( Imagen1, 'rgb', 'gmm', w))
    np.savetxt("./missegmentaciones/Test/kmeans/"+str(o)+".mat",np.array(listakmeans))
    np.savetxt("./missegmentaciones/Test/gmm/"+str(o)+".mat",np.array(listagmm))
    contadorima=contadorima+1
    print(float(contadorima/nimagenesseg)*100.0)




#Probemos las imagenes del validation
for o in nombresval:
    filename = "./images/val/"+str(o)+".jpg"
    Imagen1 = io.imread(filename)
    listakmeans=[]
    listagmm=[]
    for w in valoresk:
        listakmeans.append(segmentByClustering( Imagen1, 'rgb', 'kmeans', w))
        listagmm.append(segmentByClustering( Imagen1, 'rgb', 'gmm', w))
    np.savetxt("./missegmentaciones/Validation/kmeans/"+str(o)+".mat",np.array(listakmeans))
    np.savetxt("./missegmentaciones/Validation/gmm/"+str(o)+".mat",np.array(listagmm))
    contadorima=contadorima+1
    print(float(contadorima/nimagenesseg)*100.0)

    




###############################################################################################
###############################################################################################







	




