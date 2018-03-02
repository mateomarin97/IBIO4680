import sys
sys.path.append('./lib/python')

#Create a filter bank with deafult params
from fbCreate import fbCreate
fb = fbCreate()
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from scipy import misc
import scipy.ndimage as ndimage
import cv2 as cv
import os
import time
from sklearn.ensemble import RandomForestClassifier
from fbRun import fbRun
from computeTextons import computeTextons
from assignTextons import assignTextons


#Esta funcion pasa la matriz de textones a un histograma
def histc(X, bins):
    map_to_bins = np.digitize(X,bins)
    r = np.zeros(bins.shape)
    for i in map_to_bins:
        r[i-1] += 1
    return np.array(r)




#Definimos el kernel de la interseccion
def kernelint(A,B):
    if(len(A)!=len(B)):
        print("Error, los dos histogramas deben tener el mismo tamano")
    else:
        suma=0
        for i in range(len(A)):
            suma+=min(A[i],B[i])
        return suma


#Definimos la funcion de clasificacion
def clasifica(A):
    n1=kernelint(A,histBark1Tra)
    n2=kernelint(A,histBark2Tra)
    n3=kernelint(A,histBark3Tra)
    n4=kernelint(A,histWood1Tra)
    n5=kernelint(A,histWood2Tra)
    n6=kernelint(A,histWood3Tra)
    n7=kernelint(A,histWaterTra)
    n8=kernelint(A,histGraniteTra)
    n9=kernelint(A,histMarbleTra)
    n10=kernelint(A,histFloor1Tra)
    n11=kernelint(A,histFloor2Tra)
    n12=kernelint(A,histPebblesTra)
    n13=kernelint(A,histWallTra)
    n14=kernelint(A,histBrick1Tra)
    n15=kernelint(A,histBrick2Tra)
    n16=kernelint(A,histGlass1Tra)
    n17=kernelint(A,histGlass2Tra)
    n18=kernelint(A,histCarpet1Tra)
    n19=kernelint(A,histCarpet2Tra)
    n20=kernelint(A,histUpholsteryTra)
    n21=kernelint(A,histWallpaperTra)
    n22=kernelint(A,histFurTra)
    n23=kernelint(A,histKnitTra)
    n24=kernelint(A,histCorduroyTra)
    n25=kernelint(A,histPlaidTra)
    lista=[n1,n2,n3,n4,n5,n6,n7,n8,n9,n10,n11,n12,n13,n14,n15,n16,n17,n18,n19,n20,n21,n22,n23,n24,n25]
    maximo=n1
    indice=1
    for i in range(len(lista)):
        if(lista[i]>maximo):
            maximo=lista[i]
            indice=i+1
    return indice


#Definimos una funcion que cuenta el numero de veces que aparece un numero en una lista
def cuenta(numero,lista):
    contador=0
    for i in lista:
        if(i==numero):
            contador=contador+1
    return contador





#Definimos el ancho y largo de las imagenes
ancho=480
largo=640

ListaNpix=[80,90]
Listak=[140,145,150,155,160]

for f in ListaNpix:

	#Empiezo a contar el tiempo
	tiempoi=time.time()

	#No vamos a trabajar con toda la imagen, es muy grande. Vamos a agarrar un cuadro de 80x80 en la mitad de ella 
	Npix=f
	csi0=int(ancho*0.5 +1.0 - Npix*0.5)
	csi1=int(largo*0.5 +1.0 - Npix*0.5)
	cid0=int(ancho*0.5 -1.0 +Npix*0.5)
	cid1=int(largo*0.5 -1.0 +Npix*0.5)


	#Procedemos a importar las imagenes del Training, no va a ser bonito.


	#Bark1
	#GUardamos los nombres de las imagenes del Training
	os.system(' ls ./Data/T01_bark1Tra > Bark1Tra.dat')

	#Ahora cargamos dicha lista
	with open('Bark1Tra.dat', 'r') as myfile:
	    data=myfile.readlines()
	#Estons nombres tienen un /n que me molesta asi que lo voy a quitar.
	nom=[s.replace('\n','') for s in data]
	#Ahora cargamos las imagenes
	Bark1Tra=[]
	for i in nom:
	    Bark1Tra.append(np.asarray(Image.open(r'./Data/T01_bark1Tra/'+str(i)).convert("L").crop((csi0,csi1,cid0,cid1))))

	    
	#Bark2
	#GUardamos los nombres de las imagenes del Training
	os.system(' ls ./Data/T02_bark2Tra > Bark2Tra.dat')

	#Ahora cargamos dicha lista
	with open('Bark2Tra.dat', 'r') as myfile:
	    data=myfile.readlines()
	#Estons nombres tienen un /n que me molesta asi que lo voy a quitar.
	nom=[s.replace('\n','') for s in data]
	#Ahora cargamos las imagenes
	Bark2Tra=[]
	for i in nom:
	    Bark2Tra.append(np.asarray(Image.open(r'./Data/T02_bark2Tra/'+str(i)).convert("L").crop((csi0,csi1,cid0,cid1))))


	#Bark3
	#GUardamos los nombres de las imagenes del Training
	os.system(' ls ./Data/T03_bark3Tra > Bark3Tra.dat')

	#Ahora cargamos dicha lista
	with open('Bark3Tra.dat', 'r') as myfile:
	    data=myfile.readlines()
	#Estons nombres tienen un /n que me molesta asi que lo voy a quitar.
	nom=[s.replace('\n','') for s in data]
	#Ahora cargamos las imagenes
	Bark3Tra=[]
	for i in nom:
	    Bark3Tra.append( np.asarray(Image.open(r'./Data/T03_bark3Tra/'+str(i)).convert("L").crop((csi0,csi1,cid0,cid1))))



	#Wood1
	#GUardamos los nombres de las imagenes del Training
	os.system(' ls ./Data/T04_wood1Tra > Wood1Tra.dat')

	#Ahora cargamos dicha lista
	with open('Wood1Tra.dat', 'r') as myfile:
	    data=myfile.readlines()
	#Estons nombres tienen un /n que me molesta asi que lo voy a quitar.
	nom=[s.replace('\n','') for s in data]
	#Ahora cargamos las imagenes
	Wood1Tra=[]
	for i in nom:
	    Wood1Tra.append( np.asarray(Image.open(r'./Data/T04_wood1Tra/'+str(i)).convert("L").crop((csi0,csi1,cid0,cid1))))



	#Wood2
	#GUardamos los nombres de las imagenes del Training
	os.system(' ls ./Data/T05_wood2Tra > Wood2Tra.dat')

	#Ahora cargamos dicha lista
	with open('Wood2Tra.dat', 'r') as myfile:
	    data=myfile.readlines()
	#Estons nombres tienen un /n que me molesta asi que lo voy a quitar.
	nom=[s.replace('\n','') for s in data]
	#Ahora cargamos las imagenes
	Wood2Tra=[]
	for i in nom:
	    Wood2Tra.append( np.asarray(Image.open(r'./Data/T05_wood2Tra/'+str(i)).convert("L").crop((csi0,csi1,cid0,cid1))))


	#Wood3
	#GUardamos los nombres de las imagenes del Training
	os.system(' ls ./Data/T06_wood3Tra > Wood3Tra.dat')

	#Ahora cargamos dicha lista
	with open('Wood3Tra.dat', 'r') as myfile:
	    data=myfile.readlines()
	#Estons nombres tienen un /n que me molesta asi que lo voy a quitar.
	nom=[s.replace('\n','') for s in data]
	#Ahora cargamos las imagenes
	Wood3Tra=[]
	for i in nom:
	    Wood3Tra.append( np.asarray(Image.open(r'./Data/T06_wood3Tra/'+str(i)).convert("L").crop((csi0,csi1,cid0,cid1))))

	#Water
	#GUardamos los nombres de las imagenes del Training
	os.system(' ls ./Data/T07_waterTra > WaterTra.dat')

	#Ahora cargamos dicha lista
	with open('WaterTra.dat', 'r') as myfile:
	    data=myfile.readlines()
	#Estons nombres tienen un /n que me molesta asi que lo voy a quitar.
	nom=[s.replace('\n','') for s in data]
	#Ahora cargamos las imagenes
	WaterTra=[]
	for i in nom:
	    WaterTra.append( np.asarray(Image.open(r'./Data/T07_waterTra/'+str(i)).convert("L").crop((csi0,csi1,cid0,cid1))))



	#Granite
	#GUardamos los nombres de las imagenes del Training
	os.system(' ls ./Data/T08_graniteTra > GraniteTra.dat')

	#Ahora cargamos dicha lista
	with open('GraniteTra.dat', 'r') as myfile:
	    data=myfile.readlines()
	#Estons nombres tienen un /n que me molesta asi que lo voy a quitar.
	nom=[s.replace('\n','') for s in data]
	#Ahora cargamos las imagenes
	GraniteTra=[]
	for i in nom:
	    GraniteTra.append( np.asarray(Image.open(r'./Data/T08_graniteTra/'+str(i)).convert("L").crop((csi0,csi1,cid0,cid1))))


	#Marble
	#GUardamos los nombres de las imagenes del Training
	os.system(' ls ./Data/T09_marbleTra > MarbleTra.dat')

	#Ahora cargamos dicha lista
	with open('MarbleTra.dat', 'r') as myfile:
	    data=myfile.readlines()
	#Estons nombres tienen un /n que me molesta asi que lo voy a quitar.
	nom=[s.replace('\n','') for s in data]
	#Ahora cargamos las imagenes
	MarbleTra=[]
	for i in nom:
	    MarbleTra.append( np.asarray(Image.open(r'./Data/T09_marbleTra/'+str(i)).convert("L").crop((csi0,csi1,cid0,cid1))))

	#Floor1
	#GUardamos los nombres de las imagenes del Training
	os.system(' ls ./Data/T10_floor1Tra > Floor1Tra.dat')

	#Ahora cargamos dicha lista
	with open('Floor1Tra.dat', 'r') as myfile:
	    data=myfile.readlines()
	#Estons nombres tienen un /n que me molesta asi que lo voy a quitar.
	nom=[s.replace('\n','') for s in data]
	#Ahora cargamos las imagenes
	Floor1Tra=[]
	for i in nom:
	    Floor1Tra.append( np.asarray(Image.open(r'./Data/T10_floor1Tra/'+str(i)).convert("L").crop((csi0,csi1,cid0,cid1))))


	#Floor2
	#GUardamos los nombres de las imagenes del Training
	os.system(' ls ./Data/T11_floor2Tra > Floor2Tra.dat')

	#Ahora cargamos dicha lista
	with open('Floor2Tra.dat', 'r') as myfile:
	    data=myfile.readlines()
	#Estons nombres tienen un /n que me molesta asi que lo voy a quitar.
	nom=[s.replace('\n','') for s in data]
	#Ahora cargamos las imagenes
	Floor2Tra=[]
	for i in nom:
	    Floor2Tra.append( np.asarray(Image.open(r'./Data/T11_floor2Tra/'+str(i)).convert("L").crop((csi0,csi1,cid0,cid1))))


	#Pebbles
	#GUardamos los nombres de las imagenes del Training
	os.system(' ls ./Data/T12_pebblesTra > PebblesTra.dat')

	#Ahora cargamos dicha lista
	with open('PebblesTra.dat', 'r') as myfile:
	    data=myfile.readlines()
	#Estons nombres tienen un /n que me molesta asi que lo voy a quitar.
	nom=[s.replace('\n','') for s in data]
	#Ahora cargamos las imagenes
	PebblesTra=[]
	for i in nom:
	    PebblesTra.append( np.asarray(Image.open(r'./Data/T12_pebblesTra/'+str(i)).convert("L").crop((csi0,csi1,cid0,cid1))))


	#Wall
	#GUardamos los nombres de las imagenes del Training
	os.system(' ls ./Data/T13_wallTra > WallTra.dat')

	#Ahora cargamos dicha lista
	with open('WallTra.dat', 'r') as myfile:
	    data=myfile.readlines()
	#Estons nombres tienen un /n que me molesta asi que lo voy a quitar.
	nom=[s.replace('\n','') for s in data]
	#Ahora cargamos las imagenes
	WallTra=[]
	for i in nom:
	    WallTra.append( np.asarray(Image.open(r'./Data/T13_wallTra/'+str(i)).convert("L").crop((csi0,csi1,cid0,cid1))))


	#Brick1
	#GUardamos los nombres de las imagenes del Training
	os.system(' ls ./Data/T14_brick1Tra > Brick1Tra.dat')

	#Ahora cargamos dicha lista
	with open('Brick1Tra.dat', 'r') as myfile:
	    data=myfile.readlines()
	#Estons nombres tienen un /n que me molesta asi que lo voy a quitar.
	nom=[s.replace('\n','') for s in data]
	#Ahora cargamos las imagenes
	Brick1Tra=[]
	for i in nom:
	    Brick1Tra.append( np.asarray(Image.open(r'./Data/T14_brick1Tra/'+str(i)).convert("L").crop((csi0,csi1,cid0,cid1))))


	#Brick2
	#GUardamos los nombres de las imagenes del Training
	os.system(' ls ./Data/T15_brick2Tra > Brick2Tra.dat')

	#Ahora cargamos dicha lista
	with open('Brick2Tra.dat', 'r') as myfile:
	    data=myfile.readlines()
	#Estons nombres tienen un /n que me molesta asi que lo voy a quitar.
	nom=[s.replace('\n','') for s in data]
	#Ahora cargamos las imagenes
	Brick2Tra=[]
	for i in nom:
	    Brick2Tra.append( np.asarray(Image.open(r'./Data/T15_brick2Tra/'+str(i)).convert("L").crop((csi0,csi1,cid0,cid1))))


	#Glass1
	#GUardamos los nombres de las imagenes del Training
	os.system(' ls ./Data/T16_glass1Tra > Glass1Tra.dat')

	#Ahora cargamos dicha lista
	with open('Glass1Tra.dat', 'r') as myfile:
	    data=myfile.readlines()
	#Estons nombres tienen un /n que me molesta asi que lo voy a quitar.
	nom=[s.replace('\n','') for s in data]
	#Ahora cargamos las imagenes
	Glass1Tra=[]
	for i in nom:
	    Glass1Tra.append( np.asarray(Image.open(r'./Data/T16_glass1Tra/'+str(i)).convert("L").crop((csi0,csi1,cid0,cid1))))


	#Glass2
	#GUardamos los nombres de las imagenes del Training
	os.system(' ls ./Data/T17_glass2Tra > Glass2Tra.dat')

	#Ahora cargamos dicha lista
	with open('Glass2Tra.dat', 'r') as myfile:
	    data=myfile.readlines()
	#Estons nombres tienen un /n que me molesta asi que lo voy a quitar.
	nom=[s.replace('\n','') for s in data]
	#Ahora cargamos las imagenes
	Glass2Tra=[]
	for i in nom:
	    Glass2Tra.append( np.asarray(Image.open(r'./Data/T17_glass2Tra/'+str(i)).convert("L").crop((csi0,csi1,cid0,cid1))))


	#Carpet1
	#GUardamos los nombres de las imagenes del Training
	os.system(' ls ./Data/T18_carpet1Tra > Carpet1Tra.dat')

	#Ahora cargamos dicha lista
	with open('Carpet1Tra.dat', 'r') as myfile:
	    data=myfile.readlines()
	#Estons nombres tienen un /n que me molesta asi que lo voy a quitar.
	nom=[s.replace('\n','') for s in data]
	#Ahora cargamos las imagenes
	Carpet1Tra=[]
	for i in nom:
	    Carpet1Tra.append( np.asarray(Image.open(r'./Data/T18_carpet1Tra/'+str(i)).convert("L").crop((csi0,csi1,cid0,cid1))))


	#Carpet2
	#GUardamos los nombres de las imagenes del Training
	os.system(' ls ./Data/T19_carpet2Tra > Carpet2Tra.dat')

	#Ahora cargamos dicha lista
	with open('Carpet2Tra.dat', 'r') as myfile:
	    data=myfile.readlines()
	#Estons nombres tienen un /n que me molesta asi que lo voy a quitar.
	nom=[s.replace('\n','') for s in data]
	#Ahora cargamos las imagenes
	Carpet2Tra=[]
	for i in nom:
	    Carpet2Tra.append( np.asarray(Image.open(r'./Data/T19_carpet2Tra/'+str(i)).convert("L").crop((csi0,csi1,cid0,cid1))))


	#Upholstery
	#GUardamos los nombres de las imagenes del Training
	os.system(' ls ./Data/T20_upholsteryTra > UpholsteryTra.dat')

	#Ahora cargamos dicha lista
	with open('UpholsteryTra.dat', 'r') as myfile:
	    data=myfile.readlines()
	#Estons nombres tienen un /n que me molesta asi que lo voy a quitar.
	nom=[s.replace('\n','') for s in data]
	#Ahora cargamos las imagenes
	UpholsteryTra=[]
	for i in nom:
	    UpholsteryTra.append( np.asarray(Image.open(r'./Data/T20_upholsteryTra/'+str(i)).convert("L").crop((csi0,csi1,cid0,cid1))))


	#Wallpaper
	#GUardamos los nombres de las imagenes del Training
	os.system(' ls ./Data/T21_wallpaperTra > WallpaperTra.dat')

	#Ahora cargamos dicha lista
	with open('WallpaperTra.dat', 'r') as myfile:
	    data=myfile.readlines()
	#Estons nombres tienen un /n que me molesta asi que lo voy a quitar.
	nom=[s.replace('\n','') for s in data]
	#Ahora cargamos las imagenes
	WallpaperTra=[]
	for i in nom:
	    WallpaperTra.append( np.asarray(Image.open(r'./Data/T21_wallpaperTra/'+str(i)).convert("L").crop((csi0,csi1,cid0,cid1))))


	#Fur
	#GUardamos los nombres de las imagenes del Training
	os.system(' ls ./Data/T22_furTra > FurTra.dat')

	#Ahora cargamos dicha lista
	with open('FurTra.dat', 'r') as myfile:
	    data=myfile.readlines()
	#Estons nombres tienen un /n que me molesta asi que lo voy a quitar.
	nom=[s.replace('\n','') for s in data]
	#Ahora cargamos las imagenes
	FurTra=[]
	for i in nom:
	    FurTra.append( np.asarray(Image.open(r'./Data/T22_furTra/'+str(i)).convert("L").crop((csi0,csi1,cid0,cid1))))


	#Knit
	#GUardamos los nombres de las imagenes del Training
	os.system(' ls ./Data/T23_knitTra > KnitTra.dat')

	#Ahora cargamos dicha lista
	with open('KnitTra.dat', 'r') as myfile:
	    data=myfile.readlines()
	#Estons nombres tienen un /n que me molesta asi que lo voy a quitar.
	nom=[s.replace('\n','') for s in data]
	#Ahora cargamos las imagenes
	KnitTra=[]
	for i in nom:
	    KnitTra.append( np.asarray(Image.open(r'./Data/T23_knitTra/'+str(i)).convert("L").crop((csi0,csi1,cid0,cid1))))


	#Corduroy
	#GUardamos los nombres de las imagenes del Training
	os.system(' ls ./Data/T24_corduroyTra > CorduroyTra.dat')

	#Ahora cargamos dicha lista
	with open('CorduroyTra.dat', 'r') as myfile:
	    data=myfile.readlines()
	#Estons nombres tienen un /n que me molesta asi que lo voy a quitar.
	nom=[s.replace('\n','') for s in data]
	#Ahora cargamos las imagenes
	CorduroyTra=[]
	for i in nom:
	    CorduroyTra.append( np.asarray(Image.open(r'./Data/T24_corduroyTra/'+str(i)).convert("L").crop((csi0,csi1,cid0,cid1))))


	#Plaid
	#GUardamos los nombres de las imagenes del Training
	os.system(' ls ./Data/T25_plaidTra > PlaidTra.dat')

	#Ahora cargamos dicha lista
	with open('PlaidTra.dat', 'r') as myfile:
	    data=myfile.readlines()
	#Estons nombres tienen un /n que me molesta asi que lo voy a quitar.
	nom=[s.replace('\n','') for s in data]
	#Ahora cargamos las imagenes
	PlaidTra=[]
	for i in nom:
	    PlaidTra.append( np.asarray(Image.open(r'./Data/T25_plaidTra/'+str(i)).convert("L").crop((csi0,csi1,cid0,cid1))))

	print("Fase de carga completada")


	#Ahora procedemos a cargar los datos de Test

	#Bark1
	#GUardamos los nombres de las imagenes del Test
	os.system(' ls ./Data/T01_bark1Test > Bark1Tra.dat')

	#Ahora cargamos dicha lista
	with open('Bark1Tra.dat', 'r') as myfile:
	    data=myfile.readlines()
	#Estons nombres tienen un /n que me molesta asi que lo voy a quitar.
	nom=[s.replace('\n','') for s in data]
	#Ahora cargamos las imagenes
	Bark1Test=[]
	for i in nom:
	    Bark1Test.append(np.asarray(Image.open(r'./Data/T01_bark1Test/'+str(i)).convert("L").crop((csi0,csi1,cid0,cid1))))

	    
	#Bark2
	#GUardamos los nombres de las imagenes del Test
	os.system(' ls ./Data/T02_bark2Test > Bark2Tra.dat')

	#Ahora cargamos dicha lista
	with open('Bark2Tra.dat', 'r') as myfile:
	    data=myfile.readlines()
	#Estons nombres tienen un /n que me molesta asi que lo voy a quitar.
	nom=[s.replace('\n','') for s in data]
	#Ahora cargamos las imagenes
	Bark2Test=[]
	for i in nom:
	    Bark2Test.append(np.asarray(Image.open(r'./Data/T02_bark2Test/'+str(i)).convert("L").crop((csi0,csi1,cid0,cid1))))


	#Bark3
	#GUardamos los nombres de las imagenes del Test
	os.system(' ls ./Data/T03_bark3Test > Bark3Tra.dat')

	#Ahora cargamos dicha lista
	with open('Bark3Tra.dat', 'r') as myfile:
	    data=myfile.readlines()
	#Estons nombres tienen un /n que me molesta asi que lo voy a quitar.
	nom=[s.replace('\n','') for s in data]
	#Ahora cargamos las imagenes
	Bark3Test=[]
	for i in nom:
	    Bark3Test.append( np.asarray(Image.open(r'./Data/T03_bark3Test/'+str(i)).convert("L").crop((csi0,csi1,cid0,cid1))))



	#Wood1
	#GUardamos los nombres de las imagenes del Test
	os.system(' ls ./Data/T04_wood1Test > Wood1Tra.dat')

	#Ahora cargamos dicha lista
	with open('Wood1Tra.dat', 'r') as myfile:
	    data=myfile.readlines()
	#Estons nombres tienen un /n que me molesta asi que lo voy a quitar.
	nom=[s.replace('\n','') for s in data]
	#Ahora cargamos las imagenes
	Wood1Test=[]
	for i in nom:
	    Wood1Test.append( np.asarray(Image.open(r'./Data/T04_wood1Test/'+str(i)).convert("L").crop((csi0,csi1,cid0,cid1))))



	#Wood2
	#GUardamos los nombres de las imagenes del Test
	os.system(' ls ./Data/T05_wood2Test > Wood2Tra.dat')

	#Ahora cargamos dicha lista
	with open('Wood2Tra.dat', 'r') as myfile:
	    data=myfile.readlines()
	#Estons nombres tienen un /n que me molesta asi que lo voy a quitar.
	nom=[s.replace('\n','') for s in data]
	#Ahora cargamos las imagenes
	Wood2Test=[]
	for i in nom:
	    Wood2Test.append( np.asarray(Image.open(r'./Data/T05_wood2Test/'+str(i)).convert("L").crop((csi0,csi1,cid0,cid1))))


	#Wood3
	#GUardamos los nombres de las imagenes del Test
	os.system(' ls ./Data/T06_wood3Test > Wood3Tra.dat')

	#Ahora cargamos dicha lista
	with open('Wood3Tra.dat', 'r') as myfile:
	    data=myfile.readlines()
	#Estons nombres tienen un /n que me molesta asi que lo voy a quitar.
	nom=[s.replace('\n','') for s in data]
	#Ahora cargamos las imagenes
	Wood3Test=[]
	for i in nom:
	    Wood3Test.append( np.asarray(Image.open(r'./Data/T06_wood3Test/'+str(i)).convert("L").crop((csi0,csi1,cid0,cid1))))

	#Water
	#GUardamos los nombres de las imagenes del Test
	os.system(' ls ./Data/T07_waterTest > WaterTra.dat')

	#Ahora cargamos dicha lista
	with open('WaterTra.dat', 'r') as myfile:
	    data=myfile.readlines()
	#Estons nombres tienen un /n que me molesta asi que lo voy a quitar.
	nom=[s.replace('\n','') for s in data]
	#Ahora cargamos las imagenes
	WaterTest=[]
	for i in nom:
	    WaterTest.append( np.asarray(Image.open(r'./Data/T07_waterTest/'+str(i)).convert("L").crop((csi0,csi1,cid0,cid1))))



	#Granite
	#GUardamos los nombres de las imagenes del Test
	os.system(' ls ./Data/T08_graniteTest > GraniteTra.dat')

	#Ahora cargamos dicha lista
	with open('GraniteTra.dat', 'r') as myfile:
	    data=myfile.readlines()
	#Estons nombres tienen un /n que me molesta asi que lo voy a quitar.
	nom=[s.replace('\n','') for s in data]
	#Ahora cargamos las imagenes
	GraniteTest=[]
	for i in nom:
	    GraniteTest.append( np.asarray(Image.open(r'./Data/T08_graniteTest/'+str(i)).convert("L").crop((csi0,csi1,cid0,cid1))))


	#Marble
	#GUardamos los nombres de las imagenes del Test
	os.system(' ls ./Data/T09_marbleTest > MarbleTra.dat')

	#Ahora cargamos dicha lista
	with open('MarbleTra.dat', 'r') as myfile:
	    data=myfile.readlines()
	#Estons nombres tienen un /n que me molesta asi que lo voy a quitar.
	nom=[s.replace('\n','') for s in data]
	#Ahora cargamos las imagenes
	MarbleTest=[]
	for i in nom:
	    MarbleTest.append( np.asarray(Image.open(r'./Data/T09_marbleTest/'+str(i)).convert("L").crop((csi0,csi1,cid0,cid1))))

	#Floor1
	#GUardamos los nombres de las imagenes del Test
	os.system(' ls ./Data/T10_floor1Test > Floor1Tra.dat')

	#Ahora cargamos dicha lista
	with open('Floor1Tra.dat', 'r') as myfile:
	    data=myfile.readlines()
	#Estons nombres tienen un /n que me molesta asi que lo voy a quitar.
	nom=[s.replace('\n','') for s in data]
	#Ahora cargamos las imagenes
	Floor1Test=[]
	for i in nom:
	    Floor1Test.append( np.asarray(Image.open(r'./Data/T10_floor1Test/'+str(i)).convert("L").crop((csi0,csi1,cid0,cid1))))


	#Floor2
	#GUardamos los nombres de las imagenes del Test
	os.system(' ls ./Data/T11_floor2Test > Floor2Tra.dat')

	#Ahora cargamos dicha lista
	with open('Floor2Tra.dat', 'r') as myfile:
	    data=myfile.readlines()
	#Estons nombres tienen un /n que me molesta asi que lo voy a quitar.
	nom=[s.replace('\n','') for s in data]
	#Ahora cargamos las imagenes
	Floor2Test=[]
	for i in nom:
	    Floor2Test.append( np.asarray(Image.open(r'./Data/T11_floor2Test/'+str(i)).convert("L").crop((csi0,csi1,cid0,cid1))))


	#Pebbles
	#GUardamos los nombres de las imagenes del Test
	os.system(' ls ./Data/T12_pebblesTest > PebblesTra.dat')

	#Ahora cargamos dicha lista
	with open('PebblesTra.dat', 'r') as myfile:
	    data=myfile.readlines()
	#Estons nombres tienen un /n que me molesta asi que lo voy a quitar.
	nom=[s.replace('\n','') for s in data]
	#Ahora cargamos las imagenes
	PebblesTest=[]
	for i in nom:
	    PebblesTest.append( np.asarray(Image.open(r'./Data/T12_pebblesTest/'+str(i)).convert("L").crop((csi0,csi1,cid0,cid1))))


	#Wall
	#GUardamos los nombres de las imagenes del Test
	os.system(' ls ./Data/T13_wallTest > WallTra.dat')

	#Ahora cargamos dicha lista
	with open('WallTra.dat', 'r') as myfile:
	    data=myfile.readlines()
	#Estons nombres tienen un /n que me molesta asi que lo voy a quitar.
	nom=[s.replace('\n','') for s in data]
	#Ahora cargamos las imagenes
	WallTest=[]
	for i in nom:
	    WallTest.append( np.asarray(Image.open(r'./Data/T13_wallTest/'+str(i)).convert("L").crop((csi0,csi1,cid0,cid1))))


	#Brick1
	#GUardamos los nombres de las imagenes del Test
	os.system(' ls ./Data/T14_brick1Test > Brick1Tra.dat')

	#Ahora cargamos dicha lista
	with open('Brick1Tra.dat', 'r') as myfile:
	    data=myfile.readlines()
	#Estons nombres tienen un /n que me molesta asi que lo voy a quitar.
	nom=[s.replace('\n','') for s in data]
	#Ahora cargamos las imagenes
	Brick1Test=[]
	for i in nom:
	    Brick1Test.append( np.asarray(Image.open(r'./Data/T14_brick1Test/'+str(i)).convert("L").crop((csi0,csi1,cid0,cid1))))


	#Brick2
	#GUardamos los nombres de las imagenes del Test
	os.system(' ls ./Data/T15_brick2Test > Brick2Tra.dat')

	#Ahora cargamos dicha lista
	with open('Brick2Tra.dat', 'r') as myfile:
	    data=myfile.readlines()
	#Estons nombres tienen un /n que me molesta asi que lo voy a quitar.
	nom=[s.replace('\n','') for s in data]
	#Ahora cargamos las imagenes
	Brick2Test=[]
	for i in nom:
	    Brick2Test.append( np.asarray(Image.open(r'./Data/T15_brick2Test/'+str(i)).convert("L").crop((csi0,csi1,cid0,cid1))))


	#Glass1
	#GUardamos los nombres de las imagenes del Test
	os.system(' ls ./Data/T16_glass1Test > Glass1Tra.dat')

	#Ahora cargamos dicha lista
	with open('Glass1Tra.dat', 'r') as myfile:
	    data=myfile.readlines()
	#Estons nombres tienen un /n que me molesta asi que lo voy a quitar.
	nom=[s.replace('\n','') for s in data]
	#Ahora cargamos las imagenes
	Glass1Test=[]
	for i in nom:
	    Glass1Test.append( np.asarray(Image.open(r'./Data/T16_glass1Test/'+str(i)).convert("L").crop((csi0,csi1,cid0,cid1))))


	#Glass2
	#GUardamos los nombres de las imagenes del Test
	os.system(' ls ./Data/T17_glass2Test > Glass2Tra.dat')

	#Ahora cargamos dicha lista
	with open('Glass2Tra.dat', 'r') as myfile:
	    data=myfile.readlines()
	#Estons nombres tienen un /n que me molesta asi que lo voy a quitar.
	nom=[s.replace('\n','') for s in data]
	#Ahora cargamos las imagenes
	Glass2Test=[]
	for i in nom:
	    Glass2Test.append( np.asarray(Image.open(r'./Data/T17_glass2Test/'+str(i)).convert("L").crop((csi0,csi1,cid0,cid1))))


	#Carpet1
	#GUardamos los nombres de las imagenes del Test
	os.system(' ls ./Data/T18_carpet1Test > Carpet1Tra.dat')

	#Ahora cargamos dicha lista
	with open('Carpet1Tra.dat', 'r') as myfile:
	    data=myfile.readlines()
	#Estons nombres tienen un /n que me molesta asi que lo voy a quitar.
	nom=[s.replace('\n','') for s in data]
	#Ahora cargamos las imagenes
	Carpet1Test=[]
	for i in nom:
	    Carpet1Test.append( np.asarray(Image.open(r'./Data/T18_carpet1Test/'+str(i)).convert("L").crop((csi0,csi1,cid0,cid1))))


	#Carpet2
	#GUardamos los nombres de las imagenes del Test
	os.system(' ls ./Data/T19_carpet2Test > Carpet2Tra.dat')

	#Ahora cargamos dicha lista
	with open('Carpet2Tra.dat', 'r') as myfile:
	    data=myfile.readlines()
	#Estons nombres tienen un /n que me molesta asi que lo voy a quitar.
	nom=[s.replace('\n','') for s in data]
	#Ahora cargamos las imagenes
	Carpet2Test=[]
	for i in nom:
	    Carpet2Test.append( np.asarray(Image.open(r'./Data/T19_carpet2Test/'+str(i)).convert("L").crop((csi0,csi1,cid0,cid1))))


	#Upholstery
	#GUardamos los nombres de las imagenes del Test
	os.system(' ls ./Data/T20_upholsteryTest > UpholsteryTra.dat')

	#Ahora cargamos dicha lista
	with open('UpholsteryTra.dat', 'r') as myfile:
	    data=myfile.readlines()
	#Estons nombres tienen un /n que me molesta asi que lo voy a quitar.
	nom=[s.replace('\n','') for s in data]
	#Ahora cargamos las imagenes
	UpholsteryTest=[]
	for i in nom:
	    UpholsteryTest.append( np.asarray(Image.open(r'./Data/T20_upholsteryTest/'+str(i)).convert("L").crop((csi0,csi1,cid0,cid1))))


	#Wallpaper
	#GUardamos los nombres de las imagenes del Test
	os.system(' ls ./Data/T21_wallpaperTest > WallpaperTra.dat')

	#Ahora cargamos dicha lista
	with open('WallpaperTra.dat', 'r') as myfile:
	    data=myfile.readlines()
	#Estons nombres tienen un /n que me molesta asi que lo voy a quitar.
	nom=[s.replace('\n','') for s in data]
	#Ahora cargamos las imagenes
	WallpaperTest=[]
	for i in nom:
	    WallpaperTest.append( np.asarray(Image.open(r'./Data/T21_wallpaperTest/'+str(i)).convert("L").crop((csi0,csi1,cid0,cid1))))


	#Fur
	#GUardamos los nombres de las imagenes del Test
	os.system(' ls ./Data/T22_furTest > FurTra.dat')

	#Ahora cargamos dicha lista
	with open('FurTra.dat', 'r') as myfile:
	    data=myfile.readlines()
	#Estons nombres tienen un /n que me molesta asi que lo voy a quitar.
	nom=[s.replace('\n','') for s in data]
	#Ahora cargamos las imagenes
	FurTest=[]
	for i in nom:
	    FurTest.append( np.asarray(Image.open(r'./Data/T22_furTest/'+str(i)).convert("L").crop((csi0,csi1,cid0,cid1))))


	#Knit
	#GUardamos los nombres de las imagenes del Test
	os.system(' ls ./Data/T23_knitTest > KnitTra.dat')

	#Ahora cargamos dicha lista
	with open('KnitTra.dat', 'r') as myfile:
	    data=myfile.readlines()
	#Estons nombres tienen un /n que me molesta asi que lo voy a quitar.
	nom=[s.replace('\n','') for s in data]
	#Ahora cargamos las imagenes
	KnitTest=[]
	for i in nom:
	    KnitTest.append( np.asarray(Image.open(r'./Data/T23_knitTest/'+str(i)).convert("L").crop((csi0,csi1,cid0,cid1))))


	#Corduroy
	#GUardamos los nombres de las imagenes del Test
	os.system(' ls ./Data/T24_corduroyTest > CorduroyTra.dat')

	#Ahora cargamos dicha lista
	with open('CorduroyTra.dat', 'r') as myfile:
	    data=myfile.readlines()
	#Estons nombres tienen un /n que me molesta asi que lo voy a quitar.
	nom=[s.replace('\n','') for s in data]
	#Ahora cargamos las imagenes
	CorduroyTest=[]
	for i in nom:
	    CorduroyTest.append( np.asarray(Image.open(r'./Data/T24_corduroyTest/'+str(i)).convert("L").crop((csi0,csi1,cid0,cid1))))


	#Plaid
	#GUardamos los nombres de las imagenes del Test
	os.system(' ls ./Data/T25_plaidTest > PlaidTra.dat')

	#Ahora cargamos dicha lista
	with open('PlaidTra.dat', 'r') as myfile:
	    data=myfile.readlines()
	#Estons nombres tienen un /n que me molesta asi que lo voy a quitar.
	nom=[s.replace('\n','') for s in data]
	#Ahora cargamos las imagenes
	PlaidTest=[]
	for i in nom:
	    PlaidTest.append( np.asarray(Image.open(r'./Data/T25_plaidTest/'+str(i)).convert("L").crop((csi0,csi1,cid0,cid1))))

	print("Fase de carga completada")

	###########################################################################################################################

	#Hacemos listas con las respuestas a filtros de los distintos materiales
	#Bark1
	filterResponsesBark1Tra=[]
	for i in Bark1Tra:
	    filterResponsesBark1Tra.append(fbRun(fb,i))
	print("Filtro 1")

	#Bark2
	filterResponsesBark2Tra=[]
	for i in Bark2Tra:
	    filterResponsesBark2Tra.append(fbRun(fb,i))
	print("Filtro 1")

	#Bark3
	filterResponsesBark3Tra=[]
	for i in Bark3Tra:
	    filterResponsesBark3Tra.append(fbRun(fb,i))

	print("Filtro 1")
	#Wood1
	filterResponsesWood1Tra=[]
	for i in Wood1Tra:
	    filterResponsesWood1Tra.append(fbRun(fb,i))

	print("Filtro 1")
	#Wood2
	filterResponsesWood2Tra=[]
	for i in Wood2Tra:
	    filterResponsesWood2Tra.append(fbRun(fb,i))

	print("Filtro 1")
	#Wood3
	filterResponsesWood3Tra=[]
	for i in Wood3Tra:
	    filterResponsesWood3Tra.append(fbRun(fb,i))

	print("Filtro 1")
	#Water
	filterResponsesWaterTra=[]
	for i in WaterTra:
	    filterResponsesWaterTra.append(fbRun(fb,i))

	print("Filtro 1")
	#Granite
	filterResponsesGraniteTra=[]
	for i in GraniteTra:
	    filterResponsesGraniteTra.append(fbRun(fb,i))

	print("Filtro 1")
	#Marble
	filterResponsesMarbleTra=[]
	for i in MarbleTra:
	    filterResponsesMarbleTra.append(fbRun(fb,i))

	print("Filtro 1")
	#Floor1
	filterResponsesFloor1Tra=[]
	for i in Floor1Tra:
	    filterResponsesFloor1Tra.append(fbRun(fb,i))

	print("Filtro 1")
	#Floor2
	filterResponsesFloor2Tra=[]
	for i in Floor2Tra:
	    filterResponsesFloor2Tra.append(fbRun(fb,i))

	print("Filtro 1")
	#Pebbles
	filterResponsesPebblesTra=[]
	for i in PebblesTra:
	    filterResponsesPebblesTra.append(fbRun(fb,i))

	print("Filtro 1")
	#Wall
	filterResponsesWallTra=[]
	for i in WallTra:
	    filterResponsesWallTra.append(fbRun(fb,i))

	print("Filtro 1")
	#Brick1
	filterResponsesBrick1Tra=[]
	for i in Brick1Tra:
	    filterResponsesBrick1Tra.append(fbRun(fb,i))


	print("Filtro 1")
	#Brick2
	filterResponsesBrick2Tra=[]
	for i in Brick2Tra:
	    filterResponsesBrick2Tra.append(fbRun(fb,i))

	print("Filtro 1")
	#Glass1
	filterResponsesGlass1Tra=[]
	for i in Glass1Tra:
	    filterResponsesGlass1Tra.append(fbRun(fb,i))

	print("Filtro 1")
	#Glass2
	filterResponsesGlass2Tra=[]
	for i in Glass2Tra:
	    filterResponsesGlass2Tra.append(fbRun(fb,i))

	print("Filtro 1")
	#Carpet1
	filterResponsesCarpet1Tra=[]
	for i in Carpet1Tra:
	    filterResponsesCarpet1Tra.append(fbRun(fb,i))

	print("Filtro 1")
	#Carpet2
	filterResponsesCarpet2Tra=[]
	for i in Carpet2Tra:
	    filterResponsesCarpet2Tra.append(fbRun(fb,i))

	print("Filtro 1")
	#Upholstery
	filterResponsesUpholsteryTra=[]
	for i in UpholsteryTra:
	    filterResponsesUpholsteryTra.append(fbRun(fb,i))

	print("Filtro 1")
	#Wallpaper
	filterResponsesWallpaperTra=[]
	for i in WallpaperTra:
	    filterResponsesWallpaperTra.append(fbRun(fb,i))

	print("Filtro 1")
	#Fur
	filterResponsesFurTra=[]
	for i in FurTra:
	    filterResponsesFurTra.append(fbRun(fb,i))
	print("Filtro 1")
	#Knit
	filterResponsesKnitTra=[]
	for i in KnitTra:
	    filterResponsesKnitTra.append(fbRun(fb,i))
	print("Filtro 1")
	#Corduroy
	filterResponsesCorduroyTra=[]
	for i in CorduroyTra:
	    filterResponsesCorduroyTra.append(fbRun(fb,i))
	print("Filtro 1")
	#Plaid
	filterResponsesPlaidTra=[]
	for i in PlaidTra:
	    filterResponsesPlaidTra.append(fbRun(fb,i))
	print("Fin inicializacion filtros")

	#Ahora debo unir todas las respuestas de los filtros en una sola super imagen para cada filtro

	Nfilas=np.asarray(filterResponsesBark1Tra[0]).shape[0]
	Ncolumnas=np.asarray(filterResponsesBark1Tra[0]).shape[1]

	filterResponses=filterResponsesBark1Tra[0]

	for i in range(Nfilas):
	    for j in range(Ncolumnas):
		for p in range(1,len(filterResponsesBark1Tra)):
		    filterResponses[i][j]=np.hstack((filterResponses[i][j],filterResponsesBark1Tra[p][i][j]))
		for p in filterResponsesBark2Tra:
		    filterResponses[i][j]=np.hstack((filterResponses[i][j],p[i][j]))
		for p in filterResponsesBark3Tra:
		    filterResponses[i][j]=np.hstack((filterResponses[i][j],p[i][j]))
		for p in filterResponsesWood1Tra:
		    filterResponses[i][j]=np.hstack((filterResponses[i][j],p[i][j]))
		for p in filterResponsesWood2Tra:
		    filterResponses[i][j]=np.hstack((filterResponses[i][j],p[i][j]))
		for p in filterResponsesWood3Tra:
		    filterResponses[i][j]=np.hstack((filterResponses[i][j],p[i][j]))
		for p in filterResponsesWaterTra:
		    filterResponses[i][j]=np.hstack((filterResponses[i][j],p[i][j]))
		for p in filterResponsesGraniteTra:
		    filterResponses[i][j]=np.hstack((filterResponses[i][j],p[i][j]))
		for p in filterResponsesMarbleTra:
		    filterResponses[i][j]=np.hstack((filterResponses[i][j],p[i][j]))
		for p in filterResponsesFloor1Tra:
		    filterResponses[i][j]=np.hstack((filterResponses[i][j],p[i][j]))
		for p in filterResponsesFloor2Tra:
		    filterResponses[i][j]=np.hstack((filterResponses[i][j],p[i][j]))
		for p in filterResponsesPebblesTra:
		    filterResponses[i][j]=np.hstack((filterResponses[i][j],p[i][j]))
		for p in filterResponsesWallTra:
		    filterResponses[i][j]=np.hstack((filterResponses[i][j],p[i][j]))
		for p in filterResponsesBrick1Tra:
		    filterResponses[i][j]=np.hstack((filterResponses[i][j],p[i][j]))
		for p in filterResponsesBrick2Tra:
		    filterResponses[i][j]=np.hstack((filterResponses[i][j],p[i][j]))
		for p in filterResponsesGlass1Tra:
		    filterResponses[i][j]=np.hstack((filterResponses[i][j],p[i][j]))
		for p in filterResponsesGlass2Tra:
		    filterResponses[i][j]=np.hstack((filterResponses[i][j],p[i][j]))
		for p in filterResponsesCarpet1Tra:
		    filterResponses[i][j]=np.hstack((filterResponses[i][j],p[i][j]))
		for p in filterResponsesCarpet2Tra:
		    filterResponses[i][j]=np.hstack((filterResponses[i][j],p[i][j]))
		for p in filterResponsesUpholsteryTra:
		    filterResponses[i][j]=np.hstack((filterResponses[i][j],p[i][j]))
		for p in filterResponsesWallpaperTra:
		    filterResponses[i][j]=np.hstack((filterResponses[i][j],p[i][j]))
		for p in filterResponsesFurTra:
		    filterResponses[i][j]=np.hstack((filterResponses[i][j],p[i][j]))
		for p in filterResponsesKnitTra:
		    filterResponses[i][j]=np.hstack((filterResponses[i][j],p[i][j]))
		for p in filterResponsesCorduroyTra:
		    filterResponses[i][j]=np.hstack((filterResponses[i][j],p[i][j]))
		for p in filterResponsesPlaidTra:
		    filterResponses[i][j]=np.hstack((filterResponses[i][j],p[i][j]))


	print("Filtros fusionados")

	tiempocarga=time.time()-tiempoi



	#Computer textons from filter
	#Textons tiene los centros de masa de cada texton
	#map es la imagen ya expresada en textones
	#Definimos el numero de textones a sacar
	
	for q in Listak:
		k=q

		tiempotextoni=time.time()

		map, textons = computeTextons(filterResponses, k)

		np.savetxt("Tiempocreaciontextones_"+"k="+str(k)+"_Pixel"+str(Npix)+".dat",np.array([time.time()-tiempotextoni]))
		print("Textones creados")

		#Guardamos los textones
		np.savetxt("Textones_"+"k="+str(k)+"_Pixel="+str(Npix)+".dat",np.array(textons))
		#Asignamos los textones


		#Bark1
		tmapBark1Tra=[]
		for i in filterResponsesBark1Tra:
		    tmapBark1Tra.append(assignTextons(i,textons.transpose()))

		print("Texton asignado")
		#Bark2
		tmapBark2Tra=[]
		for i in filterResponsesBark2Tra:
		    tmapBark2Tra.append(assignTextons(i,textons.transpose()))
		print("Texton asignado")
		#Bark3
		tmapBark3Tra=[]
		for i in filterResponsesBark3Tra:
		    tmapBark3Tra.append(assignTextons(i,textons.transpose()))
		print("Texton asignado")
		#Wood1
		tmapWood1Tra=[]
		for i in filterResponsesWood1Tra:
		    tmapWood1Tra.append(assignTextons(i,textons.transpose()))

		print("Texton asignado")
		#Wood2
		tmapWood2Tra=[]
		for i in filterResponsesWood2Tra:
		    tmapWood2Tra.append(assignTextons(i,textons.transpose()))

		print("Texton asignado")
		#Wood3
		tmapWood3Tra=[]
		for i in filterResponsesWood3Tra:
		    tmapWood3Tra.append(assignTextons(i,textons.transpose()))

		print("Texton asignado")
		#Water
		tmapWaterTra=[]
		for i in filterResponsesWaterTra:
		    tmapWaterTra.append(assignTextons(i,textons.transpose()))

		print("Texton asignado")
		#Granite
		tmapGraniteTra=[]
		for i in filterResponsesGraniteTra:
		    tmapGraniteTra.append(assignTextons(i,textons.transpose()))

		print("Texton asignado")
		#Marble
		tmapMarbleTra=[]
		for i in filterResponsesMarbleTra:
		    tmapMarbleTra.append(assignTextons(i,textons.transpose()))

		print("Texton asignado")
		#Floor1
		tmapFloor1Tra=[]
		for i in filterResponsesFloor1Tra:
		    tmapFloor1Tra.append(assignTextons(i,textons.transpose()))
		print("Texton asignado")
		#Floor2
		tmapFloor2Tra=[]
		for i in filterResponsesFloor2Tra:
		    tmapFloor2Tra.append(assignTextons(i,textons.transpose()))

		print("Texton asignado")
		#Pebbles
		tmapPebblesTra=[]
		for i in filterResponsesPebblesTra:
		    tmapPebblesTra.append(assignTextons(i,textons.transpose()))

		print("Texton asignado")
		#Wall
		tmapWallTra=[]
		for i in filterResponsesWallTra:
		    tmapWallTra.append(assignTextons(i,textons.transpose()))
		print("Texton asignado")
		#Brick1
		tmapBrick1Tra=[]
		for i in filterResponsesBrick1Tra:
		    tmapBrick1Tra.append(assignTextons(i,textons.transpose()))
		print("Texton asignado")
		#Brick2
		tmapBrick2Tra=[]
		for i in filterResponsesBrick2Tra:
		    tmapBrick2Tra.append(assignTextons(i,textons.transpose()))
		print("Texton asignado")
		#Glass1
		tmapGlass1Tra=[]
		for i in filterResponsesGlass1Tra:
		    tmapGlass1Tra.append(assignTextons(i,textons.transpose()))
		print("Texton asignado")

		#Glass2
		tmapGlass2Tra=[]
		for i in filterResponsesGlass2Tra:
		    tmapGlass2Tra.append(assignTextons(i,textons.transpose()))
		print("Texton asignado")

		#Carpet1
		tmapCarpet1Tra=[]
		for i in filterResponsesCarpet1Tra:
		    tmapCarpet1Tra.append(assignTextons(i,textons.transpose()))
		print("Texton asignado")
		#Carpet2
		tmapCarpet2Tra=[]
		for i in filterResponsesCarpet2Tra:
		    tmapCarpet2Tra.append(assignTextons(i,textons.transpose()))
		print("Texton asignado")
		#Upholstery
		tmapUpholsteryTra=[]
		for i in filterResponsesUpholsteryTra:
		    tmapUpholsteryTra.append(assignTextons(i,textons.transpose()))
		print("Texton asignado")

		#Wallpaper
		tmapWallpaperTra=[]
		for i in filterResponsesWallpaperTra:
		    tmapWallpaperTra.append(assignTextons(i,textons.transpose()))
		print("Texton asignado")
		#Fur
		tmapFurTra=[]
		for i in filterResponsesFurTra:
		    tmapFurTra.append(assignTextons(i,textons.transpose()))
		print("Texton asignado")
		#Knit
		tmapKnitTra=[]
		for i in filterResponsesKnitTra:
		    tmapKnitTra.append(assignTextons(i,textons.transpose()))

		print("Texton asignado")
		#Corduroy
		tmapCorduroyTra=[]
		for i in filterResponsesCorduroyTra:
		    tmapCorduroyTra.append(assignTextons(i,textons.transpose()))
		print("Texton asignado")
		#Plaid
		tmapPlaidTra=[]
		for i in filterResponsesPlaidTra:
		    tmapPlaidTra.append(assignTextons(i,textons.transpose()))
		print("Texton asignado")


		##################################################################################
		####################################################################################

		#Ahora sacamos los histogramas que definen cada clase
		Numerohistogramas=len(tmapBark1Tra)

		#Bark1
		histBark1Tra=histc(tmapBark1Tra[0].flatten(),np.arange(k))
		for i in range(1,Numerohistogramas):
		    histBark1Tra= histBark1Tra+histc(tmapBark1Tra[i].flatten(),np.arange(k))
		histBark1Tra=histBark1Tra/np.sum(histBark1Tra)

		print("Histograma listo")
		#Bark2
		histBark2Tra=histc(tmapBark2Tra[0].flatten(),np.arange(k))
		for i in range(1,Numerohistogramas):
		    histBark2Tra= histBark2Tra+histc(tmapBark2Tra[i].flatten(),np.arange(k))
		histBark2Tra=histBark2Tra/np.sum(histBark2Tra)

		print("Histograma listo")
		#Bark3
		histBark3Tra=histc(tmapBark3Tra[0].flatten(),np.arange(k))
		for i in range(1,Numerohistogramas):
		    histBark3Tra= histBark3Tra+histc(tmapBark3Tra[i].flatten(),np.arange(k))
		histBark3Tra=histBark3Tra/np.sum(histBark3Tra)
		    

		print("Histograma listo")
		#Wood1
		histWood1Tra=histc(tmapWood1Tra[0].flatten(),np.arange(k))
		for i in range(1,Numerohistogramas):
		    histWood1Tra= histWood1Tra+histc(tmapWood1Tra[i].flatten(),np.arange(k))
		histWood1Tra=histWood1Tra/np.sum(histWood1Tra)

		print("Histograma listo")
		#Wood2
		histWood2Tra=histc(tmapWood2Tra[0].flatten(),np.arange(k))
		for i in range(1,Numerohistogramas):
		    histWood2Tra= histWood2Tra+histc(tmapWood2Tra[i].flatten(),np.arange(k))
		histWood2Tra=histWood2Tra/np.sum(histWood2Tra)

		print("Histograma listo")
		#Wood3
		histWood3Tra=histc(tmapWood3Tra[0].flatten(),np.arange(k))
		for i in range(1,Numerohistogramas):
		    histWood3Tra= histWood3Tra+histc(tmapWood3Tra[i].flatten(),np.arange(k))
		histWood3Tra=histWood3Tra/np.sum(histWood3Tra)

		print("Histograma listo")
		#Water
		histWaterTra=histc(tmapWaterTra[0].flatten(),np.arange(k))
		for i in range(1,Numerohistogramas):
		    histWaterTra= histWaterTra+histc(tmapWaterTra[i].flatten(),np.arange(k))
		histWaterTra=histWaterTra/np.sum(histWaterTra)

		print("Histograma listo")
		#Granite
		histGraniteTra=histc(tmapGraniteTra[0].flatten(),np.arange(k))
		for i in range(1,Numerohistogramas):
		    histGraniteTra= histGraniteTra+histc(tmapGraniteTra[i].flatten(),np.arange(k))
		histGraniteTra=histGraniteTra/np.sum(histGraniteTra)

		print("Histograma listo")
		#Marble
		histMarbleTra=histc(tmapMarbleTra[0].flatten(),np.arange(k))
		for i in range(1,Numerohistogramas):
		    histMarbleTra= histMarbleTra+histc(tmapMarbleTra[i].flatten(),np.arange(k))
		histMarbleTra=histMarbleTra/np.sum(histMarbleTra)

		print("Histograma listo")
		#Floor1
		histFloor1Tra=histc(tmapFloor1Tra[0].flatten(),np.arange(k))
		for i in range(1,Numerohistogramas):
		    histFloor1Tra= histFloor1Tra+histc(tmapFloor1Tra[i].flatten(),np.arange(k))
		histFloor1Tra=histFloor1Tra/np.sum(histFloor1Tra)

		print("Histograma listo")
		#Floor2
		histFloor2Tra=histc(tmapFloor2Tra[0].flatten(),np.arange(k))
		for i in range(1,Numerohistogramas):
		    histFloor2Tra= histFloor2Tra+histc(tmapFloor2Tra[i].flatten(),np.arange(k))
		histFloor2Tra=histFloor2Tra/np.sum(histFloor2Tra)

		print("Histograma listo")
		#Pebbles
		histPebblesTra=histc(tmapPebblesTra[0].flatten(),np.arange(k))
		for i in range(1,Numerohistogramas):
		    histPebblesTra= histPebblesTra+histc(tmapPebblesTra[i].flatten(),np.arange(k))
		histPebblesTra=histPebblesTra/np.sum(histPebblesTra)

		#Wall
		histWallTra=histc(tmapWallTra[0].flatten(),np.arange(k))
		for i in range(1,Numerohistogramas):
		    histWallTra= histWallTra+histc(tmapWallTra[i].flatten(),np.arange(k))
		histWallTra=histWallTra/np.sum(histWallTra)

		print("Histograma listo")
		#Brick1
		histBrick1Tra=histc(tmapBrick1Tra[0].flatten(),np.arange(k))
		for i in range(1,Numerohistogramas):
		    histBrick1Tra= histBrick1Tra+histc(tmapBrick1Tra[i].flatten(),np.arange(k))
		histBrick1Tra=histBrick1Tra/np.sum(histBrick1Tra)

		print("Histograma listo")
		#Brick2
		histBrick2Tra=histc(tmapBrick2Tra[0].flatten(),np.arange(k))
		for i in range(1,Numerohistogramas):
		    histBrick2Tra= histBrick2Tra+histc(tmapBrick2Tra[i].flatten(),np.arange(k))
		histBrick2Tra=histBrick2Tra/np.sum(histBrick2Tra)

		print("Histograma listo")
		#Glass1
		histGlass1Tra=histc(tmapGlass1Tra[0].flatten(),np.arange(k))
		for i in range(1,Numerohistogramas):
		    histGlass1Tra= histGlass1Tra+histc(tmapGlass1Tra[i].flatten(),np.arange(k))
		histGlass1Tra=histGlass1Tra/np.sum(histGlass1Tra)

		print("Histograma listo")
		#Glass2
		histGlass2Tra=histc(tmapGlass2Tra[0].flatten(),np.arange(k))
		for i in range(1,Numerohistogramas):
		    histGlass2Tra= histGlass2Tra+histc(tmapGlass2Tra[i].flatten(),np.arange(k))
		histGlass2Tra=histGlass2Tra/np.sum(histGlass2Tra)

		print("Histograma listo")
		#Carpet1
		histCarpet1Tra=histc(tmapCarpet1Tra[0].flatten(),np.arange(k))
		for i in range(1,Numerohistogramas):
		    histCarpet1Tra= histCarpet1Tra+histc(tmapCarpet1Tra[i].flatten(),np.arange(k))
		histCarpet1Tra=histCarpet1Tra/np.sum(histCarpet1Tra)

		print("Histograma listo")
		#Carpet2
		histCarpet2Tra=histc(tmapCarpet2Tra[0].flatten(),np.arange(k))
		for i in range(1,Numerohistogramas):
		    histCarpet2Tra= histCarpet2Tra+histc(tmapCarpet2Tra[i].flatten(),np.arange(k))
		histCarpet2Tra=histCarpet2Tra/np.sum(histCarpet2Tra)

		print("Histograma listo")
		#Upholstery
		histUpholsteryTra=histc(tmapUpholsteryTra[0].flatten(),np.arange(k))
		for i in range(1,Numerohistogramas):
		    histUpholsteryTra= histUpholsteryTra+histc(tmapUpholsteryTra[i].flatten(),np.arange(k))
		histUpholsteryTra=histUpholsteryTra/np.sum(histUpholsteryTra)

		print("Histograma listo")
		#Wallpaper
		histWallpaperTra=histc(tmapWallpaperTra[0].flatten(),np.arange(k))
		for i in range(1,Numerohistogramas):
		    histWallpaperTra= histWallpaperTra+histc(tmapWallpaperTra[i].flatten(),np.arange(k))
		histWallpaperTra=histWallpaperTra/np.sum(histWallpaperTra)

		print("Histograma listo")
		#Fur
		histFurTra=histc(tmapFurTra[0].flatten(),np.arange(k))
		for i in range(1,Numerohistogramas):
		    histFurTra= histFurTra+histc(tmapFurTra[i].flatten(),np.arange(k))
		histFurTra=histFurTra/np.sum(histFurTra)

		print("Histograma listo")
		#Knit
		histKnitTra=histc(tmapKnitTra[0].flatten(),np.arange(k))
		for i in range(1,Numerohistogramas):
		    histKnitTra= histKnitTra+histc(tmapKnitTra[i].flatten(),np.arange(k))
		histKnitTra=histKnitTra/np.sum(histKnitTra)

		print("Histograma listo")
		#Corduroy
		histCorduroyTra=histc(tmapCorduroyTra[0].flatten(),np.arange(k))
		for i in range(1,Numerohistogramas):
		    histCorduroyTra= histCorduroyTra+histc(tmapCorduroyTra[i].flatten(),np.arange(k))
		histCorduroyTra=histCorduroyTra/np.sum(histCorduroyTra)

		print("Histograma listo")
		#Plaid
		histPlaidTra=histc(tmapPlaidTra[0].flatten(),np.arange(k))
		for i in range(1,Numerohistogramas):
		    histPlaidTra= histPlaidTra+histc(tmapPlaidTra[i].flatten(),np.arange(k))
		histPlaidTra=histPlaidTra/np.sum(histPlaidTra)


		print("Listos todos los histogramas")



		####################################################################################################################################
		################################################################################################################################
		#Ahora asignamos los textones al test

		#Bark1
		tmapBark1Test=[]
		for i in Bark1Test:
		    tmapBark1Test.append(assignTextons(fbRun(fb,i),textons.transpose()))

		print("Texton asignado")

		#Bark2
		tmapBark2Test=[]
		for i in Bark2Test:
		    tmapBark2Test.append(assignTextons(fbRun(fb,i),textons.transpose()))

		print("Texton asignado")

		#Bark3
		tmapBark3Test=[]
		for i in Bark3Test:
		    tmapBark3Test.append(assignTextons(fbRun(fb,i),textons.transpose()))

		print("Texton asignado")


		#Wood1
		tmapWood1Test=[]
		for i in Wood1Test:
		    tmapWood1Test.append(assignTextons(fbRun(fb,i),textons.transpose()))

		print("Texton asignado")


		#Wood2
		tmapWood2Test=[]
		for i in Wood2Test:
		    tmapWood2Test.append(assignTextons(fbRun(fb,i),textons.transpose()))

		print("Texton asignado")


		#Wood3
		tmapWood3Test=[]
		for i in Wood3Test:
		    tmapWood3Test.append(assignTextons(fbRun(fb,i),textons.transpose()))

		print("Texton asignado")


		#Water
		tmapWaterTest=[]
		for i in WaterTest:
		    tmapWaterTest.append(assignTextons(fbRun(fb,i),textons.transpose()))

		print("Texton asignado")

		#Granite
		tmapGraniteTest=[]
		for i in GraniteTest:
		    tmapGraniteTest.append(assignTextons(fbRun(fb,i),textons.transpose()))

		print("Texton asignado")

		#Marble
		tmapMarbleTest=[]
		for i in MarbleTest:
		    tmapMarbleTest.append(assignTextons(fbRun(fb,i),textons.transpose()))

		print("Texton asignado")

		#Floor1
		tmapFloor1Test=[]
		for i in Floor1Test:
		    tmapFloor1Test.append(assignTextons(fbRun(fb,i),textons.transpose()))

		print("Texton asignado")

		#Floor2
		tmapFloor2Test=[]
		for i in Floor2Test:
		    tmapFloor2Test.append(assignTextons(fbRun(fb,i),textons.transpose()))

		print("Texton asignado")

		#Pebbles
		tmapPebblesTest=[]
		for i in PebblesTest:
		    tmapPebblesTest.append(assignTextons(fbRun(fb,i),textons.transpose()))

		print("Texton asignado")


		#Wall
		tmapWallTest=[]
		for i in WallTest:
		    tmapWallTest.append(assignTextons(fbRun(fb,i),textons.transpose()))

		print("Texton asignado")


		#Brick1
		tmapBrick1Test=[]
		for i in Brick1Test:
		    tmapBrick1Test.append(assignTextons(fbRun(fb,i),textons.transpose()))

		print("Texton asignado")

		#Brick2
		tmapBrick2Test=[]
		for i in Brick2Test:
		    tmapBrick2Test.append(assignTextons(fbRun(fb,i),textons.transpose()))

		print("Texton asignado")


		#Glass1
		tmapGlass1Test=[]
		for i in Glass1Test:
		    tmapGlass1Test.append(assignTextons(fbRun(fb,i),textons.transpose()))

		print("Texton asignado")


		#Glass2
		tmapGlass2Test=[]
		for i in Glass2Test:
		    tmapGlass2Test.append(assignTextons(fbRun(fb,i),textons.transpose()))

		print("Texton asignado")


		#Carpet1
		tmapCarpet1Test=[]
		for i in Carpet1Test:
		    tmapCarpet1Test.append(assignTextons(fbRun(fb,i),textons.transpose()))

		print("Texton asignado")


		#Carpet2
		tmapCarpet2Test=[]
		for i in Carpet2Test:
		    tmapCarpet2Test.append(assignTextons(fbRun(fb,i),textons.transpose()))

		print("Texton asignado")

		#Upholstery
		tmapUpholsteryTest=[]
		for i in UpholsteryTest:
		    tmapUpholsteryTest.append(assignTextons(fbRun(fb,i),textons.transpose()))

		print("Texton asignado")

		#Wallpaper
		tmapWallpaperTest=[]
		for i in WallpaperTest:
		    tmapWallpaperTest.append(assignTextons(fbRun(fb,i),textons.transpose()))

		print("Texton asignado")


		#Fur
		tmapFurTest=[]
		for i in FurTest:
		    tmapFurTest.append(assignTextons(fbRun(fb,i),textons.transpose()))

		print("Texton asignado")

		#Knit
		tmapKnitTest=[]
		for i in KnitTest:
		    tmapKnitTest.append(assignTextons(fbRun(fb,i),textons.transpose()))

		print("Texton asignado")


		#Corduroy
		tmapCorduroyTest=[]
		for i in CorduroyTest:
		    tmapCorduroyTest.append(assignTextons(fbRun(fb,i),textons.transpose()))

		print("Texton asignado")


		#Plaid
		tmapPlaidTest=[]
		for i in PlaidTest:
		    tmapPlaidTest.append(assignTextons(fbRun(fb,i),textons.transpose()))

		print("Texton asignado")

		#######################################################################################
		#Ahora sacamos los histogramas 

		#Bark1
		histBark1Test=[]
		for i in tmapBark1Test :
		    histBark1Test.append(histc(i.flatten(),np.arange(k))/np.sum(histc(i.flatten(),np.arange(k))))
		print("Histograma listo")

		#Bark2
		histBark2Test=[]
		for i in tmapBark2Test :
		    histBark2Test.append(histc(i.flatten(),np.arange(k))/np.sum(histc(i.flatten(),np.arange(k))))
		print("Histograma listo")

		#Bark3
		histBark3Test=[]
		for i in tmapBark3Test :
		    histBark3Test.append(histc(i.flatten(),np.arange(k))/np.sum(histc(i.flatten(),np.arange(k))))
		print("Histograma listo")

		#Wood1
		histWood1Test=[]
		for i in tmapWood1Test :
		    histWood1Test.append(histc(i.flatten(),np.arange(k))/np.sum(histc(i.flatten(),np.arange(k))))
		print("Histograma listo")

		#Wood2
		histWood2Test=[]
		for i in tmapWood2Test :
		    histWood2Test.append(histc(i.flatten(),np.arange(k))/np.sum(histc(i.flatten(),np.arange(k))))
		print("Histograma listo")

		#Wood3
		histWood3Test=[]
		for i in tmapWood3Test :
		    histWood3Test.append(histc(i.flatten(),np.arange(k))/np.sum(histc(i.flatten(),np.arange(k))))
		print("Histograma listo")

		#Water
		histWaterTest=[]
		for i in tmapWaterTest :
		    histWaterTest.append(histc(i.flatten(),np.arange(k))/np.sum(histc(i.flatten(),np.arange(k))))
		print("Histograma listo")


		#Granite
		histGraniteTest=[]
		for i in tmapGraniteTest :
		    histGraniteTest.append(histc(i.flatten(),np.arange(k))/np.sum(histc(i.flatten(),np.arange(k))))
		print("Histograma listo")

		#Marble
		histMarbleTest=[]
		for i in tmapMarbleTest :
		    histMarbleTest.append(histc(i.flatten(),np.arange(k))/np.sum(histc(i.flatten(),np.arange(k))))
		print("Histograma listo")

		#Floor1
		histFloor1Test=[]
		for i in tmapFloor1Test :
		    histFloor1Test.append(histc(i.flatten(),np.arange(k))/np.sum(histc(i.flatten(),np.arange(k))))
		print("Histograma listo")

		#Floor2
		histFloor2Test=[]
		for i in tmapFloor2Test :
		    histFloor2Test.append(histc(i.flatten(),np.arange(k))/np.sum(histc(i.flatten(),np.arange(k))))
		print("Histograma listo")

		#Pebbles
		histPebblesTest=[]
		for i in tmapPebblesTest :
		    histPebblesTest.append(histc(i.flatten(),np.arange(k))/np.sum(histc(i.flatten(),np.arange(k))))
		print("Histograma listo")

		#Wall
		histWallTest=[]
		for i in tmapWallTest :
		    histWallTest.append(histc(i.flatten(),np.arange(k))/np.sum(histc(i.flatten(),np.arange(k))))
		print("Histograma listo")

		#Brick1
		histBrick1Test=[]
		for i in tmapBrick1Test :
		    histBrick1Test.append(histc(i.flatten(),np.arange(k))/np.sum(histc(i.flatten(),np.arange(k))))
		print("Histograma listo")

		#Brick2
		histBrick2Test=[]
		for i in tmapBrick2Test :
		    histBrick2Test.append(histc(i.flatten(),np.arange(k))/np.sum(histc(i.flatten(),np.arange(k))))
		print("Histograma listo")

		#Glass1
		histGlass1Test=[]
		for i in tmapGlass1Test :
		    histGlass1Test.append(histc(i.flatten(),np.arange(k))/np.sum(histc(i.flatten(),np.arange(k))))
		print("Histograma listo")

		#Glass2
		histGlass2Test=[]
		for i in tmapGlass2Test :
		    histGlass2Test.append(histc(i.flatten(),np.arange(k))/np.sum(histc(i.flatten(),np.arange(k))))
		print("Histograma listo")

		#Carpet1
		histCarpet1Test=[]
		for i in tmapCarpet1Test :
		    histCarpet1Test.append(histc(i.flatten(),np.arange(k))/np.sum(histc(i.flatten(),np.arange(k))))
		print("Histograma listo")

		#Carpet2
		histCarpet2Test=[]
		for i in tmapCarpet2Test :
		    histCarpet2Test.append(histc(i.flatten(),np.arange(k))/np.sum(histc(i.flatten(),np.arange(k))))
		print("Histograma listo")

		#Upholstery
		histUpholsteryTest=[]
		for i in tmapUpholsteryTest :
		    histUpholsteryTest.append(histc(i.flatten(),np.arange(k))/np.sum(histc(i.flatten(),np.arange(k))))
		print("Histograma listo")

		#Wallpaper
		histWallpaperTest=[]
		for i in tmapWallpaperTest :
		    histWallpaperTest.append(histc(i.flatten(),np.arange(k))/np.sum(histc(i.flatten(),np.arange(k))))
		print("Histograma listo")

		#Fur
		histFurTest=[]
		for i in tmapFurTest :
		    histFurTest.append(histc(i.flatten(),np.arange(k))/np.sum(histc(i.flatten(),np.arange(k))))
		print("Histograma listo")


		#Knit
		histKnitTest=[]
		for i in tmapKnitTest :
		    histKnitTest.append(histc(i.flatten(),np.arange(k))/np.sum(histc(i.flatten(),np.arange(k))))
		print("Histograma listo")

		#Corduroy
		histCorduroyTest=[]
		for i in tmapCorduroyTest :
		    histCorduroyTest.append(histc(i.flatten(),np.arange(k))/np.sum(histc(i.flatten(),np.arange(k))))
		print("Histograma listo")

		#Plaid
		histPlaidTest=[]
		for i in tmapPlaidTest :
		    histPlaidTest.append(histc(i.flatten(),np.arange(k))/np.sum(histc(i.flatten(),np.arange(k))))
		print("Histograma listo")

		########################################################################################################################
		########################################################################################################################
		########################################################################################################################
		#Procedemos a clasificar los datos del Test

		#Bark1
		clasiBark1=[]
		for i in histBark1Test:
		    clasiBark1.append(clasifica(i))

		#Bark2
		clasiBark2=[]
		for i in histBark2Test:
		    clasiBark2.append(clasifica(i))

		#Bark3
		clasiBark3=[]
		for i in histBark3Test:
		    clasiBark3.append(clasifica(i))

		#Wood1
		clasiWood1=[]
		for i in histWood1Test:
		    clasiWood1.append(clasifica(i))


		#Wood2
		clasiWood2=[]
		for i in histWood2Test:
		    clasiWood2.append(clasifica(i))

		#Wood3
		clasiWood3=[]
		for i in histWood3Test:
		    clasiWood3.append(clasifica(i))

		#Water
		clasiWater=[]
		for i in histWaterTest:
		    clasiWater.append(clasifica(i))

		#Granite
		clasiGranite=[]
		for i in histGraniteTest:
		    clasiGranite.append(clasifica(i))

		#Marble
		clasiMarble=[]
		for i in histMarbleTest:
		    clasiMarble.append(clasifica(i))

		#Floor1
		clasiFloor1=[]
		for i in histFloor1Test:
		    clasiFloor1.append(clasifica(i))

		#Floor2
		clasiFloor2=[]
		for i in histFloor2Test:
		    clasiFloor2.append(clasifica(i))

		#Pebbles
		clasiPebbles=[]
		for i in histPebblesTest:
		    clasiPebbles.append(clasifica(i))

		#Wall
		clasiWall=[]
		for i in histWallTest:
		    clasiWall.append(clasifica(i))

		#Brick1
		clasiBrick1=[]
		for i in histBrick1Test:
		    clasiBrick1.append(clasifica(i))

		#Brick2
		clasiBrick2=[]
		for i in histBrick2Test:
		    clasiBrick2.append(clasifica(i))

		#Glass1
		clasiGlass1=[]
		for i in histGlass1Test:
		    clasiGlass1.append(clasifica(i))

		#Glass2
		clasiGlass2=[]
		for i in histGlass2Test:
		    clasiGlass2.append(clasifica(i))

		#Carpet1
		clasiCarpet1=[]
		for i in histCarpet1Test:
		    clasiCarpet1.append(clasifica(i))

		#Carpet2
		clasiCarpet2=[]
		for i in histCarpet2Test:
		    clasiCarpet2.append(clasifica(i))

		#Upholstery
		clasiUpholstery=[]
		for i in histUpholsteryTest:
		    clasiUpholstery.append(clasifica(i))

		#Wallpaper
		clasiWallpaper=[]
		for i in histWallpaperTest:
		    clasiWallpaper.append(clasifica(i))

		#Fur
		clasiFur=[]
		for i in histFurTest:
		    clasiFur.append(clasifica(i))

		#Knit
		clasiKnit=[]
		for i in histKnitTest:
		    clasiKnit.append(clasifica(i))

		#Corduroy
		clasiCorduroy=[]
		for i in histCorduroyTest:
		    clasiCorduroy.append(clasifica(i))

		#Plaid
		clasiPlaid=[]
		for i in histPlaidTest:
		    clasiPlaid.append(clasifica(i))

		###################################################################################################################
		#####################################################################################################################
		#Creamos la matriz de confusion
		confu=np.zeros((25,25))
		aux=range(1,26)
		for i in range(len(aux)):
		    confu[i][0]=cuenta(aux[i],clasiBark1)
		    confu[i][1]=cuenta(aux[i],clasiBark2)
		    confu[i][2]=cuenta(aux[i],clasiBark3)
		    confu[i][3]=cuenta(aux[i],clasiWood1)
		    confu[i][4]=cuenta(aux[i],clasiWood2)
		    confu[i][5]=cuenta(aux[i],clasiWood3)
		    confu[i][6]=cuenta(aux[i],clasiWater)
		    confu[i][7]=cuenta(aux[i],clasiGranite)
		    confu[i][8]=cuenta(aux[i],clasiMarble)
		    confu[i][9]=cuenta(aux[i],clasiFloor1)
		    confu[i][10]=cuenta(aux[i],clasiFloor2)
		    confu[i][11]=cuenta(aux[i],clasiPebbles)
		    confu[i][12]=cuenta(aux[i],clasiWall)
		    confu[i][13]=cuenta(aux[i],clasiBrick1)
		    confu[i][14]=cuenta(aux[i],clasiBrick2)
		    confu[i][15]=cuenta(aux[i],clasiGlass1)
		    confu[i][16]=cuenta(aux[i],clasiGlass2)
		    confu[i][17]=cuenta(aux[i],clasiCarpet1)
		    confu[i][18]=cuenta(aux[i],clasiCarpet2)
		    confu[i][19]=cuenta(aux[i],clasiUpholstery)
		    confu[i][20]=cuenta(aux[i],clasiWallpaper)
		    confu[i][21]=cuenta(aux[i],clasiFur)
		    confu[i][22]=cuenta(aux[i],clasiKnit)
		    confu[i][23]=cuenta(aux[i],clasiCorduroy)
		    confu[i][24]=cuenta(aux[i],clasiPlaid)
		#Guardamos la matriz de confusion
		np.savetxt("ConfuTestvecinos_"+"k="+str(k)+"_Pixel="+str(Npix)+".dat",confu)



		#Normalizamos las columnas de la matriz de confusion

		total=[]
		for i in range(25):
		    total.append(np.sum(confu[:,i]))
		for i in range(25):
		    for j in range(25):
			confu[i][j]=float(confu[i][j])/float(total[j])

		#Calculamos el ACA
		ACA=0
		for i in range(25):
		    ACA=ACA+confu[i][i]
		ACA=float(ACA)/float(25)
		#Guardamos el ACA
		np.savetxt("ACAVecinos_"+"k="+str(k)+"_Pixel="+str(Npix)+".dat",np.array([ACA]))


		#Ahora hacemos una lista con los vectores del numero de textones de cada clase para cada una de las imagenes del Training. Esta informacion
		#la tienen los histogramas del Training, mala cosa que los fusione en uno solo.
		vectores=[]
		#Bark1
		for i in tmapBark1Tra:
		    vectores.append(histc(i.flatten(),np.arange(k)))

		#Bark2
		for i in tmapBark2Tra:
		    vectores.append(histc(i.flatten(),np.arange(k)))

		#Bark3
		for i in tmapBark3Tra:
		    vectores.append(histc(i.flatten(),np.arange(k)))

		#Wood1
		for i in tmapWood1Tra:
		    vectores.append(histc(i.flatten(),np.arange(k)))

		#Wood2
		for i in tmapWood2Tra:
		    vectores.append(histc(i.flatten(),np.arange(k)))

		#Wood3
		for i in tmapWood3Tra:
		    vectores.append(histc(i.flatten(),np.arange(k)))

		#Water
		for i in tmapWaterTra:
		    vectores.append(histc(i.flatten(),np.arange(k)))

		#Granite
		for i in tmapGraniteTra:
		    vectores.append(histc(i.flatten(),np.arange(k)))

		#Marble
		for i in tmapMarbleTra:
		    vectores.append(histc(i.flatten(),np.arange(k)))

		#Floor1
		for i in tmapFloor1Tra:
		    vectores.append(histc(i.flatten(),np.arange(k)))

		#Floor2
		for i in tmapFloor2Tra:
		    vectores.append(histc(i.flatten(),np.arange(k)))

		#Pebbles
		for i in tmapPebblesTra:
		    vectores.append(histc(i.flatten(),np.arange(k)))

		#Wall
		for i in tmapWallTra:
		    vectores.append(histc(i.flatten(),np.arange(k)))

		#Brick1
		for i in tmapBrick1Tra:
		    vectores.append(histc(i.flatten(),np.arange(k)))

		#Brick2
		for i in tmapBrick2Tra:
		    vectores.append(histc(i.flatten(),np.arange(k)))

		#Glass1
		for i in tmapGlass1Tra:
		    vectores.append(histc(i.flatten(),np.arange(k)))

		#Glass2
		for i in tmapGlass2Tra:
		    vectores.append(histc(i.flatten(),np.arange(k)))

		#Carpet1
		for i in tmapCarpet1Tra:
		    vectores.append(histc(i.flatten(),np.arange(k)))

		#Carpet2
		for i in tmapCarpet2Tra:
		    vectores.append(histc(i.flatten(),np.arange(k)))

		#Upholstery
		for i in tmapUpholsteryTra:
		    vectores.append(histc(i.flatten(),np.arange(k)))

		#Wallpaper
		for i in tmapWallpaperTra:
		    vectores.append(histc(i.flatten(),np.arange(k)))

		#Fur
		for i in tmapFurTra:
		    vectores.append(histc(i.flatten(),np.arange(k)))

		#Knit
		for i in tmapKnitTra:
		    vectores.append(histc(i.flatten(),np.arange(k)))

		#Corduroy
		for i in tmapCorduroyTra:
		    vectores.append(histc(i.flatten(),np.arange(k)))

		#Plaid
		for i in tmapPlaidTra:
		    vectores.append(histc(i.flatten(),np.arange(k)))


		#Procedemos a clasificar los datos del Training

		#Bark1
		clasiBark1=[]
		for i in vectores[0:30]:
		    clasiBark1.append(clasifica(i))

		#Bark2
		clasiBark2=[]
		for i in vectores[30:60]:
		    clasiBark2.append(clasifica(i))

		#Bark3
		clasiBark3=[]
		for i in vectores[60:90]:
		    clasiBark3.append(clasifica(i))

		#Wood1
		clasiWood1=[]
		for i in vectores[90:120]:
		    clasiWood1.append(clasifica(i))


		#Wood2
		clasiWood2=[]
		for i in vectores[120:150]:
		    clasiWood2.append(clasifica(i))

		#Wood3
		clasiWood3=[]
		for i in vectores[150:180]:
		    clasiWood3.append(clasifica(i))

		#Water
		clasiWater=[]
		for i in vectores[180:210]:
		    clasiWater.append(clasifica(i))

		#Granite
		clasiGranite=[]
		for i in vectores[210:240]:
		    clasiGranite.append(clasifica(i))

		#Marble
		clasiMarble=[]
		for i in vectores[240:270]:
		    clasiMarble.append(clasifica(i))

		#Floor1
		clasiFloor1=[]
		for i in vectores[270:300]:
		    clasiFloor1.append(clasifica(i))

		#Floor2
		clasiFloor2=[]
		for i in vectores[300:330]:
		    clasiFloor2.append(clasifica(i))

		#Pebbles
		clasiPebbles=[]
		for i in vectores[330:360]:
		    clasiPebbles.append(clasifica(i))

		#Wall
		clasiWall=[]
		for i in vectores[360:390]:
		    clasiWall.append(clasifica(i))

		#Brick1
		clasiBrick1=[]
		for i in vectores[390:420]:
		    clasiBrick1.append(clasifica(i))

		#Brick2
		clasiBrick2=[]
		for i in vectores[420:450]:
		    clasiBrick2.append(clasifica(i))

		#Glass1
		clasiGlass1=[]
		for i in vectores[450:480]:
		    clasiGlass1.append(clasifica(i))

		#Glass2
		clasiGlass2=[]
		for i in vectores[480:510]:
		    clasiGlass2.append(clasifica(i))

		#Carpet1
		clasiCarpet1=[]
		for i in vectores[510:540]:
		    clasiCarpet1.append(clasifica(i))

		#Carpet2
		clasiCarpet2=[]
		for i in vectores[540:570]:
		    clasiCarpet2.append(clasifica(i))

		#Upholstery
		clasiUpholstery=[]
		for i in vectores[570:600]:
		    clasiUpholstery.append(clasifica(i))

		#Wallpaper
		clasiWallpaper=[]
		for i in vectores[600:630]:
		    clasiWallpaper.append(clasifica(i))

		#Fur
		clasiFur=[]
		for i in vectores[630:660]:
		    clasiFur.append(clasifica(i))

		#Knit
		clasiKnit=[]
		for i in vectores[660:690]:
		    clasiKnit.append(clasifica(i))

		#Corduroy
		clasiCorduroy=[]
		for i in vectores[690:720]:
		    clasiCorduroy.append(clasifica(i))

		#Plaid
		clasiPlaid=[]
		for i in vectores[720:750]:
		    clasiPlaid.append(clasifica(i))



		#Hacemos la matriz de confusion del Training confu

		confu=np.zeros((25,25))
		for i in range(len(aux)):
		    confu[i][0]=cuenta(aux[i],clasiBark1)
		    confu[i][1]=cuenta(aux[i],clasiBark2)
		    confu[i][2]=cuenta(aux[i],clasiBark3)
		    confu[i][3]=cuenta(aux[i],clasiWood1)
		    confu[i][4]=cuenta(aux[i],clasiWood2)
		    confu[i][5]=cuenta(aux[i],clasiWood3)
		    confu[i][6]=cuenta(aux[i],clasiWater)
		    confu[i][7]=cuenta(aux[i],clasiGranite)
		    confu[i][8]=cuenta(aux[i],clasiMarble)
		    confu[i][9]=cuenta(aux[i],clasiFloor1)
		    confu[i][10]=cuenta(aux[i],clasiFloor2)
		    confu[i][11]=cuenta(aux[i],clasiPebbles)
		    confu[i][12]=cuenta(aux[i],clasiWall)
		    confu[i][13]=cuenta(aux[i],clasiBrick1)
		    confu[i][14]=cuenta(aux[i],clasiBrick2)
		    confu[i][15]=cuenta(aux[i],clasiGlass1)
		    confu[i][16]=cuenta(aux[i],clasiGlass2)
		    confu[i][17]=cuenta(aux[i],clasiCarpet1)
		    confu[i][18]=cuenta(aux[i],clasiCarpet2)
		    confu[i][19]=cuenta(aux[i],clasiUpholstery)
		    confu[i][20]=cuenta(aux[i],clasiWallpaper)
		    confu[i][21]=cuenta(aux[i],clasiFur)
		    confu[i][22]=cuenta(aux[i],clasiKnit)
		    confu[i][23]=cuenta(aux[i],clasiCorduroy)
		    confu[i][24]=cuenta(aux[i],clasiPlaid)

		
		#Guardamos la matriz de confusion
		np.savetxt("ConfuTravecinos_"+"k="+str(k)+"_Pixel="+str(Npix)+".dat",confu)
		#Guardamos el tiempo que le tomo ejecutar esta clasificacion
		np.savetxt("Tiempovecinos_"+"k="+str(k)+"_Pixel="+str(Npix)+".dat",np.array([time.time()-tiempotextoni+tiempocarga]))




		#################################################################################################################################
		#Esto en cuanto a lo que respecta a kvecinos ahora hagamos un random forest. Cada feature va a ser el numero de ocurrencias de un texton en
		#la imagen

		#Empecemos por hacer la lista de labels
		labels=[]
		for j in range(1,26):
		    for i in range(len(Bark1Tra)):
			labels.append(j)

		#Ahora armamos una lista con los datos del test
		vectoresTest=[]
		#Bark1
		for i in tmapBark1Test:
		    vectoresTest.append(histc(i.flatten(),np.arange(k)))

		#Bark2
		for i in tmapBark2Test:
		    vectoresTest.append(histc(i.flatten(),np.arange(k)))

		#Bark3
		for i in tmapBark3Test:
		    vectoresTest.append(histc(i.flatten(),np.arange(k)))

		#Wood1
		for i in tmapWood1Test:
		    vectoresTest.append(histc(i.flatten(),np.arange(k)))

		#Wood2
		for i in tmapWood2Test:
		    vectoresTest.append(histc(i.flatten(),np.arange(k)))

		#Wood3
		for i in tmapWood3Test:
		    vectoresTest.append(histc(i.flatten(),np.arange(k)))

		#Water
		for i in tmapWaterTest:
		    vectoresTest.append(histc(i.flatten(),np.arange(k)))

		#Granite
		for i in tmapGraniteTest:
		    vectoresTest.append(histc(i.flatten(),np.arange(k)))

		#Marble
		for i in tmapMarbleTest:
		    vectoresTest.append(histc(i.flatten(),np.arange(k)))

		#Floor1
		for i in tmapFloor1Test:
		    vectoresTest.append(histc(i.flatten(),np.arange(k)))

		#Floor2
		for i in tmapFloor2Test:
		    vectoresTest.append(histc(i.flatten(),np.arange(k)))

		#Pebbles
		for i in tmapPebblesTest:
		    vectoresTest.append(histc(i.flatten(),np.arange(k)))

		#Wall
		for i in tmapWallTest:
		    vectoresTest.append(histc(i.flatten(),np.arange(k)))

		#Brick1
		for i in tmapBrick1Test:
		    vectoresTest.append(histc(i.flatten(),np.arange(k)))

		#Brick2
		for i in tmapBrick2Test:
		    vectoresTest.append(histc(i.flatten(),np.arange(k)))

		#Glass1
		for i in tmapGlass1Test:
		    vectoresTest.append(histc(i.flatten(),np.arange(k)))

		#Glass2
		for i in tmapGlass2Test:
		    vectoresTest.append(histc(i.flatten(),np.arange(k)))

		#Carpet1
		for i in tmapCarpet1Test:
		    vectoresTest.append(histc(i.flatten(),np.arange(k)))

		#Carpet2
		for i in tmapCarpet2Test:
		    vectoresTest.append(histc(i.flatten(),np.arange(k)))

		#Upholstery
		for i in tmapUpholsteryTest:
		    vectoresTest.append(histc(i.flatten(),np.arange(k)))

		#Wallpaper
		for i in tmapWallpaperTest:
		    vectoresTest.append(histc(i.flatten(),np.arange(k)))

		#Fur
		for i in tmapFurTest:
		    vectoresTest.append(histc(i.flatten(),np.arange(k)))

		#Knit
		for i in tmapKnitTest:
		    vectoresTest.append(histc(i.flatten(),np.arange(k)))

		#Corduroy
		for i in tmapCorduroyTest:
		    vectoresTest.append(histc(i.flatten(),np.arange(k)))

		#Plaid
		for i in tmapPlaidTest:
		    vectoresTest.append(histc(i.flatten(),np.arange(k)))


		    
		#Ahora hacemos los arboles
		Narboles=[50,75,100,125]
		Profundidad=[50,75,100,125]

		for u in Narboles:
		    for p in Profundidad:
			rf = RandomForestClassifier(n_estimators=u,max_depth=p)
			rf.fit(vectores, labels)

			#Hacemos la prediccion del Training
			predictionTra = rf.predict(vectores)

			#Hacemos la matriz de confusion del Training confu

			confu=np.zeros((25,25))
			for i in range(len(aux)):
			    confu[i][0]=cuenta(aux[i],predictionTra[0:30])
			    confu[i][1]=cuenta(aux[i],predictionTra[30:60])
			    confu[i][2]=cuenta(aux[i],predictionTra[60:90])
			    confu[i][3]=cuenta(aux[i],predictionTra[90:120])
			    confu[i][4]=cuenta(aux[i],predictionTra[120:150])
			    confu[i][5]=cuenta(aux[i],predictionTra[150:180])
			    confu[i][6]=cuenta(aux[i],predictionTra[180:210])
			    confu[i][7]=cuenta(aux[i],predictionTra[210:240])
			    confu[i][8]=cuenta(aux[i],predictionTra[240:270])
			    confu[i][9]=cuenta(aux[i],predictionTra[270:300])
			    confu[i][10]=cuenta(aux[i],predictionTra[300:330])
			    confu[i][11]=cuenta(aux[i],predictionTra[330:360])
			    confu[i][12]=cuenta(aux[i],predictionTra[360:390])
			    confu[i][13]=cuenta(aux[i],predictionTra[390:420])
			    confu[i][14]=cuenta(aux[i],predictionTra[420:450])
			    confu[i][15]=cuenta(aux[i],predictionTra[450:480])
			    confu[i][16]=cuenta(aux[i],predictionTra[480:510])
			    confu[i][17]=cuenta(aux[i],predictionTra[510:540])
			    confu[i][18]=cuenta(aux[i],predictionTra[540:570])
			    confu[i][19]=cuenta(aux[i],predictionTra[570:600])
			    confu[i][20]=cuenta(aux[i],predictionTra[600:630])
			    confu[i][21]=cuenta(aux[i],predictionTra[630:660])
			    confu[i][22]=cuenta(aux[i],predictionTra[660:690])
			    confu[i][23]=cuenta(aux[i],predictionTra[690:720])
			    confu[i][24]=cuenta(aux[i],predictionTra[720:750])

		
			#Guardamos la matriz de confusion
			np.savetxt("ConfuTrabosque_"+"k="+str(k)+"_Pixel="+str(Npix)+"n_estimators="+str(u)+"max_depth="+str(p)+".dat",confu)


			#Hacemos la prediccion del Test
			predictionTra = rf.predict(vectoresTest)

			#Hacemos la matriz de confusion del Training confu

			confu=np.zeros((25,25))
			for i in range(len(aux)):
			    confu[i][0]=cuenta(aux[i],predictionTra[0:10])
			    confu[i][1]=cuenta(aux[i],predictionTra[10:20])
			    confu[i][2]=cuenta(aux[i],predictionTra[20:30])
			    confu[i][3]=cuenta(aux[i],predictionTra[30:40])
			    confu[i][4]=cuenta(aux[i],predictionTra[40:50])
			    confu[i][5]=cuenta(aux[i],predictionTra[50:60])
			    confu[i][6]=cuenta(aux[i],predictionTra[60:70])
			    confu[i][7]=cuenta(aux[i],predictionTra[70:80])
			    confu[i][8]=cuenta(aux[i],predictionTra[80:90])
			    confu[i][9]=cuenta(aux[i],predictionTra[90:100])
			    confu[i][10]=cuenta(aux[i],predictionTra[100:110])
			    confu[i][11]=cuenta(aux[i],predictionTra[110:120])
			    confu[i][12]=cuenta(aux[i],predictionTra[120:130])
			    confu[i][13]=cuenta(aux[i],predictionTra[130:140])
			    confu[i][14]=cuenta(aux[i],predictionTra[140:150])
			    confu[i][15]=cuenta(aux[i],predictionTra[150:160])
			    confu[i][16]=cuenta(aux[i],predictionTra[160:170])
			    confu[i][17]=cuenta(aux[i],predictionTra[170:180])
			    confu[i][18]=cuenta(aux[i],predictionTra[180:190])
			    confu[i][19]=cuenta(aux[i],predictionTra[190:200])
			    confu[i][20]=cuenta(aux[i],predictionTra[200:210])
			    confu[i][21]=cuenta(aux[i],predictionTra[210:220])
			    confu[i][22]=cuenta(aux[i],predictionTra[220:230])
			    confu[i][23]=cuenta(aux[i],predictionTra[230:240])
			    confu[i][24]=cuenta(aux[i],predictionTra[240:250])

		
			#Guardamos la matriz de confusion
			np.savetxt("ConfuTestbosque_"+"k="+str(k)+"_Pixel="+str(Npix)+"n_estimators="+str(u)+"max_depth="+str(p)+".dat",confu)

			#Normalizamos las columnas de la matriz de confusion

			total=[]
			for i in range(25):
			    total.append(np.sum(confu[:,i]))
			for i in range(25):
			    for j in range(25):
				confu[i][j]=float(confu[i][j])/float(total[j])

			#Calculamos el ACA
			ACA=0
			for i in range(25):
			    ACA=ACA+confu[i][i]
			ACA=float(ACA)/float(25)
			#Guardamos el ACA
			np.savetxt("ACAbosque_"+"k="+str(k)+"_Pixel="+str(Npix)+"n_estimators="+str(u)+"max_depth="+str(p)+".dat",np.array([ACA]))















