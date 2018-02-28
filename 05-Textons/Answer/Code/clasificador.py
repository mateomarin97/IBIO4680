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

#Definimos el ancho y largo de las imagenes
ancho=480
largo=640

#No vamos a trabajar con toda la imagen, es muy grande. Vamos a agarrar un cuadro de 80x80 en la mitad de ella 
Npix=80
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




###########################################################################################################################


#Definimos el numero de textones a sacar
k=5*25
from fbRun import fbRun
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

#Computer textons from filter
#Textons tiene los centros de masa de cada texton
#map es la imagen ya expresada en textones
from computeTextons import computeTextons
map, textons = computeTextons(filterResponses, k)

print("Textones creados")
#Asignamos los textones
from assignTextons import assignTextons

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

#Esta funcion pasa la matriz de textones a un histograma
def histc(X, bins):
    map_to_bins = np.digitize(X,bins)
    r = np.zeros(bins.shape)
    for i in map_to_bins:
        r[i-1] += 1
    return np.array(r)
####################################################################################

#Ahora sacamos los histogramas que definen cada clase
Factor=tmapBark1Tra[0].size
Numerohistogramas=len(tmapBark1Tra)

#Bark1
histBark1Tra=histc(tmapBark1Tra[0].flatten(),np.arange(k))/Factor
for i in range(1,Numerohistogramas):
    histBark1Tra= histBark1Tra+histc(tmapBark1Tra[i].flatten(),np.arange(k))/Factor
histBark1Tra=histBark1Tra/Numerohistogramas


#Bark2
histBark2Tra=histc(tmapBark2Tra[0].flatten(),np.arange(k))/Factor
for i in range(1,Numerohistogramas):
    histBark2Tra= histBark2Tra+histc(tmapBark2Tra[i].flatten(),np.arange(k))/Factor
histBark2Tra=histBark2Tra/Numerohistogramas


#Bark3
histBark3Tra=histc(tmapBark3Tra[0].flatten(),np.arange(k))/Factor
for i in range(1,Numerohistogramas):
    histBark3Tra= histBark3Tra+histc(tmapBark3Tra[i].flatten(),np.arange(k))/Factor
histBark3Tra=histBark3Tra/Numerohistogramas
    
#np.savetxt("Histograma.dat",np.array(histBark3Tra))

#Wood1
histWood1Tra=histc(tmapWood1Tra[0].flatten(),np.arange(k))/Factor
for i in range(1,Numerohistogramas):
    histWood1Tra= histWood1Tra+histc(tmapWood1Tra[i].flatten(),np.arange(k))/Factor
histWood1Tra=histWood1Tra/Numerohistogramas

#Wood2
histWood2Tra=histc(tmapWood2Tra[0].flatten(),np.arange(k))/Factor
for i in range(1,Numerohistogramas):
    histWood2Tra= histWood2Tra+histc(tmapWood2Tra[i].flatten(),np.arange(k))/Factor
histWood2Tra=histWood2Tra/Numerohistogramas


#Wood3
histWood3Tra=histc(tmapWood3Tra[0].flatten(),np.arange(k))/Factor
for i in range(1,Numerohistogramas):
    histWood3Tra= histWood3Tra+histc(tmapWood3Tra[i].flatten(),np.arange(k))/Factor
histWood3Tra=histWood3Tra/Numerohistogramas

#Water
histWaterTra=histc(tmapWaterTra[0].flatten(),np.arange(k))/Factor
for i in range(1,Numerohistogramas):
    histWaterTra= histWaterTra+histc(tmapWaterTra[i].flatten(),np.arange(k))/Factor
histWaterTra=histWaterTra/Numerohistogramas


#Granite
histGraniteTra=histc(tmapGraniteTra[0].flatten(),np.arange(k))/Factor
for i in range(1,Numerohistogramas):
    histGraniteTra= histGraniteTra+histc(tmapGraniteTra[i].flatten(),np.arange(k))/Factor
histGraniteTra=histGraniteTra/Numerohistogramas


#Marble
histMarbleTra=histc(tmapMarbleTra[0].flatten(),np.arange(k))/Factor
for i in range(1,Numerohistogramas):
    histMarbleTra= histMarbleTra+histc(tmapMarbleTra[i].flatten(),np.arange(k))/Factor
histMarbleTra=histMarbleTra/Numerohistogramas

#Floor1
histFloor1Tra=histc(tmapFloor1Tra[0].flatten(),np.arange(k))/Factor
for i in range(1,Numerohistogramas):
    histFloor1Tra= histFloor1Tra+histc(tmapFloor1Tra[i].flatten(),np.arange(k))/Factor
histFloor1Tra=histFloor1Tra/Numerohistogramas
