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
    Bark1Tra.append( Image.open(r'./Data/T01_bark1Tra/'+str(i)).convert("L").crop((csi0,csi1,cid0,cid1)))

    
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
    Bark2Tra.append( Image.open(r'./Data/T02_bark2Tra/'+str(i)).convert("L").crop((csi0,csi1,cid0,cid1)))


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
    Bark3Tra.append( Image.open(r'./Data/T03_bark3Tra/'+str(i)).convert("L").crop((csi0,csi1,cid0,cid1)))



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
    Wood1Tra.append( Image.open(r'./Data/T04_wood1Tra/'+str(i)).convert("L").crop((csi0,csi1,cid0,cid1)))



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
    Wood2Tra.append( Image.open(r'./Data/T05_wood2Tra/'+str(i)).convert("L").crop((csi0,csi1,cid0,cid1)))


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
    Wood3Tra.append( Image.open(r'./Data/T06_wood3Tra/'+str(i)).convert("L").crop((csi0,csi1,cid0,cid1)))

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
    WaterTra.append( Image.open(r'./Data/T07_waterTra/'+str(i)).convert("L").crop((csi0,csi1,cid0,cid1)))



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
    GraniteTra.append( Image.open(r'./Data/T08_graniteTra/'+str(i)).convert("L").crop((csi0,csi1,cid0,cid1)))


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
    MarbleTra.append( Image.open(r'./Data/T09_marbleTra/'+str(i)).convert("L").crop((csi0,csi1,cid0,cid1)))

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
    Floor1Tra.append( Image.open(r'./Data/T10_floor1Tra/'+str(i)).convert("L").crop((csi0,csi1,cid0,cid1)))


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
    Floor2Tra.append( Image.open(r'./Data/T11_floor2Tra/'+str(i)).convert("L").crop((csi0,csi1,cid0,cid1)))


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
    PebblesTra.append( Image.open(r'./Data/T12_pebblesTra/'+str(i)).convert("L").crop((csi0,csi1,cid0,cid1)))


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
    WallTra.append( Image.open(r'./Data/T13_wallTra/'+str(i)).convert("L").crop((csi0,csi1,cid0,cid1)))


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
    Brick1Tra.append( Image.open(r'./Data/T14_brick1Tra/'+str(i)).convert("L").crop((csi0,csi1,cid0,cid1)))


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
    Brick2Tra.append( Image.open(r'./Data/T15_brick2Tra/'+str(i)).convert("L").crop((csi0,csi1,cid0,cid1)))


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
    Glass1Tra.append( Image.open(r'./Data/T16_glass1Tra/'+str(i)).convert("L").crop((csi0,csi1,cid0,cid1)))


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
    Glass2Tra.append( Image.open(r'./Data/T17_glass2Tra/'+str(i)).convert("L").crop((csi0,csi1,cid0,cid1)))


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
    Carpet1Tra.append( Image.open(r'./Data/T18_carpet1Tra/'+str(i)).convert("L").crop((csi0,csi1,cid0,cid1)))


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
    Carpet2Tra.append( Image.open(r'./Data/T19_carpet2Tra/'+str(i)).convert("L").crop((csi0,csi1,cid0,cid1)))


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
    UpholsteryTra.append( Image.open(r'./Data/T20_upholsteryTra/'+str(i)).convert("L").crop((csi0,csi1,cid0,cid1)))


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
    WallpaperTra.append( Image.open(r'./Data/T21_wallpaperTra/'+str(i)).convert("L").crop((csi0,csi1,cid0,cid1)))


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
    FurTra.append( Image.open(r'./Data/T22_furTra/'+str(i)).convert("L").crop((csi0,csi1,cid0,cid1)))


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
    KnitTra.append( Image.open(r'./Data/T23_knitTra/'+str(i)).convert("L").crop((csi0,csi1,cid0,cid1)))


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
    CorduroyTra.append( Image.open(r'./Data/T24_corduroyTra/'+str(i)).convert("L").crop((csi0,csi1,cid0,cid1)))


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
    PlaidTra.append( Image.open(r'./Data/T25_plaidTra/'+str(i)).convert("L").crop((csi0,csi1,cid0,cid1)))




