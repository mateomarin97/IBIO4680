import numpy as np

def buscamaximo(A):
    maximo=0
    for i in range(A.shape[0]):
        for j in range(A.shape[1]):
            if(A[i][j]>maximo):
                maximo=A[i][j]
    return maximo

def contarocurrencias(A,n):
    cont=0
    for i in range(A.shape[0]):
        for j in range(A.shape[1]):
            if(A[i][j]==n):
                cont=cont+1
    return cont

def contarocurrenciasconjuntas(A,na,B,nb):
    cont=0
    for i in range(A.shape[0]):
        for j in range(A.shape[1]):
            if(A[i][j]==na and B[i][j]==nb):
                cont=cont+1
    return cont

def indiceJacard(A,na,B,nb):
    interseccion=contarocurrenciasconjuntas(A,na,B,nb)
    union=contarocurrencias(A,na)+contarocurrencias(B,nb)-interseccion
    jacard=float(interseccion)/float(union)
    return jacard
