# -*- coding: utf-8 -*-
"""
Created on Fri Apr 19 09:40:36 2024

@author: julia
"""
#------------------------------------------------------------------------------- 
#Librerias
#-------------------------------------------------------------------------------

import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd

#------------------------------------------------------------------------------- 
#Parte 1 leer el archivo csv
#-------------------------------------------------------------------------------

archivo1= "muon.csv"
datos1= pd.read_csv(archivo1,sep=',')

#------------------------------------------------------------------------------- 
#Parte 2 tratamiento de datos
#-------------------------------------------------------------------------------

datos1.rename(columns={'40000':"Tiempo de decaimiento"},inplace=True)
datos1= datos1.shift()

datos1.loc[0]=[40000]
filtro1= datos1.loc[datos1['Tiempo de decaimiento']>=9900]
datos1=datos1.drop(filtro1.index)
datos1=datos1.sort_values(by='Tiempo de decaimiento',ascending=True)
tiempo_decaimiento1 = datos1['Tiempo de decaimiento'].values
tiempos1, particulas1= np.unique(tiempo_decaimiento1, return_counts=True)
tiempos1= tiempos1/1000
suma1= np.cumsum(particulas1)
total1= np.max(suma1)
decaimiento1= total1-suma1
decaimiento1= decaimiento1[:-1]
tiempos1= tiempos1[:-1]



#-------------------------------------------------------------------------------
ajuste1= np.log(decaimiento1/total1)
A1= np.vstack([-tiempos1, np.ones(len(-tiempos1))]).T
m1,c1= np.linalg.lstsq(A1,ajuste1, rcond=None)[0]

matriz_covarianza= np.linalg.inv(A1.T@A1)
incertidumbre_m1= np.sqrt(matriz_covarianza[0,0])
tau1=1/m1

incertidumbre_tau= (1/m1**2)*incertidumbre_m1

#-------------------------------------------------------------------------------


plt.scatter(tiempos1, decaimiento1, color= "blue")
plt.plot(tiempos1,total1*np.exp(-tiempos1*m1), color="blue")
plt.xlabel("Tiempo [μs]")
plt.ylabel("Número de particulas")
