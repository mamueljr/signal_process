# -*- coding: utf-8 -*-
"""
Editor de Spyder

Este es un archivo temporal.
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate

######################################################################
#Abre señal de ECG (Base de datos Physionet)
file = open('ecg.txt',"r")
#Reemplaza el \n por un espacio ' '
openedFile = file.read().replace('\n',' ')
#Divide los datos y genera una lista con datos del tipo string
data_str = openedFile.split(' ')
#Convierte los datos del tipo string a un tipo entero
data_num = list(map(int,data_str))
#Cantidad de datos en el archivo
array_length = len(data_num)
#Declaramos dos arreglos con ceros del tamaño de array_length
filtered_signal = np.zeros(int(array_length/2))
raw_signal = np.zeros(int(array_length/2))
#Ciclo para separar señales
i = 0
j = 0
while(j < array_length):
    raw_signal[i] = data_num[j]
    filtered_signal[i] = data_num[j+1]
    j += 2
    i += 1
#Gráficos
plt.figure(1)
plt.subplot(2,1,1)
plt.plot(raw_signal)
plt.title('Señal de ECG pura y filtrada')
plt.subplot(2,1,2)
plt.plot(filtered_signal)
######################################################################
#                   EXTRAE UN CICLO CARDÍACO
#Extrae ciclo cardíaco de la señal filtrada
signal = raw_signal[5900:6300]
#Número de muestras en el ciclo cardíaco
M = len(signal)
#Condición de periodicidad
signal[M-1] = signal[0]
#Frecuencia de muestreo en Hertz
fs = 500
#Periodo de muestreo
ts = 1/fs
#Vector de tiempos
l = np.arange(0,M)
t = l*ts 
#Gráfico
plt.figure(2)
plt.plot(t,signal)
plt.title('Ciclo Cardíaco')
plt.xlabel('Tiempo (seg)')
######################################################################
#                         Análisis en Frecuencia
#Especificar el número de muestras en la señal (siempre impar)
N = 300
#Vector de tiempos para la interpolación
te = np.linspace(0,t[-1],N)
#Aplica método de interpolación
inter = interpolate.splrep(t,signal)
ye = interpolate.splev(te,inter)
#Gráfica
plt.figure(2)
plt.plot(te,ye,'r.')
#Número de trapecios
Nt = N-1
#Factor multiplicativo de las sumatorias
fact = 2/Nt
#Incremento en radianes 
inc = 2*np.pi/Nt
#Vector de ángulos para cada una de las muestras
tn = np.linspace(0,2*np.pi-inc,Nt)
#Gráfico
plt.figure(3)
plt.plot(tn*360/(2*np.pi),ye[0:N-1])
plt.title('Ciclo Cardíaco')
plt.xlabel('Ángulo (grados)')
#Cálcula el coeficiente a0
a0 = fact*np.sum(ye[0:Nt])
#Número de componentes de frecuencia
K = int(Nt/2+1)
#Periodo de la señal
T = M*ts
#Frecuencia fundamental
fo = 1/T
#Inicializa arreglos para ak,bk,ck,phik y fk
ak = np.zeros(K)
bk = np.zeros(K)
ck = np.zeros(K)
phik = np.zeros(K)
fk = np.zeros(K)
#Asignar el valor de a0 en ak y ck
ck[0] = np.abs(a0)
ak[0] = a0
#Calcula los coeficientes de la serie de Fourier
for k in range(1,K):
    #Coeficientes Rectangulares
    ak[k] = fact*np.sum(ye[0:Nt]*np.cos(k*tn))
    bk[k] = fact*np.sum(ye[0:Nt]*np.sin(k*tn))
    
    #Coeficientes Polares
    ck[k] = np.sqrt(ak[k]**2 + bk[k]**2)
    #Primer cuadrante
    if ak[k] >= 0 and bk[k] >= 0:
        phik[k] = np.arctan(bk[k]/ak[k])
    
    #Segundo cuadrante
    if ak[k] < 0 and bk[k] >= 0:
        phik[k] = np.pi - np.arctan(bk[k]/np.abs(ak[k]))

    #Tercer cuadrante
    if ak[k] < 0 and bk[k] < 0:
        phik[k] = np.pi + np.arctan(np.abs(bk[k])/np.abs(ak[k]))

    #Cuarto cuadrante
    if ak[k] >= 0 and bk[k] < 0:
        phik[k] = np.arctan(bk[k]/ak[k])
    #Múltiplos de la frecuencia fundamental
    fk[k] = k*fo
    #Volver 0 el componente fk cuando su valor es 50 hz
    if fk[k] == 50:
        ck[k]=0
        #fk[k]=0
        ak[k]=0
        bk[k]=0
        phik[k]=0
            

#Gráfico
plt.figure(4)
plt.subplot(2,1,1)
#Volver 0 el componente kf cuando su valor es 50 hz
plt.stem(fk,ak)
plt.title('Espectro de Magnitud para "ak y bk"')
plt.subplot(2,1,2)
plt.stem(fk,bk)
plt.xlabel('Frecuencia en Hertz')    
    
plt.figure(5)
plt.subplot(2,1,1)
plt.stem(fk,ck)
plt.title('Espectro de Magnitud y Fase')
plt.subplot(2,1,2)
plt.stem(fk,phik*180/np.pi)
plt.xlabel('Frecuencia en Hertz')

plt.figure(6)
plt.subplot(2,1,1)
plt.plot(fk,ck)
plt.title('Espectro de Magnitud y Fase')
plt.subplot(2,1,2)
plt.plot(fk,phik*180/np.pi)
plt.xlabel('Frecuencia en Hertz')
#Hacer phik ck bk y ak cero para eliminar el pido de 50 hrtz

################################################################
#reconstruccion de la señal
#basarse en el script 4

#generar vector tamaño M que valla desde 0<= t <=2pi
t=len(signal)
#substituir L por M
xr = np.zeros(M)
# Evaluamos la serie de Fourier Rectangular
# OJO quitar el valor de cero en la parte de bk para otra señal
for k in range(1, K + 2):
    xr = xr + ak[k + K + 1] * np.cos(2 * np.pi * k * fo * t) + bk[k + K + 1] * 0 * np.sin(2 * np.pi * k * fo * t)

# Agregamos el offset a la señal
xr = xr + a0

# Gráfico
plt.figure(4)
plt.plot(t, xt, t, xr)
plt.title('Señal Original y Reconstruida')
plt.xlabel('Tiempo (seg)')

 



