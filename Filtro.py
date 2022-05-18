import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

# =============================================================================
#                   Filtro análogico 
# =============================================================================
#                   Filtor rechaza banda 
#Frecuencia central en Hz

fo = 60

#Wo 
wo = fo*np.pi*2
#Frecuencia inferior en Hz 
fL = 59.5
#wl
wL = 2*np.pi*fL
#Frecuencia superior 
wH = wo**2/wL
#Ancho de banda 
B = wH - wL
#De otra forma 
"""
Q=1
B=wo/Q
"""
#B = wo/Q donde Q es el factor de calidad 5 < Q <50 
#Númerador de la función de transferencia 
num = [1,0,wo**2]
#Denominador 
den = [1,B,wo**2]
#Respuesta en frecuencia del filtro analógico 
w, Hw = signal.freqs(num,den)

#Convertir w (rad/seg) a heartz
fhz = w/(2*np.pi)
#Convertir Hw a magnitud
Hwm = np.abs(Hw)
#
plt.figure(1)
plt.plot(fhz,Hwm)
plt.title("Espectro de magnitud")
plt.xlabel("Frecuencia en Hz")
###########################################################################33
#       DISEÑO DEL FILTRO DIGITAL
#Frecuencia de muestreo den Hz
fs = 1200
#Periodo de muestreo
ts = 1/fs
#Numerador de la funcion de transferencia de H(z)
numz = [4+wo**2*ts**2,2*wo**2*ts**2-8,4+wo**2*ts**2]
#Denominador de la funcion de transferencias de H(z)
denz = [4+2*B*ts+wo**2*ts**2,2*wo**2*ts**2-8,4-2*B*ts+wo**2*ts**2]
#Respuesta en frecuencia del filtro digital
wz, Hwz = signal.freqz(numz,denz)
#Convierte wz a frecuencia digital a normalizada
F = wz/(2*np.pi)
#Convierte frecuencia digital a Hertz
f = F*fs
#Se obtiene el modulo de Hwz
Hwzm = np.abs(Hwz)
#Grafico
plt.plot(f,Hwzm)
#Pag 691 transofmacion lineal

 
