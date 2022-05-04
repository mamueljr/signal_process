# -*- coding: utf-8 -*-
"""
Created on Mon Mar 14 11:05:55 2022

@author: alain

Series de Fourier (Dominio del tiempo Continuo)

"""

import numpy as np
import matplotlib.pyplot as plt
import sys

##############################################################
#                DEFINICIÓN DE LA FUNCIÓN x(t)
#   x(t) = 1, -1/2 < t < 1/2  y   x(t) = 0,  1/2 < t < 3/2

# Incremento de tiempo
dt = 0.01
# Tiempo inicial
t_min = -1 / 2
# Tiempo final
t_max = 3 / 2
# vector de tiempos
t = np.arange(t_min, t_max, dt)
# Tamaño del vector de tiempos
L = len(t)
# Vector de magnitudes para x(t)
xt = np.ones(L)
xt[round(L / 2):L] = 0
# Gráfico
plt.figure(1)
plt.plot(t, xt)
plt.title('Señal Cuadrada (Función x(t))')
plt.xlabel('Tiempo (seg)')
#############################################################
#       ANÁLISIS EN FRECUENCIA (SERIES DE FOURIER)
# Periodo de la señal
T = t_max - t_min
# Frecuencia fundamental
fo = 1 / T
# Coeficiente a0 (Offset)
a0 = 1 / 2
# Definimos el armónicos (Impulsos que tendremos en los espectros)
K = 10
# Inicializamos 4 vectores para ak, bk, phik y ck
ak = np.zeros(2 * K + 3)
bk = np.zeros(2 * K + 3)
ck = np.zeros(2 * K + 3)
phik = np.zeros(2 * K + 3)
# Vcetor de frecuencias
f = np.zeros(2 * K + 3)
# Cálculo de los coeficientes de Fourier
for k in range(-K - 1, K + 2):

    # Verifica signo de la componente de offset
    if a0 < 0:
        phik[k + K + 1] = np.pi
    else:
        phik[k + K + 1] = 0

    # Caso cuando k = 0
    if k == 0:
        ak[k + K + 1] = a0
        ck[k + K + 1] = np.abs(a0)

    else:
        ak[k + K + 1] = np.sin(np.pi * k / 2) / (np.pi * k / 2)
        # Para resolver el problema de "NAN" "no a number"
        # epsilon = 2.22 e-16 aproximado a 0
        bk[k + K + 1] = sys.float_info.epsilon
        ck[k + K + 1] = np.sqrt(ak[k + K + 1] ** 2 + bk[k + K + 1] ** 2)

        # Cómputo del ángulo
        # Primer cuadrante
        if ak[k + K + 1] >= 0 and bk[k + K + 1] >= 0:
            phik[k + K + 1] = np.arctan(bk[k + K + 1] / ak[k + K + 1])

        # Segundo cuadrante
        if ak[k + K + 1] < 0 and bk[k + K + 1] >= 0:
            phik[k + K + 1] = np.pi - np.arctan(bk[k + K + 1] / np.abs(ak[k + K + 1]))

        # Tercer cuadrante
        if ak[k + K + 1] < 0 and bk[k + K + 1] < 0:
            phik[k + K + 1] = np.pi + np.arctan(np.bs(bk[k + K + 1]) / np.abs(ak[k + K + 1]))

        # Cuarto cuadrante
        if ak[k + K + 1] >= 0 and bk[k + K + 1] < 0:
            phik[k + K + 1] = np.arctan(bk[k + K + 1] / ak[k + K + 1])

        # Cálculo de las Frecuencias
        f[k + K + 1] = k * fo

        # ángulo en grados
        phik[k + K + 1] = phik[k + K + 1] * 180 / np.pi

    # Gráfico
plt.figure(2)
plt.subplot(2, 1, 1)
plt.stem(f, ak)
plt.title('Espectro de Magnitud para "ak y bk"')
plt.subplot(2, 1, 2)
plt.stem(f, bk * 0)
plt.xlabel('Frecuencia en Hertz')

plt.figure(3)
plt.subplot(2, 1, 1)
plt.stem(f, ck)
plt.title('Espectro de Magnitud y Fase')
plt.subplot(2, 1, 2)
plt.stem(f, phik)
plt.xlabel('Frecuencia en Hertz')

#############################################################################
#                   RECONSTRUCCIÓN DE LA SEÑAL
# Vector para la señal reconstruída
xr = np.zeros(L)
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

##################################################################
# TRANSFORMADA DE FOURIER CONTINUA

f_max = f[-1]
# Vector de frecuencias (continuo)
fc = np.arange(-f_max, f_max, 0.001)
# Evaluamos el resultado de la transformada de fourier continua
# Usar teorema de pitagoraspara otro caso donde exista Re + i Imaginaria
Xf = np.abs(np.sin(np.pi * fc) / (np.pi * fc))
# Grafico
plt.figure(7)
plt.stem(f, ck)
plt.plot(fc, Xf, 'r')
plt.title('Transformada de fourier continua')
plt.xlabel('Frecuencia Hertz')

####################################################################
# TRANSFORMADA DE FOURIER DE TIEMPO DISCRETO
# Periodo de muestreo
ts = dt
# Vector de tiempos discretos
kts = t
# Señal discreta
x_kts = xt

# Grafico
plt.figure(8)
plt.plot(kts, x_kts, '.')
plt.title('Transformada de fourier continua')
plt.xlabel('Frecuencia Hertz')

# Frecuencia de muestreo
fs = 1 / ts
# Ancho de banda de la señal
B = fs / 2
# Vector de frecuencias (3 triangulos o espectros)
# fz = np.arange(-fs-B,fs+B,0.01)
# Para ver solo espectro central
fz = np.arange(-B, B, 0.01)
# Frecuencia normalizada
F = fz / fs
# Inicializa dos vectores para la TFTD
x_real = np.zeros(len(F))
x_imag = np.zeros(len(F))
# Evakuamos la TFTD
for k in range(0, L):
    x_real = x_real + x_kts[k] * np.cos(2 * np.pi * k * F)
    x_imag = x_imag + x_kts[k] * np.sin(2 * np.pi * k * F)
# Involucramos el termino ts de la sumatoria
x_real = ts * x_real
x_imag = ts * x_imag
# Encontrar el modulo
Xf_mod = np.sqrt(x_real ** 2 + x_imag ** 2)
# Grafico
plt.figure(9)
plt.plot(fz, Xf_mod)
plt.plot(fc, Xf, 'r')
plt.title('Transformada de fourier de tiempo discreto')
plt.xlabel('Frecuencia Hertz')
