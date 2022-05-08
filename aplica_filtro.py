#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May  6 22:47:10 2022

@author: emmanuel
"""
#https://programmerclick.com/article/6115655816/
import numpy as np
#from matplotlib import pyplot as plt
#https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.signal.butter.html
from scipy import signal
#Filter.py
iSampleRate = 2000					# Frecuencia de muestreo, 2000 muestras por segundo
x = np.fromfile("ecgsignal.dat",dtype=np.float32)

# Realizar filtrado de paso de banda
b,a = butterBandPassFilter(3,70,iSampleRate,order=4)
x = signal.lfilter(b,a,x)

# Realizar banda para detener el filtrado
b,a = butterBandStopFilter(48,52,iSampleRate,order=2)
x = signal.lfilter(b,a,x)

...  # Se abrevia el análisis de espectro y parte del dibujo del código
