# -*- coding: utf-8 -*-
"""
Created on Mon Apr 25 12:13:19 2022

@author: alain
"""

import scipy.io.wavfile as rd
import numpy as np
import matplotlib.pyplot as plt
from mlp_toolkits.mplot3d.axes3d import Axes3D

class espectrograma:
    
    def __init__(self, audio_signal,fs,frame_size,overlap):
        self.audio = audio_signal
        self.sampling_freq = fs
        self.monoaural = self.__Stereo()
        self.norm = self.__Norma()
        self.length = len(self.norm)
        self.number_frame_samples = int(np.fix(frame_size * fs))
        self.number_overlap_samples = int(np.fix(self.number_frame_samples-self.number_frame_samples*overlap/100))
        self.number_frames = self.__nFrames()
        self.half_samples = int(np.fix(self.number_frame_samples/2.0))
        self.Hann_window = self.__Window()
        self.freq = self.__Frequency()
        self.spect = self.__Spect()
        
        
        
    def __Stereo(self):
        n_channels = len(np.shape(self.audio))
        if n_channels > 1:
            temp_1 = self.audio[:,0]
            temp_2 = self.audio[:,1]
            avg = (temp_1 + temp_2)/2
        else:
            avg = self.audio
        return avg
    
    def __Norma(self):
        mean = np.mean(self.monoaural)
        std = np.std(self.monoaural)
        norm_signal = (self.monoaural-mean)/std
        return norm_signal
        
    def __nFrames(self):
        cont = 0
        i = 0
        while(cont < self.length):
            cont = i*self.number_overlap_samples + self.number_frame_samples
            i += 1
        return i - 1 
    
    def __Window(self):
        W = np.zeros(self.number_frame_samples)
        mod = self.number_frame_samples%2
        if mod == 0:
            end = self.half_samples
        else:
            end = self.half_samples + 1
        
        for n in range(-self.half_samples,end):
            W[n + self.half_samples] = 0.5 + 0.5*np.cos(2*np.pi*n/self.number_frame_samples)
        return W
        
    def __Frequency(self):
        mod = self.number_frame_samples%2
        if mod == 0:
            end = self.half_samples
            f = np.zeros(end)
        else:
            end = self.half_samples + 1
            f = np.zeros(end)
            
        for n in range(0,end):
            f[n] = (n*fs)/(self.number_frame_samples-1)
        return f
        
    def __Spect(self):
        frame_data = np.zeros(self.number_frame_samples)
        
        mod = self.number_frame_samples%2
        if mod == 0:
            end = self.half_samples
        else:
            end = self.half_samples + 1
        
        energy_matrix = np.zeros((end,self.number_frames))
        
        for n in range(0,self.number_frames):
            frame_data = self.norm[n*self.number_overlap_samples:n*self.number_overlap_samples + self.number_frame_samples]
            #plt.figure(5)
            #plt.subplot(211)
            #plt.plot(frame_data)
            
            #Verifica si el frame tiene energía cero
            energy = np.sum(frame_data**2)
            if energy != 0:
                frame_data = frame_data * self.Hann_window
                #plt.subplot(212)
                #plt.plot(frame_data)
                
                ftd = np.fft.fft(frame_data)
                mag = np.abs(ftd)
                #plt.figure(6)
                #plt.plot(self.freq,mag[0:end])
                #plt.show()
                
                
                
                energy_matrix[0:end,n] = 10*np.log10(mag[0:end])
                
                
        return energy_matrix                
        
#Abre archivo de audio
fs, audio_signal = rd.read('04_Electric_Mixer_Food.wav')
#Parámetros de usuario
#Tamaño de frame en segundos
frame_size = 0.050
#Traslape entre frames en porcentaje
overlap = 80


#Crea objeto de la clase espectrograma
spectro = espectrograma(audio_signal,fs,frame_size,overlap)

#Despliega en consola la frecuencia de muestreo
print('Frecuencia de Muestreo: ',spectro.sampling_freq)
#Gráfica de la señal
plt.figure(1) 
plt.subplot(211)       
#plt.plot(spectro.audio[0:-1,0])
plt.title('Gráfica del Canal Izq y Der')
plt.subplot(212) 
#plt.plot(spectro.audio[0:-1,1])
plt.xlabel('Muestras')
#Gráfica de la señal promediada
plt.figure(2)
plt.plot(spectro.monoaural)
plt.title('Señal Monoaural')
plt.xlabel('Muestras')
#Gráfica de la señal normalizada
plt.figure(3)
plt.plot(spectro.norm)
plt.title('Señal Normalizada')
plt.xlabel('Muestras')

print('Tamaño de la señal en muestras: ',spectro.length)
print('Tamaño del frame en muestras: ',spectro.number_frame_samples)
print('Traslape en muestras: ',spectro.number_overlap_samples)
print('Número de Frames para analizar: ',spectro.number_frames)
print('Mitad de muestras: ',spectro.half_samples)

#Gráfica de la señal normalizada
plt.figure(4)
plt.plot(spectro.Hann_window)
plt.title('Función Ventana de Hann')
plt.xlabel('Muestras')

print('Frecuencias en el espectrograma: ',spectro.freq)


mod = spectro.number_frame_samples%2
if mod == 0:
    end = spectro.half_samples
else:
    end = spectro.half_samples +1     

plt.figure(7)
plt.imshow(spectro.spect[::-1],cmap = plt.get_cmap('jet'),
           extent=[0,spectro.number_frames,0,spectro.freq[-1]],
           aspect="auto")
plt.xlabel('Número de Frames')
plt.ylabel('Frecuencia en Hertz')
plt.title('Espectrograma')

#Grafico en 3d
fig = plt.figure(8)
axes3d = Axes3D(fig)
x = np.linspace(0,spectro.number_frames/overlap,spectro.numbrer_frames)
y=np.linspace(spectro.freq[-1],0,end)
X,Y = np.meshgrid(x,y)
axes3d.plot_surface(X,Y, spectro.spect[::-1],cmap="jet")



