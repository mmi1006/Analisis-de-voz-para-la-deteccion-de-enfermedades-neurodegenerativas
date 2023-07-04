#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import numpy as np
import pandas as pd
import csv
from scipy.io import wavfile
from scipy.stats import skew, kurtosis
from scipy.signal import butter, lfilter, freqz


def extraer_caracteristicas_Spicy(ruta, nombre_csv):
    
    archivos_audio = os.listdir(ruta)
    
    # Diccionario para meter als características
    caract_dict = {}
    
    for archivo in archivos_audio:
        
        # Obtener el ID y el tipo del archivo
        tipo = archivo[-5]
        ID = archivo[:5]
        
        ID_array = np.array([ID])
        tipo_array = np.array([tipo])

        # Cargar el archivo
        fs, audio = wavfile.read(os.path.join(ruta, archivo))
        
        # Extraer la duración 
        duracion = len(audio) / fs
        umbral_duracion = 1.0  
        
        # Umbral de duración en segundos, para asignar un valor nulo a lso audios vacios
        if duracion < umbral_duracion:
            caract_dict[archivo] = [None] * 9
            continue
        
        # Extraer la 1ª y 2ª derivada fundamental
        ff1 = np.gradient(audio)
        ff2 = np.gradient(ff1)
        
        # Extraer jitter
        jitter = np.mean(np.abs(np.diff(audio))) / np.mean(audio)
        
        # Extraer shimmer
        shimmer = np.mean(np.abs(np.diff(ff1))) / np.mean(np.abs(ff1))
        
        
        # Extraer APQ y PPQ
        apq = np.sum(np.square(np.abs(ff1))) / len(ff1)
        ppq = np.sum(np.square(np.abs(ff2))) / len(ff2)
        
        # Extraer la amplitud media del audio
        ama_mean = np.mean(np.abs(audio))
        
        # Extraer log-energy
        energy = np.sum(np.square(audio))
        log_energy = np.log(energy)
        
        # Almacenar las características en el diccionario
        caract_dict[archivo] = np.concatenate((ID_array, tipo_array, np.mean(ff1).reshape(-1,1)[0],
                                               np.mean(ff2).reshape(-1,1)[0], 
                                               jitter.reshape(-1,1)[0], 
                                               shimmer.reshape(-1,1)[0], 
                                               log_energy.reshape(-1,1)[0], 
                                               apq.reshape(-1,1)[0], 
                                               ppq.reshape(-1,1)[0], 
                                               ama_mean.reshape(-1,1)[0], 
                                               np.array([duracion]).reshape(-1,1)[0]), axis=0)
    
    
    carpeta_csv = "caracteríticas_Spicy"
    
    with open(carpeta_csv+"/"+nombre_csv, 'w', newline='') as archivo_csv:
        writer = csv.writer(archivo_csv, delimiter=';')
        
        # Determinar los valores de als columans
        writer.writerow(['archivo', 'ID','tipo', 'ff1', 'ff2', 'jitter', 'shimmer', 'log_energy', 'apq', 'ppq', 'Amplitud media', 'duracion'])
        
        # Introducir los valroes del directorio en el csv
        for archivo, caracteristicas in caract_dict.items():
            writer.writerow([archivo] + list(caracteristicas))

