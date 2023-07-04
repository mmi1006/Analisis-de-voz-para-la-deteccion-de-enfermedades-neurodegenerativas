#!/usr/bin/env python
# coding: utf-8

# In[1]:


import csv
import os
import torch
import torchaudio
import numpy as np
import pandas as pd

def extraer_caracteristicas_PyThorch(ruta, nombre_csv):

    # Ruta de la carpeta que contiene los archivos de audio
    archivos_audio = os.listdir(ruta)

    # Crear un diccionario para almacenar las características de los archivos de audio
    caract_dict = {}

    # Recorrer cada archivo en la carpeta
    for archivo in archivos_audio:

        # Obtener el ID y el tipo del archivo
        tipo = archivo[-5]
        ID = archivo[:5]

        # Convertimos en arrays de dimensiones 1
        ID_array = np.array([ID])
        tipo_array = np.array([tipo])

        # Cargar el archivo 
        file_path = os.path.join(ruta, archivo)
        waveform, sample_rate = torchaudio.load(file_path)
        
        try:
            # Transformaciones a forma de onda
            spectrogram = torchaudio.transforms.Spectrogram()(waveform)
            amplitude_to_db = torchaudio.transforms.AmplitudeToDB()(spectrogram)
            loudness = torchaudio.transforms.Loudness(sample_rate=sample_rate)(waveform)
            mfcc = torchaudio.transforms.MFCC(sample_rate=sample_rate)(waveform)
            psd = torchaudio.transforms.Spectrogram(n_fft=2048)(waveform)
            spectral_centroid = torchaudio.transforms.SpectralCentroid(sample_rate=sample_rate)(waveform)

            # Calcular la media 
            amplitude_to_db_mean = torch.mean(amplitude_to_db)
            loudness_mean = torch.mean(loudness, axis=0)
            mfcc_mean = torch.mean(mfcc)
            psd_mean = torch.mean(psd)
            spectral_centroid_mean = torch.mean(spectral_centroid)
            spectrogram_mean = torch.mean(spectrogram)

            # Agregar las características al diccionario 
            caract_dict[archivo] = np.concatenate((ID_array, tipo_array, amplitude_to_db_mean.reshape(-1,1)[0], 
                                                    loudness_mean.reshape(-1,1)[0], 
                                                    mfcc_mean.reshape(-1,1)[0], 
                                                    psd_mean.reshape(-1,1)[0], 
                                                    spectral_centroid_mean.reshape(-1,1)[0], 
                                                    spectrogram_mean.reshape(-1,1)[0]), axis=0)
            
        except RuntimeError as e:
            print(f"Error al procesar el archivo {archivo}: {str(e)}")

    # Escribir el contenido del diccionario en un archivo CSV
    with open(nombre_csv, 'w', newline='') as archivo_csv:
        writer = csv.writer(archivo_csv, delimiter=';')

        # Escribir los nombres de las características en la primera fila del CSV
        writer.writerow(['archivo', 'ID', 'tipo', 'amplitude_to_db', 'loudness', 'mfcc', 'psd', 'spectral_centroid', 'spectrogram'])

        # Escribir las características en cada línea del CSV
        for archivo, caracteristicas in caract_dict.items():
            writer.writerow([archivo] + list(caracteristicas))

