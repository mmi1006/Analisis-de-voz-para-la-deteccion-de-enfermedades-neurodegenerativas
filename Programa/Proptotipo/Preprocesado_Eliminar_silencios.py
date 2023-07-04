#!/usr/bin/env python
# coding: utf-8

# In[7]:


import os
from pydub import AudioSegment
import numpy as np


def EliminarSilencion(carpeta, umbral):
    for i in os.listdir(carpeta):

        try:
            # Cargar el archivo en cada caso
            audio = AudioSegment.from_file(os.path.join(carpeta, i))
            print(i)

            # Obtener la matriz del audio
            matriz_audio = np.array(audio.get_array_of_samples())

            # Índice del primer valor que no sea un silencio
            inicio_sin_s = np.argmax(matriz_audio > 0.00)

            # Índice del último valor que no sea un silencio
            final_sin_s = len(matriz_audio) - np.argmax(matriz_audio[::-1] > 0.00)

            # Calcular la duración del audio original y del audio sin silencios
            duracion = len(audio)
            duracion_sin_s = final_sin_s - inicio_sin_s

            # Comprobar si la diferencia de duración es menor que el umbral marcado
            if abs(duracion - duracion_sin_s) < umbral:
                # Si la diferencia de duración es menor que el umbral, guardar el audio original
                audio.export(os.path.join(carpeta, "original_" + i), format="wav")
            else:
                # Si la diferencia de duración es mayor que el umbral, recortar el audio y guardarlo en la misma carpeta en formato .wav
                audio_sin_s = audio[inicio_sin_s:final_sin_s]
                audio_sin_s.export(os.path.join(carpeta, i), format="wav")

        except Exception as e:
            print(f"No se pudo cargar el archivo {i}: {e}")

