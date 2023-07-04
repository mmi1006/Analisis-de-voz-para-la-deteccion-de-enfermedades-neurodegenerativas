#!/usr/bin/env python
# coding: utf-8

# In[6]:


import os
from pydub import AudioSegment


def NormalizarVolumen(carpeta):
    for archivo in os.listdir(carpeta):
        # Comprobar el formato de audio
        if archivo.endswith(".wav"):

            try:
                # Cargar audio
                audio = AudioSegment.from_file(os.path.join(carpeta, archivo))

                # Normalizar el volumen 
                audio_normalizado = audio.normalize()

            # Excepción con mensaje de salida del archivo que ha no se ha podido normalizar
            except Exception as e:
                print(f"No se pudo cargar el archivo {archivo}: {e}")

            #Exportar el sudio normalizado en la carpeta indicada
            audio_normalizado.export(os.path.join(carpeta, archivo), format="wav")

