#!/usr/bin/env python
# coding: utf-8

# In[1]:


#!pip install streamlit


# In[1]:


#Empezamos importando todos los archivos que contendrán las funciones a utilizar
import Preprocesado_Normalizar_volumen
import Preprocesado_Eliminar_silencios
import Union_dataset
import Extraer_caract_Spicy
import Extraer_caract_Pytorch
import csv
import st
import os
import pandas as pd
import joblib


# In[2]:


class ClasificadorAudios:
    
    def __init__(self, carpeta):
        #lista de audios y lista de modelos
        self.carpeta = carpeta
        self.modelos = []
    
    def preprocesador(self, umbral):
        audio_preprocesado = Preprocesado_Normalizar_volumen.NormalizarVolumen(self.carpeta)
        audio_preprocesado = Preprocesado_Eliminar_silencios.EliminarSilencion(self.carpeta, umbral)
        return audio_preprocesado

    def Extractor_carac(self, nombre_csv):
        caracteristicas = Extraer_caract_Spicy.extraer_caracteristicas_Spicy(self.carpeta, nombre_csv)
        return caracteristicas


    def agregar_columnas(self, caracteristicas, genero, edad):
        # Leer el archivo CSV existente
        with open(caracteristicas, "r", newline='') as archivo:
            filas = list(csv.reader(archivo, delimiter=';'))

        filas[0].extend(["genero", "edad"])

        # Agregar las tres columnas a las filas existentes
        for fila in filas[1:]:
            fila.extend([genero, edad])

        # Escribir la lista de filas actualizada en el archivo CSV
        with open(caracteristicas, "w", newline='') as archivo:
            escribir = csv.writer(archivo, delimiter=';')
            escribir.writerows(filas)
    
    def predictor(self, datos, *modelo): 
         # El diccionario donde almacenaremos la clasificacion de cada audio
        clasificacion = []
        df = pd.read_csv(datos, sep=";")
        archivos = df['archivo']
        
        #Cargamos los posibles modelos a utilizar
        modelo1 = joblib.load(modelo[0])
        modelo2 = joblib.load(modelo[1])
        modelo3 = joblib.load(modelo[2])
        modelo4 = joblib.load(modelo[3])
        modelo5 = joblib.load(modelo[4])
        modelo6 = joblib.load(modelo[5])
        
        #Preprocesamos los datos para que el modelo entrenado los acepte
        nuevos_datos = pd.read_csv(datos, sep=';')

        datos_procesados = nuevos_datos.drop(['ID', 'archivo'], axis=1)
        datos_proces = pd.get_dummies(datos_procesados, columns=['genero'],dummy_na=True)
        datos_pro = datos_proces.fillna(datos_proces.mean())
        print(datos_pro)
        for archivo in archivos:
            # Obtener el tipo del archivo
            tipo = archivo[-5]

            if tipo == '1':
                prediccion = modelo1.predict(datos_pro)
            elif tipo == '2':
                prediccion = modelo2.predict(datos_pro)
            elif tipo == '3':
                prediccion = modelo3.predict(datos_pro)
            elif tipo == '4':
                prediccion = modelo4.predict(datos_pro)
            elif tipo == '5':
                prediccion = modelo5.predict(datos_pro)
            elif tipo == '6':
                prediccion = modelo6.predict(datos_pro)
            else:
                return print("El audio no esta identificado con el tipo de audio concreto.")
        
        # Obtener los nombres de los grupos
        nombres_grupos = modelo1.classes_  # Tomamos el modelo1 como referencia

        # Obtener el nombre del grupo de la predicción
        nombre_grupo = nombres_grupos[prediccion]

        clasificacion.append(f"El audio de tipo {tipo} pertenece al grupo: {nombre_grupo}")
            
        return clasificacion
        print(clasificacion)
        


# In[3]:


import streamlit as st
import Suma_prueba

def main():
    st.title("Clasificador de Audios")

    ruta_carpeta = st.text_input("Ingrese la ruta de la carpeta que contiene los audios a analizar:")
    id_paciente = st.text_input("Ingrese el nº ID del paciente:")
    edad_paciente = st.text_input("Ingrese la edad del paciente:")
    genero_paciente = st.text_input("Ingrese el género del paciente:")

    clasificador = ClasificadorAudios(ruta_carpeta)

    # Preprocesamiento de los audios
    umbral = 0.5
    audio_preprocesado = clasificador.preprocesador(umbral)

    # Extracción de características y guardar en un archivo CSV
    nombre_csv = 'caracteristicas.csv'
    caracteristicas = clasificador.Extractor_carac(nombre_csv)

    # Guardar datos clínicos en un archivo CSV
    ruta_csv = "caracteríticas_Spicy/caracteristicas.csv"
    clasificador.agregar_columnas(ruta_csv, genero_paciente, edad_paciente)
    
     # Cargar los modelos y realizar la predicción
    modelo1 = 'Modelos/modelo_entrenado_DecisionTreeClassifier_tipo1_spicy.plk'  
    modelo2 = 'Modelos/modelo_entrenado_DecisionTreeClassifier_tipo2_spicy.plk'  
    modelo3 = 'Modelos/modelo_entrenado_DecisionTreeClassifier_tipo3_spicy.plk'  
    modelo4 = 'Modelos/modelo_entrenado_DecisionTreeClassifier_tipo4_spicy.plk' 
    modelo5 = 'Modelos/modelo_entrenado_DecisionTreeClassifier_tipo5_spicy.plk' 
    modelo6 = 'Modelos/modelo_entrenado_DecisionTreeClassifier_tipo6_spicy.plk'

    if st.button("Realizar predicción"):
        resultado = clasificador.predictor(ruta_csv, modelo1, modelo2, modelo3, modelo4, modelo5, modelo6)
        st.write(resultado)

    os.remove("caracteríticas_Spicy/caracteristicas.csv")

if __name__ == "__main__":
    main()


# In[ ]:




