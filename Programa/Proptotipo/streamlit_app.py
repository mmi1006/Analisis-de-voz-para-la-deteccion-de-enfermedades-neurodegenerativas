#!/usr/bin/env python
# coding: utf-8

# In[1]:


#!pip install streamlit


# In[8]:


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
import matplotlib.pyplot as plt


# In[23]:


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

        filas[0].extend(["Genero", "Edad"])

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
        datos_proces = pd.get_dummies(datos_procesados, columns=['Genero'],dummy_na=True)
        datos_pro = datos_proces.fillna(datos_proces.mean())
        print(datos_pro)
        
        cant_positivos = 0
        cant_negativos = 0
        
        for archivo in archivos:
            # Obtener el tipo del archivo
            tipo = archivo[-5]

            if tipo == '1':
                prediccion = modelo1.predict(datos_pro)[0]
            elif tipo == '2':
                prediccion = modelo2.predict(datos_pro)[1]
            elif tipo == '3':
                prediccion = modelo3.predict(datos_pro)[2]
            elif tipo == '4':
                prediccion = modelo4.predict(datos_pro)[3]
            elif tipo == '5':
                prediccion = modelo5.predict(datos_pro)[4]
            elif tipo == '6':
                prediccion = modelo6.predict(datos_pro)[5]
            else:
                return print("El audio no esta identificado con el tipo de audio concreto.")

            # Obtener los nombres de los grupos
            nombres_grupos = modelo1.classes_  # Tomamos el modelo1 como referencia

            # Obtener el nombre del grupo de la predicción
            nombre_grupo = nombres_grupos[prediccion]

            if (nombre_grupo == 1):
                resultado = "Positiva"
                cant_positivos += 1

            else:
                resultado = "Negativa"
                cant_negativos += 1

            clasificacion.append(f"El audio de tipo {tipo} obtiene una predicción de {resultado}.")
            
        # Crear el gráfico de tipo tarta
        labels = ['Positivo', 'Negativo']
        sizes = [cant_positivos, cant_negativos]
        explode = (0.1, 0)  # Explode para resaltar el primer sector
        colors = ['#66b3ff', '#ff9999']

        # Generar el gráfico
        plt.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
        plt.axis('equal')  # Para que el gráfico de tarta sea un círculo en lugar de una elipse

        # Mostrar el gráfico
        plt.show() 
        return clasificacion,  cant_positivos, cant_negativos   


# In[27]:


import streamlit as st
import numpy as np

def main():
    st.title("Clasificador de Audios")

    ruta_carpeta = ""

    ruta_carpeta = st.sidebar.text_input("Ruta de la carpeta:")
    ruta_carpeta = os.path.abspath(ruta_carpeta)  # Obtener la ruta absoluta    
    
    nom_paciente = st.text_input("Nombre:")  # Valor del ID del paciente
    apello1_paciente = st.text_input("1º Apellido:")  # Valor del ID del paciente
    apello2_paciente = st.text_input("2º Apellido:")  # Valor del ID del paciente
    num_historia = st.text_input("Número de historia:")
    edad_paciente = st.text_input("Edad:")  # Valor de la edad del paciente
    genero_paciente = st.text_input("Género: (Mujer/Hombre/Indefinido)") # Valor del género del paciente

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
    modelo1 = 'Modelos/modelo_entrenado_KNeighborsClassifier_tipo1_spicy.joblib'  
    modelo2 = 'Modelos/modelo_entrenado_KNeighborsClassifier_tipo2_spicy.joblib'  
    modelo3 = 'Modelos/modelo_entrenado_KNeighborsClassifier_tipo3_spicy.joblib'  
    modelo4 = 'Modelos/modelo_entrenado_KNeighborsClassifier_tipo4_spicy.joblib' 
    modelo5 = 'Modelos/modelo_entrenado_KNeighborsClassifier_tipo5_spicy.joblib' 
    modelo6 = 'Modelos/modelo_entrenado_KNeighborsClassifier_tipo6_spicy.joblib'
    

    if st.button("Realizar predicción"):
        
        st.write(f"Nombre: {nom_paciente}")
        st.write(f"1º Apellido: {apello1_paciente}")
        st.write(f"2º Apellido: {apello2_paciente}")
        st.write(f"Número de historia: {num_historia}")
        st.write(f"Edad: {edad_paciente}")
        st.write(f"Género: {genero_paciente}")
        
        st.write("RESULTADOS:")
        
        resultado, cant_positivos, cant_negativos = clasificador.predictor(ruta_csv, modelo1, modelo2, modelo3, modelo4, modelo5, modelo6)
        for clasif in resultado:
            st.markdown(f"- {clasif}", unsafe_allow_html=True)

        # Crear el gráfico de tipo tarta
        labels = ['Positivo', 'Negativo']
        sizes = [cant_positivos, cant_negativos]
        explode = (0.1, 0)  # Explode para resaltar el primer sector
        colors = ['#66b3ff', '#ff9999']

        # Generar el gráfico
        fig, ax  = plt.subplots()
        ax.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
        ax .axis('equal')  # Para que el gráfico de tarta sea un círculo en lugar de una elipse
        
        if cant_positivos>cant_negativos:
            resultado_final = "Positiva"
        else:
            resultado_final = "Negativa"

        st.write(f"CONCLUSIÓN: Positivo en la enfermedad ed Parkinson en {cant_positivos}/{cant_positivos+cant_negativos} audios analizados.")
        # Mostrar el gráfico utilizando st.pyplot()
        st.pyplot(fig)

    os.remove("caracteríticas_Spicy/caracteristicas.csv")

if __name__ == "__main__":
    main()
    


# In[ ]:




