#!/usr/bin/env python
# coding: utf-8

# In[7]:


import pandas as pd
def unir_csv(datos_general, *datos_tipos):
    # Leer el archivo con los datos clinicos los generales
    general = pd.read_csv(datos_general, sep=';', encoding='ISO-8859-1')

    for tipo in datos_tipos:
        # Leer el que contiene las caracteristicas
        tipos = pd.read_csv(tipo, sep=';')

        tipo_nombre = tipo.split('.')[0]

        # Unir mediante join con laa columna ID
        resultado = general.join(tipos.set_index('ID'), on='ID')

        # Guardar, nuevo csv
        resultado.to_csv(f'unificacion_{tipo_nombre}.csv', index=False, sep=';')

    print("Unión de CSV realizada.")

