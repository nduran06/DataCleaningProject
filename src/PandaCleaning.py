#!/usr/bin/env python
# encoding: utf-8

import pandas as pd
import NaN as nan
import numpy as np
import dedupe
import os
import csv
import re
import json
import _pickle as pickle
import matplotlib.pyplot as plt

from unidecode import unidecode
from sklearn import preprocessing
from sklearn import svm
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error


"""
    Retorna la variable que continene a los datos según el path dado
"""
def formatData(data_path):
    return pd.read_csv (data_path) 

    
"""
    Se obtiene el nombre de las columnas
"""
def getListColumns(data):
    return data.columns.tolist()


"""
    Tipo de datos en las columnas
"""
def getColumnsTypes(data):
    return data.dtypes.tolist()


"""
    Muestra qué dato en las columnas es único (true/false)
"""
def printColumnsUniqueValues(data):
    column_names=getListColumns
    for i in column_names:
        print('{} is unique: {}'.format(i, data[i].is_unique))
    

"""
    Establece a una columna como índice
"""
def setDataIndex(data, column_index_name):
    data.set_index(column_index_name, inplace=True)


"""
    Retorna los valores de la columna índice
    throwsException: Si no existe la columna índice
"""
def getIndexValues(data):
    return data.index.values

    
"""
    Verifica si existe cierto índice
"""
def verifyIndex(data, column_index_name):
    return column_index_name in data.index.values


"""
    Retorna los valores de la columna dada
"""
def getColumnValues(data, column_name):
    return data[column_name]
    

"""
    Retorna si los valores de la columna están o no vacíos
"""
def getColumnNullValues(data, column_name):
    return data[column_name].isnull()

    
"""
    Elimina las columnas dadas de los datos 
"""
def dropColumns(data, num_columns_list):
    column_names=getListColumns(data)
    columns_to_drop = [column_names[i] for i in num_columns_list]
    data.drop(columns_to_drop, inplace=True, axis=1)


"""
    Reemplaza los NaN por el valor de la media de la columna (numérico)
"""
def replace_with_mean(data):
    "Lista de las columnas"
    columns=getListColumns(data)
    
    "Lista de los tipos de valores de las columnas"
    columns_type=getColumnsTypes(data)

    "Número de columnas"
    long_col=len(columns)

    valores_con_media=[]

    c=0
    while c<long_col:
        if(columns_type[c]!=np.dtype('O')):
            valores_con_media.append(columns[c])
        c=c+1
  
    for value in valores_con_media:
        data[value]=data[value].fillna(data[value].mean())

"""
    Borra las filas duplicadas
"""
def drop_duplicates(data):
    data=data.drop_duplicates()
    

"""
    Reemplaza caracteres que pueden generar inconvenientes en el proceso dedupe
"""
def utfProcessData(column_name):
    try : 
        column_name = column_name.decode('utf8')
    except AttributeError:
        pass
    
    column_name=unidecode(column_name)
    column_name=re.sub('  +', ' ', column_name)
    column_name=re.sub('\n', ' ', column_name)
    column_name=column_name.strip().strip('"').strip("'").lower().strip()
    
    if not column_name:
        column_name = None
    return column_name
        

def main():
    data=formatData("resources/movie_metadata.csv")
    data = data.dropna(subset=["title_year"])
    data.duration = data.duration.fillna(data.duration.mean())
    data.duration = pd.Series(data["duration"], dtype="int32")
    data.content_rating = data.content_rating.fillna("Not Known")
    data = data.rename(columns = {"title_year":"release_date", "movie_facebook_likes":"facebook_likes"})
    data.movie_title = data["movie_title"].str.upper()
    data.movie_title = data["movie_title"].str.strip()
    data.to_csv("resources/resultClean.csv", encoding="utf-8")
    


"""
    Reemplaza los NaN por el valor de la media de la columna (numérico)
"""
def set_index(data):
    "Lista de las columnas"
    columns=getListColumns(data)

    valores=[]
  
    for value in columns:
        valores.append(len(data[value].unique().tolist()))

    new_index=columns[valores.index(max(valores))]

    data = data.dropna(subset=[new_index])

    data_size=len(data)

    valores_unicos=[]
    filas_unicas=[]
    filas_repetidas=[]
    posiciones=[]
    c=0

    while c<data_size:
        fila=data.iloc[c]
        valor_fila_index=fila[new_index]
        if(valor_fila_index in valores_unicos):
            filas_repetidas.append(fila)
            posiciones.append(c)
            
        else:
            filas_unicas.append(fila)
            valores_unicos.append(valor_fila_index)

        c=c+1
    
    data=data.drop(posiciones)

    setDataIndex(data, new_index)

    print(len(data))

    return new_index


def axes_compare(data, index_value, depen_value):
    df=pd.DataFrame(data)

    size=len(data)
    c=0

    "df['index_order'] = [0:size]"
    lista=list(range(0,size))
    df['index_order'] =lista


    x=df['index_order']
    y=df[depen_value]

    x_train, x_test, y_train, y_test=train_test_split(x,y, test_size=0.5, random_state=42)

    print("x_train")
    print(x_train)
    print("y_train")
    print(y_train)

    x_train=x_train.values.reshape([x_train.values.shape[0],1])
    x_test=x_test.values.reshape([x_test.values.shape[0],1])


    regr=linear_model.LinearRegression()
    regr.fit(x_train, y_train)
    y_pred=regr.predict(x_test)

    plt.scatter(x_train, y_train, color='red')
    plt.scatter(x_test, y_pred, color='blue')

    plt.show()

def prueba():
    data=formatData("resources/movie_metadata.csv")
    """data.to_csv("resources/resultClean.csv", encoding="utf-8")"""
    index_value=set_index(data)
    replace_with_mean(data)

    axes_compare(data, index_value, 'duration')
    
prueba()





