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
import copy

from unidecode import unidecode
from sklearn import preprocessing
from sklearn import svm
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestClassifier 
from sklearn.preprocessing import LabelEncoder 
from sklearn.metrics import confusion_matrix 
from sklearn.metrics import accuracy_score 
from sklearn.metrics import classification_report 
from sklearn.feature_extraction import DictVectorizer
from numpy import nan, NAN

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
    column_names = getListColumns
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
    column_names = getListColumns(data)
    columns_to_drop = [column_names[i] for i in num_columns_list]
    data.drop(columns_to_drop, inplace=True, axis=1)

"""
    Reemplaza los NaN por el valor de la media de la columna (numérico)
"""


def replace_with_mean(data):
    "Lista de las columnas"
    columns = getListColumns(data)
    
    "Lista de los tipos de valores de las columnas"
    columns_type = getColumnsTypes(data)

    "Número de columnas"
    long_col = len(columns)

    valores_con_media = []

    c = 0
    while c < long_col:
        if(columns_type[c] != np.dtype('O')):
            valores_con_media.append(columns[c])
        c = c + 1
  
    for value in valores_con_media:
        data[value] = data[value].fillna(data[value].mean())

"""
    Reemplaza los NaN por el valor de la media (numérico) o moda (texto) de la columna 
"""


def replace_with_mean_and_mode(data):
    
    tiposDatos = data.columns.to_series().groupby(data.dtypes).groups
    columnas = data.columns  # lista de todas las columnas

    # Columnas con valores de texto
    columns_text = tiposDatos[np.dtype('object')]
    
    # Columnas con valores numéricos
    columns_num = list(set(columnas) - set(columns_text))
    
    for c in columns_text:
        data[c] = data[c].fillna(data[c].mode()[0])
        
    for c in columns_num:
        data[c] = data[c].fillna(data[c].mean())

"""
    Borra las filas duplicadas
"""


def drop_duplicates(data):
    data = data.drop_duplicates()

"""
    Reemplaza caracteres que pueden generar inconvenientes en el proceso dedupe
"""


def utfProcessData(column_name):
    try : 
        column_name = column_name.decode('utf8')
    except AttributeError:
        pass
    
    column_name = unidecode(column_name)
    column_name = re.sub('  +', ' ', column_name)
    column_name = re.sub('\n', ' ', column_name)
    column_name = column_name.strip().strip('"').strip("'").lower().strip()
    
    if not column_name:
        column_name = None
    return column_name
        

def main():
    data = formatData("resources/movie_metadata.csv")
    data = data.dropna(subset=["title_year"])
    data.duration = data.duration.fillna(data.duration.mean())
    data.duration = pd.Series(data["duration"], dtype="int32")
    data.content_rating = data.content_rating.fillna("Not Known")
    data = data.rename(columns={"title_year":"release_date", "movie_facebook_likes":"facebook_likes"})
    data.movie_title = data["movie_title"].str.upper()
    data.movie_title = data["movie_title"].str.strip()
    data.to_csv("resources/resultClean.csv", encoding="utf-8")

"""
    Le asigna un índice a los datos
"""


def set_index(data):
    "Lista de las columnas"
    columns = getListColumns(data)

    valores = []
  
    for value in columns:
        valores.append(len(data[value].unique().tolist()))

    new_index = columns[valores.index(max(valores))]

    data = data.dropna(subset=[new_index])

    data_size = len(data)

    valores_unicos = []
    filas_unicas = []
    filas_repetidas = []
    posiciones = []
    c = 0

    while c < data_size:
        fila = data.iloc[c]
        valor_fila_index = fila[new_index]
        if(valor_fila_index in valores_unicos):
            filas_repetidas.append(fila)
            posiciones.append(c)
            
        else:
            filas_unicas.append(fila)
            valores_unicos.append(valor_fila_index)

        c = c + 1
    
    data = data.drop(posiciones)

    setDataIndex(data, new_index)

    return [data, new_index]


def axes_compare(data, index_value, depen_value):
    df = pd.DataFrame(data)

    size = len(data)
    c = 0

    "df['index_order'] = [0:size]"
    lista = list(range(0, size))
    df['index_order'] = lista

    x = df['index_order']
    y = df[depen_value]

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.5, random_state=42)

    print("x_train")
    print(x_train)
    print("y_train")
    print(y_train)

    x_train = x_train.values.reshape([x_train.values.shape[0], 1])
    x_test = x_test.values.reshape([x_test.values.shape[0], 1])

    regr = linear_model.LinearRegression()
    regr.fit(x_train, y_train)
    y_pred = regr.predict(x_test)

    plt.scatter(x_train, y_train, color='red')
    plt.scatter(x_test, y_pred, color='blue')

    plt.show()


def entrenamiento(data, independ_values, depen_value):
    
    dict_values=dict()
    
    tiposDatos = data.columns.to_series().groupby(data.dtypes).groups
    
    # Columnas con valores de texto
    columns_text = tiposDatos[np.dtype('object')]
    
    
    """df = pd.DataFrame(data)
    dummies = pd.get_dummies(df.categorical)
    print(dummies)"""
    
    values_train=[]
    
        
    #for c in indepen_values:
        #column_type=data.dtypes[c]
    """if data.dtypes[c]!=np.dtype('object') and data.dtypes[depen_value]!=np.dtype('object'):
            
        values_train.append(getColumnValues(data, c))
    
        depend_value=getColumnValues(data, depen_value)
            
        X=np.array(values_train).T
        Y=np.array(depend_value)
            
        reg=LinearRegression()
        reg=reg.fit(X,Y)
        Y_pred=reg.predict(X)"""
    
    """"X=data[independ_values]
    X = pd.get_dummies(data=X, drop_first=True)

    Y=data[depen_value]
    
    if(data.dtypes[depen_value]==np.dtype('object')):
        Y=pd.get_dummies(data[depen_value], drop_first=True)"""
        
    X = data.drop(columns=depen_value).iloc[:, :].values
    y = data[depen_value].copy()
        
    #Encoding the predicting variable 
    labelencoder_y = LabelEncoder() 
    y = labelencoder_y.fit_transform(y) 
  
    #Splitting the data into test and train dataset 
    X_train, X_test, y_train, y_test = train_test_split( 
              X, y, test_size = 0.3, random_state = 0) 
  
    #Using the random forest classifier for the prediction 
    classifier=RandomForestClassifier() 
    classifier=classifier.fit(X_train,y_train) 
    predicted=classifier.predict(X_test) 
   
  
    print ('Confusion Matrix :') 
    print(confusion_matrix(y_test, predicted)) 
    print ('Accuracy Score :',accuracy_score(y_test, predicted)) 
    print ('Report : ') 
    print (classification_report(y_test, predicted)) 

"""Predice los valores"""


def predicted_values(data, midSize):
    
    use_to_predicted=data.ix[:midSize, :]  
    other_values=data.ix[midSize:, :]

    columns_list=getListColumns(data)
    columns_list.pop() 
    columns_dict=dict(zip(columns_list, range(len(columns_list))))
    
    print(data.shape)
    print(columns_list)
    
    vectorizer = DictVectorizer()
    
    """le = preprocessing.LabelEncoder()
    
    "le = LabelEncoder()
    le.fit(['a', 'e', 'b', 'z'])
    le.classes_
    array(['a', 'b', 'e', 'z'], dtype='U1')
    samples = [dict(enumerate(sample)) for sample in data['color']]"
        
    indep=[]
    dep=[]
        
    data_x=use_to_predicted.drop(columns='color')
    data_y = use_to_predicted['color'].copy()
    
    dict_to_replace=dict()
    
    for column in columns_list:
        if column!='color':
            datos=use_to_predicted[column].tolist()
            if use_to_predicted.dtypes[column]==np.dtype('object'):
                set_col_1 = list(set(datos))
                le.fit(datos)
                dict_datos=dict(zip(set_col_1, le.transform(set_col_1)))
                dict_to_replace=dict_datos.copy()
                indep.append(le.transform(datos))
            else:
                indep.append(datos)
        else:
            datos=use_to_predicted['color'].tolist()
            set_col_1 = list(set(datos))
            le.fit(datos)
            dict_datos=dict(zip(set_col_1, le.transform(set_col_1)))
            dep=le.transform(datos)
    
    X=np.array(indep).T
    Y=np.array(dep)    
    reg=LinearRegression()
    reg=reg.fit(X,Y)
    Y_pred=reg.predict(X)"""
    
    
    df = pd.DataFrame(other_values)
    p=0
    c=0
    for row in df.iterrows():
        """if p==0:
            change_cell_value(df, 0, 'color', nan)
            change_cell_value(df, 0, 'director_name', nan)
            change_cell_value(df, 0, 'duration', nan)"""

            #print(df.iloc[0,  df.columns.get_loc('color')])
        null_row_values, not_null_row_values, null_column, not_null_column=div_null_row_values(c, columns_list, df)

        if len(null_column)>0:
            for column in null_column:
        
                rest_null_values=null_column.copy()
                rest_null_values.remove(column)
                    
                conf_columns_list=columns_list.copy()
                for cc in rest_null_values:
                    conf_columns_list.remove(cc)
                    
                conf_use_to_predicted=use_to_predicted.copy()
                conf_use_to_predicted.drop(rest_null_values, inplace=True, axis=1)
                    
                le = preprocessing.LabelEncoder()
                indep, dep, dict_to_replace= predicted_tool(conf_columns_list, conf_use_to_predicted, column, le)
                reg=get_prediction_tool(indep, dep)            
                        
                nueva_lista=transform_data(not_null_row_values, le)[1]

                predicted=get_prediciton(reg, nueva_lista)[0]
                new_cell_value=predicted
                if use_to_predicted.dtypes[column]==np.dtype('object'):
                    new_cell_value=get_new_cell_value(dict_to_replace, predicted)
                        
                change_cell_value(df, c, column, new_cell_value)
                    #print(df.iloc[0,  df.columns.get_loc(column)])
        print(c)
        c+=1
        
    #print(other_values['color'].tolist())
    
    
    """for column in columns_list:
        data_x = use_to_predicted.drop(columns=column)
        data_y = use_to_predicted[column].copy()
        
        entrenamiento(getListColumns(data_x), column, use_to_predicted)
       

    columns_num=len(columns_list)
    con=0
    
    while con<columns_num:
        independ_values=[]
        if(con+1<columns_num):
            independ_values=columns_list[:con]+columns_list[con+1:]
        
        else:
            independ_values=columns_list[:con]
            
        depend_value=columns_list[con]
        
        entrenamiento(use_to_predicted, independ_values, depend_value)
        
        con+=1"""
    
    
    """"b=False
    df = pd.DataFrame(use_to_predicted)
    for row in df.iterrows():
        if b==False:
            print(row)
            b=True"""
            
            

#def fill_data(null_row_values, not_null_row_values):
    
"""

"""
def get_new_cell_value(dict_to_replace, predicted_value):
    valores=list(dict_to_replace.keys())
        
    diferencia=[]
    for x in range(len(valores)):
        dict_value=dict_to_replace.get(valores[x])
        diferencia.append(abs(predicted_value-dict_value))
    
    dif_min=min(diferencia)
    pos=diferencia.index(dif_min)
    
    return valores[pos]

"""

"""

def div_null_row_values(num_row, columns_list, df):
    null_row_values=[]
    not_null_row_values=[]

    null_column=[]
    not_null_column=[]

    for column in columns_list:
        cell_value=df.iloc[num_row,  df.columns.get_loc(column)]
        if pd. notna(cell_value):
            #print(df.iloc[num_row,  df.columns.get_loc('color')])
            not_null_row_values.append(cell_value)
            not_null_column.append(column)
        else:
            null_row_values.append(cell_value)
            null_column.append(column)
            
    return [null_row_values, not_null_row_values, null_column, not_null_column]

"""

"""
def transform_data(datos, le):
    set_col_1 = list(set(datos))
    le.fit(datos)
    dict_datos=dict(zip(set_col_1, le.transform(set_col_1)))
    dict_to_replace=dict_datos.copy()
        
    return [dict_to_replace, le.transform(datos)]
    
    
"""

"""
def predicted_tool(columns_list, use_to_predicted, column_pred, le):
    indep=[]
    dep=[]
    for column in columns_list:
        if column!=column_pred:
            datos=use_to_predicted[column].tolist()
            if use_to_predicted.dtypes[column]==np.dtype('object'):
                transform=transform_data(datos,le)
                indep.append(transform[1])
            else:
                indep.append(datos)
        else:
            datos=use_to_predicted[column_pred].tolist()    
            dict_to_replace=dict()
            if use_to_predicted.dtypes[column_pred]==np.dtype('object'):
                transform=transform_data(datos,le)
                dep=transform[1]
                dict_to_replace=transform[0]
            
            else:
                dep=datos

    return [indep, dep, dict_to_replace]


"""

"""
def get_prediciton(reg, indep):
   return reg.predict([indep])


"""

"""

def get_prediction_tool(indep, dep):
    X=np.array(indep).T
    Y=np.array(dep)  
    X_scaled = preprocessing.scale(X) 
    reg=LinearRegression()
    reg=reg.fit(X_scaled,Y)
            
    return reg


"""
    Método para cambiar el valor de una celda
    df: DataFrame de los datos
    row_num: Número de la fila
    column_name: Nombre de la columna
    new_value: Nuevo valor de la celda
"""
def change_cell_value(df, row_num, column_name, new_value):
    df.iloc[row_num, df.columns.get_loc(column_name)]=new_value


def prueba(recurso):
    data = formatData(recurso)
    """data.to_csv("resources/resultClean.csv", encoding="utf-8")"""
    result = set_index(data)
    
    dataInicial = result[0]
    data = copy.copy(dataInicial)
    
    index_value = result[1]
    
    "Se reemplaza con la media y la moda hasta la mitad de los valores"

    replace_with_mean_and_mode(dataInicial)
    
    midSize = len(dataInicial) // 2
    parteData = dataInicial.ix[:midSize, :]
    
    data.ix[:midSize, :] = parteData
    
    predicted_values(data, midSize)
        
    """axes_compare(data, index_value, 'duration')"""
    data.to_csv("resources/resultClean.csv", encoding="utf-8")
    
    
prueba("resources/movie_metadata.csv")

