# encoding: utf-8

import pandas as pd
import NaN as nan
import numpy
import dedupe
import os
import csv
import re
import json
import _pickle as pickle

from unidecode import unidecode

"""
    Retorna la variable que continene a los datos según el path dado
"""
def formatData(data_path):
    return pd.read_csv (data_path) 

    
"""
    Se obtiene el nombre de las columnas
"""
def getListColumns(data):
    return data.columns


"""
    Tipo de datos en las columnas
"""
def getColumnsTypes(data):
    return data.dtypes


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
    
def readData(filename):
    
    data_d = {}
    with open(filename) as f:
        reader = csv.DictReader(f)
        data=formatData(filename)
        values=getIndexValues(data)
        con=0
        for row in reader:
            clean_row = [(k, utfProcessData(v)) for (k, v) in row.items()]
            row_id = values[con]
            data_d[row_id] = dict(clean_row)
            con+=1
            
    return data_d 


"""
    Method to save a text in a given file
    texto: Text to save
    nameFile: Name/path of file
"""
def generateFile(texto, nameFile):
    with open(nameFile, 'w') as f:
        f.write(str(texto))
        

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
    
    
    
    
main()





