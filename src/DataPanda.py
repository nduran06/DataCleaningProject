# encoding: utf-8

import pandas as pd
import NaN as nan


"""
    Retorna la variable que continene a los datos según el path dado
"""
def dataRead(data_path):
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
    Elimina las columnas dadas de los datos 
"""
def dropColumns(data, num_columns_list):
    column_names=getListColumns(data)
    columns_to_drop = [column_names[i] for i in num_columns_list]
    data.drop(columns_to_drop, inplace=True, axis=1)

def main():
    data=dataRead("resources/movie_metadata.csv")
    nans=nan.getNans(data)
    print(nans)
    
main()





