# encoding: utf-8

import pandas as pd

"""
    Retorna el número de NaNs de cada columna
"""
def getNans(data):
    return data.isna().sum()

"""
    Rellena los NaN de la columna dada con el valor dado
    (espacios --> '')
    (números --> n)
    ...
"""
def fillNaN(data, column_name, value):
    data[column_name]=data[column_name].fillna(value)
    
    
"""
    Rellena los NaN con la media de la columna
    ...
"""
def fillNaNWithMean(data, column_name):
    column_mean=data[column_name].mean()
    fillNaN(data, column_name, column_mean)
    
    
"""
    Propaga valores no nulos ( Completará los siguiente n valores NaN con 
    el valor respectivo anterior que no sea NaN.
"""
def fillNaNSpreadPreviousValue(data, limit_num):
    data.fillna(method='pad', limit=limit_num)


"""
    Rellana los 2n prmeros valores con el primer valor disponible
    (ie. Si limit_num=1, rellena los dos primeros valores NaN con
    el primer valor disponible)
"""
def fillNaNWithFirstAvailableValue(data, limit_num):
    data.fillna(method='bfill', limit=limit_num)


"""
    Elimina las filas con valores NaN
"""
def dropRowsWithNanValues(data):
    data.dropna()

"""
    Elimina las columnas con valores NaN
"""
def dropColumnsWithNaNValues(data):
    data.dropna(axis = 1)

"""
    Elimina las columnas que no tengan al menos X cantidad de valores no NaN
"""
def dropColumnsWithAtLeastXNoNNaNValues(data, value):
    percentage=value/100
    data.dropna(tresh=int(data.shape[0]*percentage), axis=1)




