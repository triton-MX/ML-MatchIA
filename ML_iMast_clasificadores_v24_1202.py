# -*- coding: utf-8 -*-
"""
Created on Mon Dec  2 06:52:36 2024

@author: iapcl

Referencia:https://panamahitek.com/analisis-de-datos-mamograficos-con-machine-learning/
"""
# Librerias a usar
import pandas as pd
import numpy as np
import pydicom as py
from PIL import Image
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.utils import all_estimators
from sklearn.preprocessing import MinMaxScaler
from sklearn.exceptions import NotFittedError
from joblib import dump

path_directory = r"C:\Users\padie\Desktop\Imagenes extraidas - General - v3 -Incluye DICOM"
path_excel = r"C:\Users\padie\Desktop\Iñigo\Listado prueba 1.xlsx"

"""----------------------------
Funciones a usar
"""

def determineResize(sizes):
    """
    Función para encontrar el tamaño mínimo entre un conjunto de dimensiones (alto, ancho).
    
    Parameters
    ----------
    sizes : list of tuple
        Lista de tuplas con las dimensiones (alto, ancho) de cada imagen.

    Returns
    -------
    tuple
        El tamaño mínimo (alto, ancho). 
    """
    min_alto = float('inf')
    min_ancho = float('inf')
    
    for alto, ancho in sizes:
        if alto < min_alto and ancho < min_ancho:
            min_alto = alto
            min_ancho = ancho
    
    min_alto = int(min_alto/5)
    min_ancho = int(min_ancho/5)
    
    return min_alto, min_ancho
    

def loadDataset(path_excel, path_directory):
    """
    Función para cargar las imágenes DICOM y sus dimensiones.
    
    Parameters
    ----------
    path_excel : str
        Ruta al archivo Excel con la lista de imágenes.
    path_directory : str
        Ruta al directorio donde se encuentran las imágenes DICOM.
    
    Returns
    -------
    images : list
        Lista de matrices de píxeles de las imágenes DICOM.
    sizes : list
        Lista de tuplas con las dimensiones (alto, ancho) de cada imagen.
    """
    print("Inicia carga de Dataset")
    # Cargar las rutas de las imágenes desde el archivo Excel
    data = pd.read_excel(path_excel)
    y = np.array(data['Clasificacion'])  # Etiquetas (Benigna/Maligna)
    
    paths = data['Ruta Relativa']  # Rutas relativas de los archivos DICOM
    
    # Construir las rutas absolutas
    paths_absolute = [os.path.join(path_directory, path[:-4]) for path in paths]
    
    images = []  # Lista para almacenar las matrices de píxeles
    sizes = []   # Lista para almacenar las dimensiones (alto, ancho) de cada imagen
    
    for path in paths_absolute:
        if os.path.exists(path):
            try:
                # Leer archivo DICOM
                dicom = py.dcmread(path)
                matrix = dicom.pixel_array  # Obtener la matriz de píxeles
                
                # Obtener las dimensiones de la imagen
                alto, ancho = matrix.shape
                images.append(matrix)
                sizes.append((alto, ancho))
            except Exception as e:
                print(f"Error al leer la imagen {path}: {e}")
    
    # Determinar el tamaño mínimo de las imágenes
    min_high, min_width = determineResize(sizes)
    
    #min_high, min_width = 512, 512
    
    print(f"Tamaño mínimo de las imágenes: {min_high} x {min_width}")
    
    x = preprocess(images, min_high, min_width)
    
    print("Dataset cargado correctamente")
    
    return x, y
"""
def loadDICOM(path, min_high, min_width ):
    
    Función para cargar y procesar una imagen DICOM

    Parameters
    ----------
    path : String
        Ruta absoluta del archivo DICOM a importar.

    Returns
    -------
    matrix_flatten : numpy.ndarray
        Vector 1D que representa la imagen redimensionada y aplanada.

    Raises
    ------
    FileNotFoundError
        Si el archivo DICOM no existe en la ruta especificada.
    ValueError
        Si el archivo DICOM no contiene una matriz válida.

    
    if not os.path.exists(path):
        raise FileNotFoundError(f"El archivo {path} no existe.")
        
    try:
        # Leer archivo DICOM
        dicom = py.dcmread(path)
        matrix = dicom.pixel_array            # matriz de píxeles de la imagen DICOM
        
        # Obtener las dimensiones de la imagen
        alto, ancho = matrix.shape
        images.append(matrix)
        sizes.append((alto, ancho))
        
    except Exception as e:
        raise ValueError(f"Error al procesar el archivo DICOM {path}: {e}")
        
    return 
"""
def preprocess(images, min_width, min_high):
    """
    Función para redimensionar las imágenes a un tamaño mínimo.

    Parameters
    ----------
    images : list
        Lista de matrices de píxeles de las imágenes DICOM.
    min_alto : int
        Altura mínima a la que redimensionar las imágenes.
    min_ancho : int
        Ancho mínimo a la que redimensionar las imágenes.

    Returns
    -------
    list
        Lista de matrices de píxeles redimensionadas.
    """
    
    resized_images = []
    
    for matrix in images:
        # Redimensionar la matriz de píxeles a las dimensiones mínimas
        matrix_resized = Image.fromarray(matrix).resize((min_width, min_high))
        resized_images.append(np.array(matrix_resized))
    
    images_flattened = [image.flatten() for image in resized_images]
    
    return images_flattened
    

def normalizeData(x):
    """
    Normalización de datos
    
    Esta función aplica una transformación a los datos para escalar cada característica 
    en un rango entre 0 y 1. Es especialmente útil cuando se desea garantizar que las 
    características tengan valores comparables y evitar que las diferencias en las magnitudes 
    de los datos afecten el rendimiento de los modelos.

    Parameters
    ----------
    x : numpy.ndarray
        Matriz de datos de entrada, donde cada fila representa una muestra y 
        cada columna una característica.

    Returns
    -------
    numpy.ndarray
        Matriz de datos escalados, donde cada característica está en el rango [0, 1].
    """
    scaler = MinMaxScaler()     # Instancia del MinMaxScaler de scikit-learn
    
    # Ajustar el escalador a los datos de entrada y transformar los valores
    return scaler.fit_transform(x)

"""--------------------------------------
Implementacion
"""

dataset_x, dataset_y = loadDataset(path_excel, path_directory)    # Cargar Dataset

dataset_x = normalizeData(dataset_x)    # Normalizar las características

# Dividir data set en entrenamiento y evaluacion 
train_x, test_x, train_y, test_y = train_test_split(dataset_x, dataset_y, test_size=0.2, stratify=dataset_y, random_state=5)
print("Dataset dividido en entrenamiento y prueba correctamente")

estimators = all_estimators(type_filter= "classifier")  # Obtener todos los clasificadores de scikit-learn

best_model = None
best_accuracy = 0

# Recorrer todos los clasificadores
for model_name, ClassifierClass in estimators: 
    try: 
        model = ClassifierClass()
        
        model.fit(train_x, train_y)
        
        pred_y = model.predict(test_x)
        
        accuracy = accuracy_score(test_y, pred_y)
        
        # Guardar cada modelo en un archivo
        dump(model, f"{model_name}.joblib")
        print(f"Model: {model_name}, Accuracy: {accuracy} - Guardado como {model_name}.joblib")
        
        if accuracy > best_accuracy:
            best_model = model
            best_accuracy = accuracy
        
        print(f"Model: {model_name}, Accuracy: {accuracy}")
        
    except Exception as e:
        # Algun modelo puede tener algun problema, por lo que queremos visualizar cual y donde
        print(f"Error with model {model_name}: {e}")
        
print(f"\nBest model : {best_model}, Accuracy: {best_accuracy}")

"""
Para usar alguno de los modelos se recomienda el siguiente script

from joblib import load

loaded_model = load("Nombre_del_Modelo.joblib")     # Cargar modelo

predictions = loaded_model.predict(nuevos_datos)    # Usar modelo para nuevas predicciones

"""