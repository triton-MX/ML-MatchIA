# -*- coding: utf-8 -*-
"""
Created on Mon Dec  2 09:56:27 2024

@author: Triton Perea
"""

# Librerias a usar
import pandas as pd
import os
import pydicom as py
from PIL import Image
import matplotlib.pyplot as plt
from transformers import ViTForImageClassification, ViTFeatureExtractor
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import Compose, Resize, ToTensor, Normalize
import torch.nn.functional as F
import torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, roc_curve, auc

# Configuracion del dispositivo
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Rutas
path_directory = r""    # Directorio con archivos DICOM
path_excel = r""        # Archivo Excel con rutas relativas y etiquetas

check_dir = "model_checkpoints" #os.path.join(path_directory,"Model_checkpoints")
os.makedirs(check_dir, exist_ok=True)
path_best_model = os.path.join(check_dir, "best_model.pth")

# Parametros
epochs= 20          # Numero de epocas
min_high= 224       # Altura de imagenes a reescalar
min_width= 224      # Ancho de imagenes a reescalar
patience = 5        # Paciencia para "early stopping"

label_mapping = {"Benigno": 0, "Maligno": 1}  # Diccionario de mapeo

# Listas para almacenar los valores de loss y accuracy
train_losses, val_losses = [], []
train_accuracies, val_accuracies = [], []

"""----------------------------------------------------------------
Funciones para cargar Dataset
"""
def loadDataset(path_excel, path_directory):
    """
    Carga las rutas absolutas de los archivos DICOM y las etiquetas.

    Parameters
    ----------
    path_excel : str
        Ruta al archivo Excel con información del dataset.
    path_directory : str
        Ruta base al directorio que contiene los archivos DICOM.

    Returns
    -------
    paths_absolute : list
        Lista con rutas absolutas de los archivos DICOM.
    labels : pandas.Series
        Etiquetas asociadas a cada archivo.
    """
    data_excel = pd.read_excel(path_excel)      # Obtener datos desde excel
    labels = data_excel["Clasificacion"].map(label_mapping)        # Lista de etiquetas
    paths_relative = data_excel["Ruta Relativa"]    # Rutas relativas de los DICOM

    # Construir rutas absolutas (remueve la extensión PNG y añade la carpeta base)
    paths_absolute= [os.path.join(path_directory,path[:-4]) for path in paths_relative]
    
    return paths_absolute, labels

def preprocessDICOM(path):
    """
    Preprocesa una imagen DICOM: la convierte en tensor normalizado.

    Parameters
    ----------
    path : str
        Ruta al archivo DICOM.
    transform : torchvision.transforms.Compose
        Transformaciones a aplicar a la imagen.

    Returns
    -------
    torch.Tensor
        Imagen procesada y lista para el modelo.
        
    Raises
    ------
    FileNotFoundError
        Si el archivo DICOM no existe en la ruta especificada.
    ValueError
        Si el archivo DICOM no contiene una matriz válida.

    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"El archivo {path} no existe.")
    try:    
        data = py.dcmread(path)             # Importar datos del DICOM
        matrix = data.pixel_array           # Extraer matriz de imagen
        image = Image.fromarray(matrix)     # Convierte la matriz en una imagen PIL
        
        # Asegurar el modo adecuado (por ejemplo, RGB)
        if image.mode != "RGB":
            image = image.convert("RGB")
    except Exception as e:
        raise ValueError(f"Error al procesar el archivo DICOM {path}: {e}")
    
    return image

"""----------------------------------------------------------------
Evaluacion de modelo
"""
def evaluateModel(loader, model):
    """
    Evalúa el rendimiento de un modelo de clasificación en un conjunto de datos.
    
    Parameters
    ----------
    loader : torch.utils.data.DataLoader
        Objeto DataLoader que contiene el conjunto de datos para evaluar. 
        Proporciona lotes de imágenes de entrada y sus etiquetas correspondientes.
    model : torch.nn.Module
        El modelo a evaluar. Debe ser compatible con el conjunto de datos y 
        soportar la devolución tanto de la pérdida como de los logits durante la inferencia.
    
    Returns
    -------
    avg_loss : float
        La pérdida promedio sobre todos los lotes en el conjunto de datos.
    accuracy : float
        La precisión general del modelo en el conjunto de datos, calculada como
        la proporción de predicciones correctas.
    class_report : str
        Un informe de clasificación detallado, que incluye precisión, recall,
        F1-score y soporte para cada clase.
    
    Notes
    -----
    - El modelo se establece en modo evaluación (`model.eval()`) durante esta 
      función para desactivar capas específicas de entrenamiento, como el dropout.
    - No se calculan gradientes (`torch.no_grad()`), lo que reduce el uso de 
      memoria y acelera el cálculo durante la evaluación.
    """
    model.eval()    # Cambiar a modo evaluacion
    all_labels = []
    all_preds, all_probs= [], [] # Obtener las probabilidades para las clases positivas
    total_loss = 0.0
    
    with torch.no_grad():    # No calcular gradientes
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images, labels=labels)
            loss = outputs.loss
            total_loss += loss.item()
            
            #Predicciones
            logits = outputs.logits     # Salidas antes de softmax
            preds = torch.argmax(F.softmax(logits, dim=-1), dim=-1)
            probs = F.softmax(logits, dim=-1)[:, 1]  # Probabilidades para la clase positiva

            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            
        avg_loss = total_loss / len(loader)
        accuracy = accuracy_score(all_labels, all_preds)
        class_report = classification_report(all_labels, all_preds)
        
        # Calcular la curva ROC y AUC
        fpr, tpr, _ = roc_curve(all_labels, all_probs)
        roc_auc = auc(fpr, tpr)
        
        # Graficar la curva ROC
        plt.figure()
        plt.plot(fpr, tpr, color='blue', label=f'ROC curve (AUC = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic')
        plt.legend(loc='lower right')
        plt.show()
        
        return avg_loss, accuracy, class_report

"""-----------------------------------------------------------------
Dataset personalizado
"""
class MammogramDataset(Dataset):
    def __init__(self, paths, labels, transform):
        """
        Inicializa el dataset personalizado.

        Parameters
        ----------
        paths : list
            Lista de rutas absolutas de las imágenes DICOM.
        labels : pandas.Series
            Etiquetas correspondientes.
        transform : torchvision.transforms.Compose
            Transformaciones a aplicar a las imágenes.
        """
        self.paths = paths
        self.labels = labels
        self.transform = transform
        
    def __len__(self):
        return len(self.paths)
    
    def __getitem__(self, idx):
        path = self.paths[idx]
        label = self.labels.iloc[idx]
        image = preprocessDICOM(path)
        image = self.transform(image)
        
        return image, torch.tensor(label, dtype=torch.long)

"""----------------------------------------------------------------
Transformer
"""
# Cargar modelo preentrenado ViT para clasificación de imágenes, con 2 etiquetas (por ejemplo, benigno/maligno)
model = ViTForImageClassification.from_pretrained("google/vit-base-patch16-224-in21k", num_labels=2)
model.to(device)  # Mover el modelo al dispositivo (GPU o CPU)

# Cargar el extractor de características preentrenado asociado al modelo ViT
feature_extractor = ViTFeatureExtractor.from_pretrained("google/vit-base-patch16-224-in21k")

# Definir la transformación de preprocesado de las imágenes
transform = Compose([
    Resize((min_high,min_width)),  # Redimensionar las imágenes al tamaño requerido
    ToTensor(),  # Convertir la imagen a tensor
    # Normalize(mean=feature_extractor.image_mean, std= feature_extractor.image_std)  # Normalizar con los valores de media y desviación estándar del extractor
    Normalize(mean=0.5, std=0.5)
    ])

"""--------------------------------------------------------------
Entrenamiento
"""
# Cargar el conjunto de datos desde un archivo Excel y un directorio de imágenes
paths, labels = loadDataset(path_excel, path_directory)

# Dividir los datos en conjunto de entrenamiento+validación y prueba (80-20)
train_val_images, test_images, train_val_labels, test_labels = train_test_split(
    paths, labels, test_size=0.2, stratify= labels, random_state = 42 )

# Dividir el conjunto de entrenamiento+validación en entrenamiento y validación (70-30)
train_images, val_images, train_labels, val_labels = train_test_split(
    train_val_images, train_val_labels, test_size=0.3, stratify=train_val_labels, random_state=42)

# Imprimir la cantidad de imágenes en cada conjunto
print(f"Entrenamiento: {len(train_images)}")
print(f"Validacion: {len(val_images)}")
print(f"Prueba: {len(test_images)}")

# Integrar los conjuntos de datos personalizados en formato adecuado para PyTorch
train_dataset = MammogramDataset(train_images, train_labels, transform)
val_dataset = MammogramDataset(val_images, val_labels, transform)
test_dataset = MammogramDataset(test_images, test_labels, transform)

# Crear los DataLoaders para los conjuntos de datos, con un tamaño de batch de 16
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

# Inicializar el optimizador AdamW con una tasa de aprendizaje pequeña (5e-5)
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)

# Inicializar la pérdida de validación más baja como infinita y la variable para contar épocas sin mejora
best_val_loss = float("inf")
epoch_no_improve = 0


# Loop de entrenamiento
for epoch in range(epochs):
    print(f"Epoca {epoch + 1}/ {epochs}")
    epoch_loss = 0.0
    epoch_corrects = 0
    total_train = 0
    model.train()  # Poner el modelo en modo entrenamiento
    
    # Iterar sobre los lotes de entrenamiento
    for batch_idx, (images, labels) in enumerate(train_loader):
        images, labels = images.to(device), labels.to(device)  # Mover imágenes y etiquetas al dispositivo
        optimizer.zero_grad()  # Resetear los gradientes
        outputs = model(images, labels=labels)  # Pasar las imágenes por el modelo
        loss = outputs.loss  # Calcular la pérdida
        loss.backward()  # Propagación hacia atrás para calcular los gradientes
        optimizer.step()  # Actualizar los parámetros del modelo
        epoch_loss += loss.item()  # Acumular la pérdida

        # Calcular accuracy para entrenamiento
        logits = outputs.logits
        preds = torch.argmax(F.softmax(logits, dim=-1), dim=-1)
        epoch_corrects += (preds == labels).sum().item()
        total_train += labels.size(0)        

        if batch_idx % 10 == 0:  # Imprimir cada 10 lotes
            print(f"Batch {batch_idx}, Loss: {loss.item():.4f}")

    # Calcular y mostrar la pérdida promedio de la época
    avg_train_loss = epoch_loss / len(train_loader)
    train_accuracy = epoch_corrects / total_train
    train_losses.append(avg_train_loss)
    train_accuracies.append(train_accuracy)
    print(f"Average Loss for Epoch {epoch+1}: {avg_train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}")
    
    # Validación después de cada época
    val_loss, val_accuracy, val_report = evaluateModel(val_loader, model)
    val_losses.append(val_loss)
    val_accuracies.append(val_accuracy)
    print(f"Validacion - perdida: {val_loss:.4f}, Accuracy: {val_accuracy:.4f}")
    print(val_report)
    
    # Implementar Early Stopping: si la pérdida de validación mejora, guardar el modelo
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        epoch_no_improve = 0
        torch.save(model.state_dict(), path_best_model)  # Guardar el mejor modelo
        print("Nuevo mejor modelo guardado")
    else:
        epoch_no_improve += 1  # Incrementar el contador de épocas sin mejora
        print(f"No mejora por {epoch_no_improve} epocas consecutivas.")
    
    # Si no hay mejora después de 'patience' épocas, detener el entrenamiento
    if epoch_no_improve >= patience:
        print("Deteniendo entrenamiento por falta de mejora.")
        break

"""-----------------------------------------------------------------------
Evaluación de mejor modelo
"""
# Cargar el mejor modelo guardado durante el entrenamiento
model.load_state_dict(torch.load(path_best_model))

# Evaluar el modelo en el conjunto de prueba
test_loss, test_accuracy, test_report = evaluateModel(test_loader, model)
print(f"Prueba - perdida: {test_loss:.4f}, Accuracy: {test_accuracy:.4f}")
print(test_report)


# Graficar Loss y Accuracy durante el entrenamiento
epochs_range = range(1, len(train_losses) + 1)

# Gráfico de Loss
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, train_losses, label='Train Loss', color='blue')
plt.plot(epochs_range, val_losses, label='Validation Loss', color='red')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

# Gráfico de Accuracy
plt.subplot(1, 2, 2)
plt.plot(epochs_range, train_accuracies, label='Train Accuracy', color='blue')
plt.plot(epochs_range, val_accuracies, label='Validation Accuracy', color='red')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

# Mostrar las gráficas
plt.show()