from typing import Dict,Text
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow_recommenders as tfrs
import os

#Ruta donde se guarda el modelo de la red neuronal
path = "modelo"

#Recuperamos el modelo con saved_model
loaded = tf.saved_model.load(path)

#Test de predicción
id_usuario = str(42)
scores,titles = loaded([id_usuario])
print(f"Recomendaciones para el usuario {id_usuario}: {titles[0, :3]}")



