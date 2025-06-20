from typing import Dict,Text
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow_recommenders as tfrs
import os
import requests

videos_url = "http://127.0.0.1:8000/api/usuarios/train/request/videos/get/all"
videos_data = requests.get(videos_url).json()
diccionario_videos = {video['id']: video for video in videos_data}
#Ruta donde se guarda el modelo de la red neuronal
path = "modelo_guardado"

#Recuperamos el modelo con saved_model
loaded = tf.saved_model.load(path)
# Obtener los 5 videos más recomendados para ese usuario
id_usuario = str(14)
scores,titles = loaded([id_usuario])
titles_np = titles.numpy()
predicted_ids_int = [int(x) for x in titles_np[0]]

print(f"{predicted_ids_int}")
recommendations = [diccionario_videos[video_id] for video_id in predicted_ids_int]
print(f"Recomendaciones para el usuario {id_usuario}: {titles[0, :3]}")
for video in recommendations:
    print(f"Title: {video['title']}")
    print(f"Description: {video['description']}")
    print(f"URL: {video['video_url']}\n")