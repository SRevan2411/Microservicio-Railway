from fastapi import FastAPI, HTTPException
from typing import Dict,Text
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow_recommenders as tfrs
import os
import requests
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

#Agregar el cors para que no nos bloquee los fetch del react
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],  # O usa ["*"] para permitir todos los orígenes (solo en desarrollo)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

videos_url = "http://127.0.0.1:8000/api/usuarios/train/request/videos/get/all"
videos_data = requests.get(videos_url).json()
diccionario_videos = {video['id']: video for video in videos_data}
#Ruta donde se guarda el modelo de la red neuronal, la carpueta pues xd
path = "modelo_guardado"

#Loaded es el modelo cargado
loaded = tf.saved_model.load(path)
# Parte de la api, vamos a usar un get
@app.get("/recomendar/{id_usuario}")
def recomendar_videos(id_usuario:str):
    if not id_usuario:
        raise HTTPException(status_code=400,detail="Falta el id del usuario")
    try:
        scores,titles = loaded([id_usuario])
        titles_np = titles.numpy()
        predicted_ids_int = [int(x) for x in titles_np[0]]
        recomendaciones = []
        print(predicted_ids_int)
        for video_id in predicted_ids_int:
            if video_id in diccionario_videos:
                video = diccionario_videos[video_id]
                recomendaciones.append({
                    "id":video['id'],
                    "title":video['title'],
                    "description":video['description'],
                    "url":video["video_url"]
                })
        return {"recomendaciones":recomendaciones}
    except Exception as e:
        raise HTTPException(status_code=500,detail=f"Error al generar recomendaciones: {str(e)}")
