from typing import Dict,Text
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow_recommenders as tfrs
import os
import requests

#urls de django
history_url = "https://django-railway-production-c5df.up.railway.app/api/usuarios/train/request/history/get/all"
videos_url = "https://django-railway-production-c5df.up.railway.app/api/usuarios/train/request/videos/get/all"

#Parte para hacerle el fetch a django
history_data = requests.get(history_url).json()
videos_data = requests.get(videos_url).json()

diccionario_videos = {video['id']: video for video in videos_data}

#Se necesitan preprocesar los datos para hacer que los datos sean strings
history_examples = [{"user": str(x["user"]), "video": str(x["video"])} for x in history_data]

# Creación del dataset de entrenamiento desde los datos preprocesados
history = tf.data.Dataset.from_generator(
    lambda: (x for x in history_examples),
    output_signature={
        "user": tf.TensorSpec(shape=(), dtype=tf.string),
        "video": tf.TensorSpec(shape=(), dtype=tf.string),
    }
)

videos = tf.data.Dataset.from_tensor_slices([str(video["id"]) for video in videos_data])

#revolver los datos y separar entrenamiento de pruebas (relación 70/30)
total_datos = len(history_data)
tam_entrenamiento = int(total_datos * 0.7) #usamos int para eliminar decimales

#revoltura
shuffled = history.shuffle(total_datos, seed=42, reshuffle_each_iteration=False)
train = shuffled.take(tam_entrenamiento)
test = shuffled.skip(tam_entrenamiento)

#Crear vocabularios para usuarios y videos
user_ids = history.map(lambda x:x["user"])
video_ids = history.map(lambda x: x["video"])


user_ids_list = []
video_ids_list = []
for item in user_ids:
    user_ids_list.append(item.numpy().decode("utf-8"))
for item in video_ids:
    video_ids_list.append(item.numpy().decode("utf-8"))

unique_user_ids = np.unique(user_ids_list)
unique_video_ids = np.unique(video_ids_list)

# Modelos de embedding
embedding_dimension = 8

user_model = tf.keras.Sequential([
    tf.keras.layers.StringLookup(vocabulary=unique_user_ids, mask_token=None),
    tf.keras.layers.Embedding(len(unique_user_ids) + 1, embedding_dimension)
])

video_model = tf.keras.Sequential([
    tf.keras.layers.StringLookup(vocabulary=unique_video_ids, mask_token=None),
    tf.keras.layers.Embedding(len(unique_video_ids) + 1, embedding_dimension)
])

# Métrica y tarea
metrics = tfrs.metrics.FactorizedTopK(
    candidates=videos.batch(128).map(video_model)
)
task = tfrs.tasks.Retrieval(metrics=metrics)
# Modelo TFRS
class VideoRecModel(tfrs.Model):
    def __init__(self, user_model, video_model):
        super().__init__()
        self.user_model = user_model
        self.video_model = video_model
        self.task = task

    def compute_loss(self, features: Dict[Text, tf.Tensor], training=False) -> tf.Tensor:
        user_embeddings = self.user_model(features["user"])
        video_embeddings = self.video_model(features["video"])
        return self.task(user_embeddings, video_embeddings)

# Compilar y entrenar
model = VideoRecModel(user_model, video_model)
model.compile(optimizer=tf.keras.optimizers.Adagrad(learning_rate=0.005))

cached_train = train.batch(16).cache()
cached_test = test.batch(16).cache()

model.fit(cached_train, epochs=6)
model.evaluate(cached_test, return_dict=True)

# Índice de recomendación

'''
index = tfrs.layers.factorized_top_k.BruteForce(model.user_model)
dataset_for_index = tf.data.Dataset.zip((videos.batch(100), videos.batch(100).map(video_model)))
index.index_from_dataset(
    tf.data.Dataset.zip((videos.batch(100), videos.batch(100).map(video_model)))
)

'''
index = tfrs.layers.factorized_top_k.BruteForce(model.user_model)
index.index_from_dataset(videos.batch(len(videos_data)).map(lambda x: (x, video_model(x))))


valor_K = 10

# Prueba para ver si si recomienda 
_, titles = index(tf.constant(["7"]),k=valor_K)
print(f"Recommendations for user 7: {titles[0, :3]}")

# Guardar modelo
path = os.path.join(os.getcwd(),"APITensorflow\modelo_guardado")
tf.saved_model.save(index,path)
print(f"Modelo guardado en: {path}")




# Número de recomendaciones a evaluar (por ejemplo, top-10)
K = 5

# Crear el índice para generar recomendaciones
index = tfrs.layers.factorized_top_k.BruteForce(model.user_model)
index.index_from_dataset(videos.batch(100).map(lambda x: (x, video_model(x))))

# Obtener predicciones y verdad real
true_videos = []
pred_videos = []

for example in test:
    user_id = example["user"].numpy().decode("utf-8")
    true_video = example["video"].numpy().decode("utf-8")

    # Obtener las K recomendaciones para el usuario
    _, recommended = index(tf.constant([user_id]), k=K)

    true_videos.append(true_video)
    pred_videos.append([vid.numpy().decode("utf-8") for vid in recommended[0]])

# Calcular métricas
aciertos = sum(true_videos[i] in pred_videos[i] for i in range(len(true_videos)))
total = len(true_videos)

precision_at_k = aciertos / (total * K)
recall_at_k = aciertos / total
error_rate = 1 - recall_at_k

print(f"Precision@{K}: {precision_at_k:.4f}")
print(f"Recall@{K}: {recall_at_k:.4f}")
print(f"Error rate: {error_rate:.4f}")

