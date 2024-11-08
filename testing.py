import mlflow
import numpy as np
import tensorflow as tf
from tensorflow import keras

mlflow.tensorflow.autolog()

# Prepare data for a 2-class classification.
data = np.random.uniform(size=[20, 28, 28, 3])
label = np.random.randint(2, size=20)

model = keras.Sequential(
    [
        keras.Input([28, 28, 3]),
        keras.layers.Conv2D(8, 2),
        keras.layers.MaxPool2D(2),
        keras.layers.Flatten(),
        keras.layers.Dense(2),
        keras.layers.Softmax(),
    ]
)

model.compile(
    loss=keras.losses.SparseCategoricalCrossentropy(),
    optimizer=keras.optimizers.Adam(0.001),
    metrics=[keras.metrics.SparseCategoricalAccuracy()],
)
mlflow.set_experiment('testing')
mlflow.set_tracking_uri('http://172.18.100.83:5000')

with mlflow.start_run():
    model.fit(data, label, batch_size=5, epochs=2)
    mlflow.tensorflow.log_model(model, "model", keras_model_kwargs={"save_format":"h5"})
