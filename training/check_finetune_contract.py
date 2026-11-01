"""Small offline TensorFlow regression for nested metric state during fine-tuning."""
import numpy as np
import tensorflow as tf
from training.train import uncompiled_classifier

head = tf.keras.Sequential([tf.keras.Input((3,)), tf.keras.layers.Dense(5, activation="softmax", kernel_initializer="zeros")])
head.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
head.train_on_batch(np.zeros((2, 3)), np.array([0, 1]))
for variable in head.weights:
    variable.assign(tf.zeros_like(variable))
inputs = tf.keras.Input((3,))
model = tf.keras.Model(inputs, uncompiled_classifier(tf, head)(inputs))
model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
result = model.train_on_batch(np.zeros((2, 3)), np.array([0, 1]), return_dict=True)
np.testing.assert_allclose(result["loss"], np.log(5), atol=1e-5)
np.testing.assert_allclose(result["accuracy"], 0.5)
print("Fine-tuning reports the correct root loss and accuracy; nested metrics do not override them.")
