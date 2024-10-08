import tensorflow as tf
import torch
# https://discuss.tensorflow.org/t/tensorflow-operator-computational-graph/3759
model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(2, activation='softmax', input_shape=(1,))
])

model.compile(optimizer='rmsprop', loss='categorical_crossentropy')

xs = tf.constant([[1.0]])
ys = tf.constant([[0.2, 0.8]])

model.train_on_batch(xs, ys)

graph = model.train_function.get_concrete_function(iter([(xs, ys)])).graph  # The concrete function takes an iterator
isinstance(graph, tf.Graph)  # True

print(len(graph.get_operations()))
for op in graph.get_operations():
    print(op.colocation_groups())
