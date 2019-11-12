#!/usr/bin/env python
# coding: utf-8

# In[1]:

from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow_datasets as tfds
import tensorflow as tf
dataset, info = tfds.load('imdb_reviews/subwords8k', with_info=True,
                          as_supervised=True)

encoder = info.features['text'].encoder
print ('Vocabulary size: {}'.format(encoder.vocab_size))

model = tf.keras.Sequential([
    tf.keras.layers.Embedding(encoder.vocab_size, 64),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64)),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])
model.compile(loss='binary_crossentropy',
              optimizer=tf.keras.optimizers.Adam(1e-4),
              metrics=['accuracy'])

# do model.fit() here once the data is ready

def pad_to_size(vec, size):
    zeros = [0] * (size - len(vec))
    vec.extend(zeros)
    return vec

def sample_predict(sentence, pad):
    encoded_sample_pred_text = encoder.encode(sentence)

    if pad:
        encoded_sample_pred_text = pad_to_size(encoded_sample_pred_text, 64)
    encoded_sample_pred_text = tf.cast(encoded_sample_pred_text, tf.float32)
    predictions = model.predict(tf.expand_dims(encoded_sample_pred_text, 0))

    return (predictions)

sample_pred_text = ('Tesla is doing really well now. It is going to be one of the greatest companies ever in the near future')
predictions = sample_predict(sample_pred_text, pad=False)
print (predictions)

sample_pred_text = ('Tesla is doing really bad now. It is going to be one of the worst companies ever in the near future')
predictions = sample_predict(sample_pred_text, pad=False)
print (predictions)




