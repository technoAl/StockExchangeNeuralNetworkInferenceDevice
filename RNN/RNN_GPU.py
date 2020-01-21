#!/usr/bin/env python
# coding: utf-8

# In[3]:


from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
import numpy as np


# In[4]:


"""
File Reading
"""
a = []
for i in range(14,31):
    try:
        for j in range(100):
            with open("../TrainingData/TeslaTrainingData_2019-10-"+str(i)+"/Tesla"+str(j)+".txt", mode='rb') as file:
                try:
                    a.append(str(file.read()))
                except Exception as e:
                    print(e)
                    print(file.name)
    except:
        print("10/"+str(i))
for i in range(1,30):
    if i != 10:
        try:
            for j in range(100):
                with open("../TrainingData/TeslaTrainingData_2019-11-"+str(i)+"/Tesla"+str(j)+".txt", mode='rb') as file:
                    try:
                        a.append(str(file.read()))
                    except Exception as e:
                        print(file.name)
        except:
            print("11/"+str(i))
print(len(a))


# In[5]:


"""
Neural Network Model Creation
"""

"""
Based on code from https://medium.com/@sabber/classifying-yelp-review-comments-using-lstm-and-word-embeddings-part-1-eb2275e4066b
"""
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.layers import Dense, Input, CuDNNLSTM, Embedding, Dropout, Activation
from tensorflow.keras.layers import Bidirectional, GlobalMaxPool1D
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras import layers
x_train = np.array(a, dtype=np.str)
x_train = x_train.astype(str)
# print(x_train.shape)
y_train = np.zeros(2700)+0.9
print(y_train.shape)
import tensorflow as tf
embed_size = 300 
max_features = 50000 
maxlen = 100
tokenizer = Tokenizer(num_words=max_features)
tokenizer.fit_on_texts(list(x_train))
x_train = tokenizer.texts_to_sequences(x_train)
model1 = tf.keras.Sequential()
model1.add(Embedding(max_features, embed_size, input_length=maxlen))
model1.add(Bidirectional(CuDNNLSTM(128, return_sequences=True)))
model1.add(GlobalMaxPool1D())
model1.add(Dense(128, activation='relu'))
model1.add(Dense(16, activation='relu'))
model1.add(Dense(2,activation='softmax'))
model1.compile(loss='sparse_categorical_crossentropy', optimizer=tf.keras.optimizers.Adam(), metrics=['accuracy'])
model1.summary()
x_trainR = np.array(x_train)[:2200]
y_trainR = np.array(y_train)[:2200]
x_test = np.array(x_train)[2200:]
y_test = np.array(y_train)[2200:]
x_train = x_trainR
y_train = y_trainR
x_train = pad_sequences(x_train, maxlen=maxlen)
x_test = pad_sequences(x_test, maxlen=maxlen)


# In[ ]:


model1.fit(x_train, y_train, epochs=5, batch_size=10)


# In[ ]:


def pad_to_size(vec, size):
    zeros = [0] * (size - len(vec))
    vec.extend(zeros)
    return vec


# In[ ]:


def sample_predict(sentence, pad):
    encoded_sample_pred_text = encoder.encode(sample_pred_text)

    if pad:
        encoded_sample_pred_text = pad_to_size(encoded_sample_pred_text, 64)
    encoded_sample_pred_text = tf.cast(encoded_sample_pred_text, tf.float32)
    predictions = model1.predict(tf.expand_dims(encoded_sample_pred_text, 0))

    return (predictions)


# In[ ]:


print(model1.weights)


# In[ ]:


def predict(x):
    tokenizer = Tokenizer(num_words=max_features)
    tokenizer.fit_on_texts(list(x))
    x =pad_sequences(tokenizer.texts_to_sequences(x), maxlen=maxlen)
    return model1.predict(x)
predict(["hello world I am alexander C sun", "world"])


# In[ ]:




