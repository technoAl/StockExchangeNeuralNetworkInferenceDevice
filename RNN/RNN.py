#!/usr/bin/env python
# coding: utf-8

# In[1]:


from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
import numpy as np


# In[2]:


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


# In[3]:


"""
Neural Network Model Creation
"""

"""
Based on code from https://medium.com/@sabber/classifying-yelp-review-comments-using-lstm-and-word-embeddings-part-1-eb2275e4066b
"""
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.layers import Dense, Input, LSTM, Embedding, Dropout, Activation
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
model1.add(Bidirectional(LSTM(128, return_sequences=True)))
model1.add(Bidirectional(LSTM(32, return_sequences=True)))
model1.add(GlobalMaxPool1D())
model1.add(Dense(16, activation='relu'))
model1.add(Dense(2,activation='softmax'))
model1.compile(loss='sparse_categorical_crossentropy', optimizer=tf.keras.optimizers.Adam(np.e), metrics=['accuracy'])
model1.summary()
x_trainR = np.array(x_train)[:2200]
y_trainR = np.array(y_train)[:2200]
x_test = np.array(x_train)[2200:]
y_test = np.array(y_train)[2200:]
x_train = x_trainR
y_train = y_trainR
x_train = pad_sequences(x_train, maxlen=maxlen)
x_test = pad_sequences(x_test, maxlen=maxlen)


# In[4]:


"""
Formats data labels
"""
def toFinal(a):
    for i in range(len(a)):
        if a[i] > 0:
            a[i] = 1
        else:
            a[i] = 0


# In[ ]:





# In[5]:


"""
Reads and formats data labels
"""
y_train = []
for i in range(14,32):
    try:
        with open("../TrainingData/TeslaTrainingData_2019-10-"+str(i)+"/Tesla.csv") as file:
            j = file.read().split(',')[1]
            assert j is not None
            j = j.replace('\n','')
            for _ in range(100):
                y_train.append(float(j))
    except:
        print("Bad: 10/"+str(i))
for i in range(1,22):
    try:
        with open("../TrainingData/TeslaTrainingData_2019-11-"+str(i)+"/Tesla.csv") as file:
            j = file.read().split(',')[1]
            assert j is not None
            j = j.replace('\n','')
            for _ in range(100):
                y_train.append(float(j))
    except:
        print("Bad: 11/"+str(i))
toFinal(y_train)
# print(y_train)
y_train = np.array(y_train)
# print(y_train)
print(y_train.shape)
print(x_train.shape)


# In[6]:


print(np.count_nonzero(y_train))
print(y_train.size)
y_test = y_train[1200:1500]
y_train = y_train[:1200]
x_test = x_train[1200:1500]
x_train = x_train[:1200]
print(x_test.shape)
print(y_test.shape)


# In[7]:


"""
Trains Model
"""
history = model1.fit(x_train, y_train, epochs=200, batch_size=50, validation_data=(x_test, y_test))


# In[ ]:


def predict(x):
    tokenizer = Tokenizer(num_words=max_features)
    tokenizer.fit_on_texts(list(x))
    x =pad_sequences(tokenizer.texts_to_sequences(x), maxlen=maxlen)
    return model1.predict(x)


# In[ ]:


print(model1.weights)


# In[ ]:


#get_ipython().run_line_magic('matplotlib', 'notebook')
import matplotlib.pyplot as plt
loss = history.history['loss']
plt.plot(loss)
acc = history.history['accuracy']
plt.plot(acc)
val_acc = history.history['val_accuracy']
plt.plot(val_acc)
plt.ylim(0,5)


# In[12]:


print(len(y_test[y_test==0]))
print(len(y_test[y_test==1]))


# In[ ]:




