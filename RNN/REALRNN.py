import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.layers import Dense, Input, LSTM, Embedding, Dropout, Activation
from tensorflow.keras.layers import Bidirectional, GlobalMaxPool1D
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras import layers


# Others
import nltk
import string
import numpy as np
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
import re

a = []
for i in range(14,31):
    try:
        for j in range(100):
            with open("../TrainingData/TeslaTrainingData_2019-10-"+str(i)+"/Tesla"+str(j)+".txt", mode='rb') as file:
                try:
                    a.append(str(file.read()))
                except Exception as e:
                    print(e)
                    # print(file.name)
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
                        print(e)
                        # print(file.name)
        except:
            print("11/"+str(i))
print(len(a))

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
model1.add(Bidirectional(LSTM(64, return_sequences=True)))
model1.add(Dense(64, activation='relu'))
model1.add(Dense(1,activation='softmax'))
model1.compile(loss='binary_crossentropy', optimizer=tf.keras.optimizers.Adam(np.e-4), metrics=['accuracy'])
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