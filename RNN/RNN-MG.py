#!/usr/bin/env python
# coding: utf-8

# In[1]:


from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
import numpy as np
import pandas as pd
import mygrad as mg
import re
from mynn.layers.dense import dense
from mynn.initializers.glorot_normal import glorot_normal
from mynn.optimizers.adam import Adam
from mygrad.nnet.losses import softmax_crossentropy
from gensim.models.keyedvectors import KeyedVectors
from noggin import create_plot

# In[2]:

glove = KeyedVectors.load_word2vec_format("glove.6B.50d.txt.w2v", binary=False)

x_bad = []
"""
File Reading
"""
a = []
for i in range(14, 32):
    if (i == 19 or i == 20 or i == 23 or i == 25 or i == 26 or i == 27): continue
    try:
        for j in range(100):
            with open("../TrainingData/TeslaTrainingData_2019-10-" + str(i) + "/Tesla" + str(j) + ".txt",
                      mode='rb') as file:

                try:
                    a.append(str(file.read()))
                except Exception as e:
                    print(e)
                    print(file.name)
    except:
        print("10/" + str(i))
        x_bad.append("10/" + str(i))
for i in range(1, 22):
    if i == 2 or i == 3 or i == 9 or i == 10 or i == 16 or i == 17 or i == 22 or i == 23 or i == 24: continue
    if i != 10:
        try:
            for j in range(100):
                with open("../TrainingData/TeslaTrainingData_2019-11-" + str(i) + "/Tesla" + str(j) + ".txt",
                          mode='rb') as file:
                    try:
                        a.append(str(file.read()))
                    except Exception as e:
                        print(file.name)
        except:
            print("11/" + str(i))
            x_bad.append("11/" + str(i))

print(len(a))

# In[3]:


x_train = np.array(a, dtype=np.str)
x_train = x_train.astype(str)
# print(x_train.shape)


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


"""
Reads and formats data labels
"""
y_bad = []
y_train = []
for i in range(14, 32):
    if i == 19 or i == 20 or i == 23 or i == 25 or i == 26 or i == 27: continue
    try:
        with open("../TrainingData/TeslaTrainingData_2019-10-" + str(i) + "/Tesla.csv") as file:
            j = file.read().split(',')[1]
            assert j is not None
            j = j.replace('\n', '')
            for _ in range(100):
                y_train.append(float(j))
    except:
        print("Bad: 10/" + str(i))
        y_bad.append("10/" + str(i))

for i in range(1, 22):
    if (i == 2 or i == 3 or i == 9 or i == 10 or i == 16 or i == 17 or i == 22 or i == 23 or i == 24): continue

    try:
        with open("../TrainingData/TeslaTrainingData_2019-11-" + str(i) + "/Tesla.csv") as file:
            j = file.read().split(',')[1]
            assert j is not None
            j = j.replace('\n', '')
            for _ in range(100):
                y_train.append(float(j))
    except:
        print("Bad: 11/" + str(i))
        y_bad.append("11/" + str(i))

toFinal(y_train)
# print(y_train)
y_train = np.array(y_train)

print(y_train.shape)
print(x_train.shape)

x_trainR = np.array(x_train)[:2200]
y_trainR = np.array(y_train)[:2200]
x_test = np.array(x_train)[2200:]
y_test = np.array(y_train)[2200:]
x_train = x_trainR
y_train = y_trainR
# print(y_train)
print(y_train.shape, y_test.shape)
print(x_train.shape, x_test.shape)




print(len(y_test[y_test == 0]))
print(len(y_test[y_test == 1]))

