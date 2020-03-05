import requests
import json
from bs4 import BeautifulSoup
from newsapi.newsapi_client import NewsApiClient
import os
import requests
from datetime import datetime
import csv
#lxml parser is also needed to run bs4 code

def obtainArticleContent(code, year, month, day): # grabs and saves the 100 files of content/new articles for the neural net to interpret
    inputData = []
    date = str(year) + '-' + str(month) + '-' + str(day)
    query = getQuery(code)
    if not isWeekday(datetime(year, month, day)):# stocks are not open on weekends, it skips this
        print('weekend, no stocks today')
        return
    url = makeURL(date, query, 100)# Calls make URL
    data = getData(url)# Calls get Data
    if data == None:
        return
    for i in data:
        tmpDict = dict(i)
        try:
            page = requests.get(tmpDict['url'])
        except:
            print('page failed')
        if page.status_code == 200:# 200 means success
            soup = BeautifulSoup(page.content, "lxml") # uses lxml parser
            all_tags = soup.find_all('p')# Finds all <p> tags
            for i in all_tags:
                try:
                    inputData.append(i.get_text())# adds all to the list
                except Exception as inst:
                    print('character error')
        else:
            print("unfortunate failure, page failure")
    return inputData # returns the data

def getData(url): # uses response to parse the internet for each data page
    try:
        response = requests.get(url)
        data = response.json()
        data = data['articles']
        return data
    except:
        print('unfortunate failure')
        return

#makes the url in the format newapi prefers
def makeURL(query, date, pageSize):
    return ('https://newsapi.org/v2/everything?'
            'q=' + query + '&'
            'from=' + date + '&'
            'sortBy=popularity&'
            'pageSize=' + str(pageSize) + '&'
            'apiKey=13bd628fa8b548738d3b113d9442574e&'
            'language=en')

#checks if the day is a weekday, if not don't do anything
def isWeekday(today):
    if today.weekday() >= 5:
        return False
    else:
        return True

#gets query from the stock code
def getQuery(code):
    return get_symbol(code).split(',')[0]

def get_symbol(symbol):
    url = "http://d.yimg.com/autoc.finance.yahoo.com/autoc?query={}&region=1&lang=en".format(symbol)
    result = requests.get(url).json()
    for x in result['ResultSet']['Result']:
        if x['symbol'] == symbol:
            return x['name']

import mygrad as mg
import numpy as np
from mynn.layers.dense import dense
from mynn.initializers.glorot_normal import glorot_normal
from mynn.optimizers.adam import Adam
from mygrad.nnet.losses import softmax_crossentropy
from gensim.models.keyedvectors import KeyedVectors
from noggin import create_plot

class RNN:  # The RNN class, which passes the data through a gated recurrent unit to convert each sentence into an array
    def __init__(self, dim_input, dim_recurrent, dim_output):
        """ Initializes all layers needed for RNN

        Parameters
        ----------
        dim_input: int
            Dimensionality of data passed to RNN (C)

        dim_recurrent: int
            Dimensionality of hidden state in RNN (D)

        dim_output: int
            Dimensionality of output of RNN (K)
        """

        self.fc_h2y = dense(dim_recurrent, dim_output, weight_initializer=glorot_normal)
        self.Uz = mg.Tensor(
            np.random.randn(dim_input * dim_recurrent).reshape(dim_input, dim_recurrent)
        )
        self.Wz = mg.Tensor(
            np.random.randn(dim_recurrent * dim_recurrent).reshape(
                dim_recurrent, dim_recurrent
            )
        )
        self.bz = mg.Tensor(np.random.randn(dim_recurrent))
        self.Ur = mg.Tensor(
            np.random.randn(dim_input * dim_recurrent).reshape(dim_input, dim_recurrent)
        )
        self.Wr = mg.Tensor(
            np.random.randn(dim_recurrent * dim_recurrent).reshape(
                dim_recurrent, dim_recurrent
            )
        )
        self.br = mg.Tensor(np.random.randn(dim_recurrent))
        self.Uh = mg.Tensor(
            np.random.randn(dim_input * dim_recurrent).reshape(dim_input, dim_recurrent)
        )
        self.Wh = mg.Tensor(
            np.random.randn(dim_recurrent * dim_recurrent).reshape(
                dim_recurrent, dim_recurrent
            )
        )
        self.bh = mg.Tensor(np.random.randn(dim_recurrent))

    def __call__(self, x):
        """ Performs the full forward pass for the RNN.

        Note that we only care about the last y - the final classification scores for the full sequence

        Parameters
        ----------
        x: Union[numpy.ndarray, mygrad.Tensor], shape=(T, C)
            The one-hot encodings for the sequence

        Returns
        -------
        mygrad.Tensor, shape=(1, K)
            The final classification of the sequence
        """

        h = mg.nnet.gru(
            x,
            self.Uz,
            self.Wz,
            self.bz,
            self.Ur,
            self.Wr,
            self.br,
            self.Uh,
            self.Wh,
            self.bh,
        )
        return self.fc_h2y(h[-1])

    @property
    def parameters(self):
        """ A convenience function for getting all the parameters of our model.

        This can be accessed as an attribute, via `model.parameters`

        Returns
        -------
        Tuple[Tensor, ...]
            A tuple containing all of the learnable parameters for our model
        """
        return self.fc_h2y.parameters + (
        self.Uz, self.Wz, self.bz, self.Ur, self.Wr, self.br, self.Uh, self.Wh, self.bh)

def to_glove(sentence):
    out = []
    for word in sentence.split():
        word = word.lower()
        try:
            out.append(glove[word])
        except:
            continue
    if len(out) > MAXLEN:
        out = out[:MAXLEN]
    elif len(out) < MAXLEN:
        for _ in range(len(out), MAXLEN):
            out.append(np.zeros(50))
    if len(out) != MAXLEN:
        print("BAAAAAAAAD")
    return out

"""
Takes in a single sentence and runs inference to determine whether the stock value will increase or decrease
"""

def predict(sentence):
    sentence = to_glove(sentence)
    w = np.ascontiguousarray(np.swapaxes(np.array(sentence).reshape(1, 100, 50), 0, 1))
    pred = Keys[np.argmax(model(w))]
    print(pred)


# In[ ]:


"""
Takes in a list of sentences about a given stock and determines whether the value of the stock will increase or decrease depending on whether there are more positive results or more negative results
"""
def predict(multiple_sentences):
    good = 0
    bad = 0
    pred = 0
    for sentence in multiple_sentences:
        sentence = to_glove(sentence)
        w = np.ascontiguousarray(np.swapaxes(np.array(sentence).reshape(1, 100, 50), 0, 1))
        pred = np.argmax(model(w))
        if pred==1:
            good +=1
        else:
            bad += 1
    if good > bad:
        pred = Keys[1]
        print(pred, good/(good+bad)*100, "percent sure")
    else:
        pred = Keys[0]
        print(pred, bad/(good+bad)*100, "percent sure")

if __name__ == '__main__':
    print('Type Code')
    code = str(input())
    print('Type Month')
    month = int(input())
    print('Type day')
    day = str(input())
    print('Type year')
    year = int(input())
    inputData = obtainArticleContent(code, month, day, year)  # example data collection
    params = np.load("model.npy", allow_pickle=True)  # loads trained model
    Keys = ["UP", "DOWN"]
    model = RNN(50, 16, 2)
    MAXLEN = 100
    model.fc_h2y.weight, model.fc_h2y.bias, model.Uz, model.Wz, model.bz, model.Ur, model.Wr, model.br, model.Uh, model.Wh, model.bh = (
        params[0],
        params[1],
        params[2],
        params[3],
        params[4],
        params[5],
        params[6],
        params[7],
        params[8],
        params[9],
        params[10]
    )
    glove = KeyedVectors.load_word2vec_format("glove.6B.50d.txt.w2v", binary=False)
    print(code, day, month, year)
    predict(inputData)


