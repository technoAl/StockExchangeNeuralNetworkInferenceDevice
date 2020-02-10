# Stock Exchange Neural Network Inference Device

A neural network that takes an input of a certain stock company and outputs a prediction about whether the stock value will increase or decrease based on news data about the company from that day.

Uses TensorFlow 2.0/Keras for the neural network forward pass and back propogation.

Uses the Global Vector Library GloVe-50, which converts words to 50-dimensional vectors. The GloVe file can be downloaded [here](https://www.dropbox.com/s/c6m006wzrzb2p6t/glove.6B.50d.txt.w2v.zip?dl=0).

GloVe requires the Gensim library to work properly. This can be installed by running `pip install gensim`
