# Stock Exchange Neural Network Inference Device

A neural network that takes an input of a certain stock company and outputs a prediction about whether the stock value will increase or decrease based on news data about the company from that day.

inference.py is the file to be run in order to run the application, model.npy is the saved NN.

The RNN folder contains all neural network training files and scripts

The WebScraper folder contains all web scraping files and scripts

The trainingdata folder shows all the files collected in the training process using scraper.py

Uses the [BeautifulSoup4](https://www.crummy.com/software/BeautifulSoup/bs4/doc/) library for all webscraping

Uses the [NewsApi](https://newsapi.org/) library to locate top articles from the day containing a query

Uses the [MyGrad](https://github.com/rsokl/MyGrad) library for the neural network forward pass and back propogation.

Uses the Global Vector Library GloVe-50, which converts words to 50-dimensional vectors. The GloVe file can be downloaded [here](https://www.dropbox.com/s/c6m006wzrzb2p6t/glove.6B.50d.txt.w2v.zip?dl=0).

