# Stock Exchange Neural Network Inference Device

A neural network that takes an input of a certain stock company and outputs a prediction about whether the stock value will increase or decrease based on news data about the company from that day.

[inference.py](inference.py) is the file to be run in order to run the application, model.npy is the saved NN.

The [RNN](RNN) folder contains all neural network training files and scripts

The [WebScraper](WebScraper) folder contains all web scraping files and scripts

The [TrainingData](TrainingData) folder shows all the files collected in the training process using [scraper.py](WebScraper/scraper.py)

## APIs + Libraries Utilized

Uses the [BeautifulSoup4](https://www.crummy.com/software/BeautifulSoup/bs4/doc/) library for all webscraping

Uses the [NewsApi](https://newsapi.org/) library to locate top articles from the day containing a query

The webscraping and NewsAPI require the lxml library. This can be installed by rinning `pip install lxml`

Uses the [MyGrad](https://github.com/rsokl/MyGrad) and [MyNN](https://github.com/davidmascharka/MyNN) libraries for the neural network forward pass and back propogation.

The [Numba](http://numba.pydata.org/) library is required to run inference. This can be installed by running `pip install numba`

Uses the Global Vector Library GloVe-50, which converts words to 50-dimensional vectors. The GloVe file can be downloaded [here](https://www.dropbox.com/s/c6m006wzrzb2p6t/glove.6B.50d.txt.w2v.zip?dl=0).

The [Gensim](https://pypi.org/project/gensim/) library is required to load the GloVe library. This can be installed by running `pip install gensim`


## Results

The neural network achieved a final accuracy of approximately 64.3 percent
