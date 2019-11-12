import requests
import json
from bs4 import BeautifulSoup
from newsapi import NewsApiClient
import os
import requests
from iexfinance.stocks import Stock
from iexfinance.stocks import get_historical_data
from datetime import datetime
import numpy as np
import csv

def obtainTrainingData(code, year, month, day):
       date = str(year) + '-' + str(month) + '-' + str(day)
       query = get_symbol(code).split(',')[0]
       url = ('https://newsapi.org/v2/everything?'
              'q=' + query + '&'
              'from=' + date + '&'
              'sortBy=popularity&'
              'apiKey=13bd628fa8b548738d3b113d9442574e&'
              'pageSize=40&'
              'language=en')
       response = requests.get(url)
       data = response.json()
       print(data)
       print(data['totalResults'])
       data = data['articles']
       print(len(data))
       count = 0
       path = '../'+ 'TrainingData' + '/' + query + 'TrainingData' + '_' +  date +'/'
       for i in data:
              tmpDict = dict(i)
              fullName = path + query + str(count) + '.txt'
              os.makedirs(os.path.dirname(fullName),exist_ok=True)
              file = open(fullName, 'w')
              try:
                     page = requests.get(tmpDict['url'])
              except:
                     print('page failed')
              if page.status_code == 200:
                     soup = BeautifulSoup(page.content, "lxml")
                     all_tags = soup.find_all('p')
                     for i in all_tags:
                            try:
                                   file.write(i.get_text())
                            except Exception as inst:
                                   print('darned emojis')
              else:
                     print("unfortunate failure")
              file.close()
              count+=1
       fullName = path + query + '.csv'
       os.makedirs(os.path.dirname(fullName), exist_ok=True)
       with open (fullName, 'w', newline='') as csvfile:
              writer = csv.writer(csvfile , delimiter=',')
              writer.writerow([query] + [getStock(code, year, month, day)])


def get_symbol(symbol):
    url = "http://d.yimg.com/autoc.finance.yahoo.com/autoc?query={}&region=1&lang=en".format(symbol)
    result = requests.get(url).json()
    for x in result['ResultSet']['Result']:
        if x['symbol'] == symbol:
            return x['name']

def getStock(code, year, month, day):
    start = datetime(year, month, day)
    stock = get_historical_data(code, start, start, token='pk_3fc4f2751a6746f3b1cdc30763095572')
    dict = stock[str(year) + '-' + str(month) + '-' + str(day)]
    return dict['close'] - dict['open']

if __name__ == '__main__':
       for month in range(1,11):
              for day in range (1, 31):
                     print(str(month) + '-' + str(day) + '\n')
                     obtainTrainingData('TSLA', 2019, month, day)

