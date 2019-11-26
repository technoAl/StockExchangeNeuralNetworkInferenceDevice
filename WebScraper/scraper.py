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
       query = getQuery(code)
       if not isWeekday(datetime(year, month, day)):
              print('weekend')
              return
       url = makeURL(date, query, 100)
       response = requests.get(url)
       data = response.json()
       data = data['articles']
       count = 0
       path = '../'+ 'TrainingData' + '/' + query + 'TrainingData' + '_' +  date +'/'
       for i in data:
              tmpDict = dict(i)
              fullName = path + query + str(count) + '.txt'
              os.makedirs(os.path.dirname(fullName), exist_ok=True)
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
       path = '../' + 'TrainingData' + '/'
       appendStock(fullName, path)

def appendStock(fullName, path):
       os.makedirs(os.path.dirname(fullName), exist_ok=True)
       with open(fullName, 'w', newline='') as csvfile:
              writer = csv.writer(csvfile, delimiter=',')
              try:
                     writer.writerow([query] + [getStock(code, year, month, day)])
              except:
                     print('no stock data')
       os.makedirs(os.path.dirname(fullName), exist_ok=True)
       with open(fullName, 'w', newline='') as csvfile:
              writer = csv.writer(csvfile, delimiter=',')
              st = str(year) + '-' + str(month) + '-' + str(day)
              try:
                     writer.writerow([st] + [getStock(code, year, month, day)])
              except:
                     print('no stock data')


def makeURL(query, date, pageSize):
       return ('https://newsapi.org/v2/everything?'
              'q=' + query + '&'
              'from=' + date + '&'
              'sortBy=popularity&'
              'pageSize=' + pageSize + '&'
              'apiKey=13bd628fa8b548738d3b113d9442574e&'
              'language=en')

def getQuery(code):
       return get_symbol(code).split(',')[0]

def isWeekday(today):
       if today.weekday() >= 5:
              return False
       else:
              return True

def get_symbol(symbol):
       url = "http://d.yimg.com/autoc.finance.yahoo.com/autoc?query={}&region=1&lang=en".format(symbol)
       result = requests.get(url).json()
       for x in result['ResultSet']['Result']:
              if x['symbol'] == symbol:
                     return x['name']

def getStock(code, year, month, day):
       start = datetime(year, month, day)
       stock = get_historical_data(code, start, start, token='pk_3fc4f2751a6746f3b1cdc30763095572')
       if day >= 10:
              dict = stock[str(year) + '-' + str(month) + '-' + str(day)]
       else:
              dict = stock[str(year) + '-' + str(month) + '-' + '0' + str(day)]
       return dict['close'] - dict['open']

if __name__ == '__main__':
       for i in range(26, 32):
               obtainTrainingData('FB', 2019, 10, i)
       for i in range(1, 24):
              obtainTrainingData('FB', 2019, 11, i)



              #obtainTrainingData('TSLA', 2019, 11, i)



