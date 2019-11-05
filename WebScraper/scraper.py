import requests
import json
from bs4 import BeautifulSoup
from newsapi import NewsApiClient
import os
import requests

query = 'Tesla'
url = ('https://newsapi.org/v2/everything?'
       'q=' + query + '&'
       'from=2019-10-31&'
       'sortBy=popularity&'
       'apiKey=13bd628fa8b548738d3b113d9442574e&'
       'pageSize=40&'
       'language=en')

response = requests.get(url)
data = response.json()
print(data['totalResults'])
data = data['articles']
print(len(data))
count = 0
path = '../' + query + 'TrainingData' + '/'
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
