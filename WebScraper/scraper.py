import requests
import json
from bs4 import BeautifulSoup
from newsapi import NewsApiClient


import requests

query = 'tesla'
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
for i in data:
       tmpDict = dict(i)
       file = open(query + str(count) + '.txt', 'w+')
       page = requests.get(tmpDict['url'])
       if page.status_code == 200:
              soup = BeautifulSoup(page.content, "lxml")
              all_tags = soup.find_all('p')
              for i in all_tags:
                     file.write(i.get_text())
       else:
              print("unfortunate failure")
       file.close()
       count+=1
