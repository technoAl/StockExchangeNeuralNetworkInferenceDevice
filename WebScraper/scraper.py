import requests
from bs4 import BeautifulSoup
from newsapi import NewsApiClient

import requests
url = ('https://newsapi.org/v2/everything?'
       'q=Apple&'
       'from=2019-10-31&'
       'sortBy=popularity&'
       'apiKey=13bd628fa8b548738d3b113d9442574e')

response = requests.get(url)

print(response.json())
# page = requests.get("https://google.com/search?q=facebook+news" + "&as_qdr=y15")
# if page.status_code == 200:
#     soup = BeautifulSoup(page.content, "lxml")
#     print(soup.prettify())
#     #print(list(soup.children))
#     all_tags = soup.find_all('p')
#     for i in all_tags:
#         print(i.get_text())gi
# else:
#     print("unfortunate failure")