import requests
from bs4 import BeautifulSoup

page = requests.get("https://google.com/search?q=facebook+news" + "&as_qdr=y15")
if page.status_code == 200:
    soup = BeautifulSoup(page.content, "lxml")
    print(soup.prettify())
    #print(list(soup.children))
    all_tags = soup.find_all('p')
    for i in all_tags:
        print(i.get_text())
else:
    print("unfortunate failure")