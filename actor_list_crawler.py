# https://www.imdb.com/list/ls058011111/?sort=list_order,asc&mode=detail&page=1
from bs4 import BeautifulSoup
from urllib.request import urlopen

url = "https://www.imdb.com/list/ls058011111/?sort=list_order,asc&mode=detail&page="

with open("top_1k_actors.txt", "a") as output:
    for count in range(1,11):
        page_url = url +str(count)
        print("fetching page "+str(count)+"...")
        data = urlopen(page_url).read()
        page = BeautifulSoup(data, 'html.parser')

        rows = page.findAll("h3", {'class': ['lister-item-header']})

        for a in rows:
            output.write(a.find('a').getText().lstrip())
