from selenium import webdriver
from selenium.webdriver.firefox.options import Options
import requests
from bs4 import BeautifulSoup as bs
import re
import pandas as pd
from selenium import webdriver
from selenium.webdriver.firefox.options import Options
import requests
from bs4 import BeautifulSoup as bs
import re
import pandas as pd
def acquire(html = 'https://github.com/trending?since=daily'):
    url = requests.get(html)
    return bs(url.content, 'html.parser')
def cleanhtml(raw_html):
  cleanr = re.compile(r'<.*?>')
  cleantext = re.sub(cleanr, '', r'%s' % raw_html)
  return cleantext
def webcrawler_repos(s):
    index = {'language' : [], 'link' : []}
    box_row = s.findAll('li')
    for soup in box_row:
        title = soup.findAll("span", itemprop="programmingLanguage")
        if len(title) == 0:
            continue
        for t in title:
            index['language'].append(t.get_text(strip = True))
        a = soup.find_all(class_="v-align-middle")
        for aa in a:
            try:
                aa = cleanhtml(aa)
                index['link'].append(aa)
            except:
                continue
    return pd.DataFrame(index)
def webscraper_readme(data):
    index = {'link' : [], 'readme' : []}
    to_pull = ['p', 'h1', 'h2', 'h3']
    for l in data['link']:
        index['link'].append(l)
        try:
            url = requests.get('https://www.github.com/' + l + '/blob/master/README.md')
        except:
            continue
        soup = bs(url.content, 'html.parser')
        html_list = []
        for to in to_pull:
            sp = soup.find_all(to)
            for s in sp:
                html_list.append(cleanhtml(s))
        index['readme'].append(html_list)
    return pd.DataFrame(index)
        
        
    
def star_pages(pages = 10):
    data = pd.DataFrame()
    for i in range(95, pages):
        print('ITERATION ' + str(i))
        url = r'https://github.com/search?p=' + str(i) + '&q=stars%3A%3E0&s=stars&type=Repositories'
        try:
            subset_one = webcrawler_repos(acquire(html = url))
            subset_two = webscraper_readme(subset_one)
            print(subset_one)
            print(subset_two)
            subset = subset_one.merge(subset_two, on = 'link', sort = True)
            print(subset)
            data = data.append(subset, sort = True)
            print('!')
            data = data.reset_index()
        except:
            print('ERROR')
    data.to_csv('new_hola.csv')
    return data