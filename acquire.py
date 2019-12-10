from selenium import webdriver
from selenium.webdriver.firefox.options import Options
import requests
from bs4 import BeautifulSoup as bs
import re
def acquire():
    url = requests.get(html)
    return bs(url.content, 'html.parser')
def webcrawler(soup, html = 'https://github.com/trending'):
    index = {'languages' : [], 'links' : [], 'read_link'}
    title = soup.findAll("span", itemprop="programmingLanguage")
    print(title)
    for t in title:
        index['languages'].append(t.get_text())
        print(t.get_text(strip = True))
    a = soup.find_all('a', href = True)
    a = [str(aa) for aa in a if 'text-normal' in str(aa) and \
         """lh-condensed-ultra d-block""" not in str(aa)]
    for aa in a:
        try:
            aa = re.sub('"', '', aa[aa.index('=') + 1:aa.index('>')])
            print(aa)
            index['links'].append(aa)
        except:
            continue
        print('-------------------')
    return index
prep(acquire())