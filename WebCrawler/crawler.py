from _hashlib import new

import requests
from bs4 import BeautifulSoup
import codecs
import pandas as pd
import csv as csv



all_the_data = []

def manipulate_set_of_links(link,counter):
    single_data = {}
    plain_text = get_plain_text(link)
    soup = BeautifulSoup(plain_text, "html.parser")
    #print(soup)
    soup_text = soup.find_all("p")
    """
    titles = soup.find_all("div",{"class":"title_container"})

    try :
        for title in titles:
            title_topic = title.contents[1].text
            title = title.contents[3].text
            #single_data['headline'] = title
            #single_data['headline_topic'] = title_topic
    except :
       print(link)
       print(" title topic (",title_topic,") title  (",title,")")
"""
    file_name = 'All Text Files/'+'file'+str(counter)+'.txt'
    #print(file_name)
    with codecs.open(file_name,'w','utf-8-sig') as temp:

        all_text = ''
        for st in soup_text:
            all_text += st.text
                #print(all_text)
        temp.write(all_text)
        temp.close()
    single_data['news'] = all_text
    single_data['sentiment'] = 0
    #print(single_data)
    return single_data


def get_plain_text(url):
    source_code = requests.get(url)
    plain_text = source_code.text
    return plain_text


def trade_spider(max_pages):
    page_cnt = 1
    news_count = 1
    counter = 1
    while page_cnt <= max_pages:

        url = 'http://www.prothom-alo.com/opinion/article?tags=60'+'&page='+str(page_cnt)
        plain_text = get_plain_text(url)
        ##print(url)
        soup = BeautifulSoup(plain_text,"html.parser")
        tag_name = 'div'
        class_name = 'each_news'

        set_of_links = set()

        for link in soup.find_all(tag_name,{'class',class_name}):

            text = link.div.prettify()
            soup_to_find_href = BeautifulSoup(text,"html.parser")
            for link2 in soup_to_find_href.find_all('a'):
                href = 'http://www.prothom-alo.com/opinion/' + link2.get('href')
                if 'comments' not in href:
                    set_of_links.add(href)

        page_cnt += 1
        for link in set_of_links:
            news_count += 1
            if((news_count%100) ==0) :
                print(" news ",news_count)
            print(counter)
            single_data = manipulate_set_of_links(link, counter)
            #print(single_data)
            all_the_data.append(single_data)
            counter += 1
            if(counter > 200 or len(all_the_data) > 200):
                page_cnt = 7
                break
            #print(link)
        df = pd.DataFrame(all_the_data)
        train = df.to_csv(path_or_buf="All Data Files/LabeledTrainData.tsv",index = len(link), mode='w', sep="\t",
                                encoding='utf-8',columns = ['sentiment','news'])


#print(train.columns.values)
trade_spider(6)
#manipulate_set_of_links("http://www.prothom-alo.com/opinion/article/990052/%E0%A6%B6%E0%A6%BF%E0%A6%B2%E0%A7%8D%E0%A6%AA%E0%A7%87%E0%A6%B0-%E0%A6%B8%E0%A6%82%E0%A6%B8%E0%A6%BE%E0%A6%B0%E0%A7%87%E0%A6%B0-%E0%A6%8F%E0%A6%95-%E0%A6%AE%E0%A6%B9%E0%A6%BE%E0%A6%AC%E0%A7%80%E0%A6%B0",1)