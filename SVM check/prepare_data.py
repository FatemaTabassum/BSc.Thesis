from bs4 import BeautifulSoup
import numpy as np
import re
import pandas as pd
from sklearn import svm
from sklearn.feature_extraction.text import CountVectorizer
import codecs
from sklearn.ensemble import RandomForestClassifier


train = pd.read_csv("data/LabeledTrainData.csv", header=0, delimiter=",",encoding = 'utf-8')
test = pd.read_csv("data/TestData.tsv", header=0, delimiter="\t",encoding = 'utf-8',quoting = 3)
file_name  = 'data/StopWords.txt'
with codecs.open(file_name,'r','utf-8') as inp:
    stop_words = [x[:-1] for x in inp.readlines()]

def tokenize(text):
    REGEX = re.compile(" ")
    sentence = text
    words = (REGEX.split(sentence))
    return  words

def news_to_words(raw_news):
    example1 = BeautifulSoup(raw_news,'html.parser')
    letters_only = re.sub("[^\u0980-\u09FF]"," ",example1.get_text())
    letters_only = re.sub("[\u09E6-\u09EF]+| ৷"," ",letters_only)
    words = letters_only.split()
    meaningful_words = [w for w in words if not w in stop_words]
    return (" ".join(meaningful_words))



def clean_train_data():
    num_news = len(train["news"])
    st = ''
    for i in range(0,num_news):
        if ((i + 1) % 3 == 0):
            #print("news %d of %d\n" % (i + 1, num_news))
            st = news_to_words(train["news"][i])
        clean_train_news.append(news_to_words(train["news"][i]))



def clean_test_data():
    num_news = len(test["news"])
    print("len of test news is ",num_news)
    st = ''
    for i in range(0, num_news):
        if ((i + 1) % 2 == 0):
            # print("news %d of %d\n" % (i + 1, num_news))
            st = news_to_words(test["news"][i])
        clean_test_news.append(news_to_words(test["news"][i]))

clean_train_news = []
clean_test_news = []

clean_train_data()
vectorizer = CountVectorizer( analyzer='word', tokenizer = lambda x : tokenize(x),  lowercase=False, ngram_range=(1,1) )
print(vectorizer)
feature_train_array = vectorizer.fit_transform(clean_train_news)
feature_train_array = feature_train_array.toarray()
feature_names = vectorizer.get_feature_names()

#print(test["news"])
clean_test_data()
feature_test_array = vectorizer.transform(clean_test_news)
feature_test_array = feature_test_array.toarray()
print(feature_test_array)



vocab = vectorizer.get_feature_names()
dist = np.sum(feature_test_array,axis = 0)
#for tag,count in zip(vocab,dist):
#    print(tag,count)
forest = RandomForestClassifier(n_estimators=100)
forest = forest.fit(feature_train_array,train["sentiment"])
result = forest.predict(feature_test_array)
ids = []
for i in range(0,len(test["news"])):
    ids.append(i)
test_id = pd.DataFrame()
test_id["id"] = ids
#for i in range(0,len(ids)):
#    print(ids[i])
output = pd.DataFrame(data = {"id":test_id["id"],"sentiment":result})
output.to_csv("result/Bag_of_Words_model.csv",index = False,quoting=3)




