import csv as csv
from tkinter.tix import _dummyText

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from bs4 import BeautifulSoup
import re
import nltk
from nltk.corpus import stopwords
from sklearn.ensemble import RandomForestClassifier
import codecs
from numpy import unicode
import unicodedata
import chardet
from nltk.tokenize import PunktSentenceTokenizer
from sklearn import svm

def object_to_number(freq):
    num_st = str(freq)
    #numbers_only = re.sub("[^0-9]", " ", num_st.get_text())
    nums = num_st.split()
    num_ar = []
    for i in nums:
        num_ar.append(int(i))
    #print("  (  ",num_ar,"  )  ")
    return num_ar



train = pd.read_csv("data/datasets.csv", header=0, delimiter=",",encoding = 'utf-8')
test = pd.read_csv("data/Testdatasets.csv", header=0, delimiter=",",encoding = 'utf-8')

print(train.columns.values)
print(train.shape)
X = []
result_Y = train["Result"]
Y = []
for y in result_Y:
    Y.append(int(y))
sz = len(train["Data"])
for i in range(0,sz):
    number_list = object_to_number(train["Data"][i]);
    if(i==sz-1):
        k = 18000 - len(number_list)
        for k in range(len(number_list),18000):
            number_list.append(0)
    X.append(number_list)

clf = svm.SVC(kernel = 'linear',gamma=0.1)
clf.fit(X,Y)
from sklearn.naive_bayes import MultinomialNB
clf1 = MultinomialNB()
clf1.fit(X, Y)
clf2 = svm.LinearSVC()
clf2.fit_transform(X,Y)
sz = len(test["Data"])
test_data = []


for i in range(0,sz):
    test_number_list = object_to_number(test["Data"][i]);
    if(i==sz-1):
        k = 18000 - len(test_number_list)
        for k in range(len(test_number_list),18000):
            test_number_list.append(0)
    test_data.append(test_number_list)

ans = []
for i in range(0,len(test_data)):
    temp = np.array(test_data[i]).reshape((1, -1))
    res = clf.predict(temp)
    res1 = clf1.predict(temp)
    res2 = clf2.predict(temp)
    ans.append(res)
    print(i," SVM  ======>  ",res,"  NB =======> ",res1,"  MaxEnt ====> ",res2)


######################## svm finished ################
testActualResult =[1,-1,-1,-1,-1,-1,1,-1,1,1,1, 1,-1,-1,-1,1,-1,1,-1,-1,1,-1,-1,1,1,-1,1,-1,1,1,-1,-1,1,-1,1,-1,-1,-1,-1,1,1,-1,-1,1,1,-1,-1,1,-1,1,-1,-1,-1,-1,-1,1,1,-1,1,-1,1,
                   1,-1,-1]
print(len(testActualResult))
"""
vectorizer = CountVectorizer(tokenizer=lambda doc: doc, lowercase=False)
train_vect = vectorizer.fit_transform(X)
train_vect = train_vect.toarray()

vocab = vectorizer.get_feature_names()
print(" ===============>>> ")
print(vocab)
dist = np.sum(train_vect,axis = 0)
for tag,count in zip(vocab,dist):
    print(tag,count)

test_vect =  vectorizer.transform(test_data)
test_vect = test_vect.toarray()
forest = RandomForestClassifier(n_estimators=100)
forest = forest.fit(train_vect,Y)
result = forest.predict(test_vect)

for i in range(0,len(result)):
    print(i," ======> ",result[i])
#output = pd.DataFrame(data = {"id":test["id"],"sentiment":result})
#output.to_csv("Bag_of_Words_model.csv",index = False,quoting=3)

"""
"""
from sklearn.naive_bayes import GaussianNB
clf1 = GaussianNB()
clf1.fit(X, Y)
GaussianNB(priors=None)
count = 0
for i in range(0,len(test_data)):
    temp = np.array(test_data[i]).reshape((1, -1))
    res = clf1.predict(temp)
    print(i,"  ======>  ",res," ",ans[i])
    if res == ans[i]:
        count = count + 1
print(count,"  ",119-count)
"""




count_nb = 0
count_svm = 0
count_linear_svm = 0
count_res_pos_svm = 0
count_res_neg_svm = 0
count_res_pos_nb = 0
count_res_neg_nb = 0
count_res_pos_mxent = 0
count_res_neg_mxent = 0


count_fp_pos_svm = 0
count_fp_neg_svm = 0
count_fp_pos_nb = 0
count_fp_neg_nb = 0
count_fp_pos_maxent = 0
count_fp_neg_maxent = 0

count_tp_pos_svm = 0
count_tp_neg_svm = 0
count_tp_pos_nb = 0
count_tp_neg_nb = 0
count_tp_pos_maxent = 0
count_tp_neg_maxent = 0



for i in range(0,len(testActualResult)):
    temp = np.array(test_data[i]).reshape((1, -1))
    res_nb = clf1.predict(temp)
    res_svm = clf.predict(temp)
    res_linear_svm = clf2.predict(temp)
    """
    if res_svm == 1:
        count_res_pos_svm += 1
    else :
        count_res_neg_svm += 1

    if res_nb == 1:
        count_res_pos_nb += 1
    else :
        count_res_neg_nb += 1

    if res_linear_svm == 1:
        count_res_pos_mxent += 1
    else :
        count_res_neg_mxent += 1
        """
   # print(i,"  ======>  ",res," ",ans[i])




    if res_nb == testActualResult[i]:
        if res_nb == 1:
            count_tp_pos_nb += 1
        else :
            count_tp_neg_nb += 1
    else :
        if res_nb == 1:
            count_fp_pos_nb += 1
        else :
            count_fp_neg_nb += 1
        count_nb = count_nb + 1



    if res_svm == testActualResult[i]:
        if res_svm == 1:
            count_tp_pos_svm += 1
        else :
            count_tp_neg_svm += 1
    else :
        if res_svm == 1:
            count_fp_pos_svm += 1
        else :
            count_fp_neg_svm += 1
        count_svm = count_svm + 1




    if res_linear_svm == testActualResult[i]:
        if res_linear_svm == 1:
            count_tp_pos_maxent += 1
        else :
            count_tp_neg_maxent += 1
    else :
        if res_linear_svm == 1:
            count_fp_pos_maxent += 1
        else :
            count_fp_neg_maxent += 1
        count_linear_svm = count_linear_svm + 1



print("  naiive bayes  milse ",count_nb," milenai ",len(testActualResult)-count_nb)
print("accuracy ",((count_nb*100)/(len(testActualResult))),"%  ","CountOfYesOfMachine ",count_res_pos_nb," CountOfNoOfMachine ",count_res_pos_nb)
print(" true_pos ",count_tp_pos_nb," true neg ",count_tp_neg_nb," false positive ",count_fp_pos_nb," false neg ",count_fp_neg_nb)
print("  svm  ",count_svm,"  ",len(testActualResult)-count_svm)
print("accuracy ",((count_svm*100)/(len(testActualResult))),"% ","CountOfYesOfMachine ",count_res_pos_svm," CountOfNoOfMachine ",count_res_pos_svm)
print(" true_pos ",count_tp_pos_svm," true neg ",count_tp_neg_svm," false positive ",count_fp_pos_svm," false neg ",count_fp_neg_svm)
print("  Linear svm  ",count_linear_svm,"  ",len(testActualResult)-count_linear_svm)
print("accuracy ",((count_linear_svm*100)/(len(testActualResult))),"% ","CountOfYesOfMachine ",count_res_pos_mxent," CountOfNoOfMachine ",count_res_pos_mxent)
print(" true_pos ",count_tp_pos_maxent," true neg ",count_tp_neg_maxent," false positive ",count_fp_pos_maxent," false neg ",count_fp_neg_maxent)