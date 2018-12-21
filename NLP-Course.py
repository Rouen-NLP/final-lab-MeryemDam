# -*- coding: utf-8 -*-
"""
Created on Thu Nov  8 16:11:30 2018

@author: Meryem
"""

#%% packages
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn import preprocessing
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import linear_model
from sklearn import metrics
from sklearn.neural_network import MLPClassifier

#%%

data= pd.read_csv('/data/Tobacco3482.csv')
#print(data.head())  #to see and example of the data we are working with
sns.countplot(data=data, y='label')   #to visualize the frequency of each type of data we have


#%% Importing the data
path = '/data/Tobacco3482-OCR'
dirs_classes = os.listdir( path )
for file in dirs_classes:
   print (file)
read=[]
dirs_contenus=[]
l=[]

for i in range(10):
    dirs_contenus.append(os.listdir('/data/Tobacco3482-OCR/%s' %dirs_classes[i]))
    for j in range(len(dirs_contenus[i])):
       op=open('/data/Tobacco3482-OCR/%s/%s' %(dirs_classes[i],dirs_contenus[i][j]), encoding="utf8")
       read.append(op.read())
       l.append(dirs_classes[i])
       
#%%create a dataframe using texts and lables
trainDF = pd.DataFrame()
trainDF['text'] = read
trainDF['label'] =  l

#%%splitting the data
     
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(trainDF['text'], trainDF['label'], test_size=0.2, random_state=1)

X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=1)

print('The training set has {} elements, and the test set has {} elements and the validation set has {} elements'.format(len(X_train),len(X_test), len(X_val)))

#%% label encode the target labels 
encoder = preprocessing.LabelEncoder()

y_train = encoder.fit_transform(y_train)
y_test = encoder.fit_transform(y_test)
y_val = encoder.fit_transform(y_val)

#%% Feature Engineering Step

#%%create a count vectorizer object 
count_vect = CountVectorizer(analyzer='word', token_pattern=r'\w{1,}')
count_vect.fit(trainDF['text'])

# transform the training and validation data using count vectorizer object
xtrain_count =  count_vect.transform(X_train)
xtest_count =  count_vect.transform(X_test)
xval_count = count_vect.transform(X_val)
#%% TF-IDF
# word level tf-idf
tfidf_vect = TfidfVectorizer(analyzer='word', token_pattern=r'\w{1,}', max_features=5000)
tfidf_vect.fit(trainDF['text'])
xtrain_tfidf =  tfidf_vect.transform(X_train)
xtest_tfidf =  tfidf_vect.transform(X_test)
xval_tfidf = tfidf_vect.transform(X_val)


# ngram level tf-idf 
tfidf_vect_ngram = TfidfVectorizer(analyzer='word', token_pattern=r'\w{1,}', ngram_range=(2,3), max_features=5000)
tfidf_vect_ngram.fit(trainDF['text'])
xtrain_tfidf_ngram =  tfidf_vect_ngram.transform(X_train)
xval_tfidf_ngram =  tfidf_vect_ngram.transform(X_val)
xtest_tfidf_ngram =  tfidf_vect_ngram.transform(X_test)

# characters level tf-idf
tfidf_vect_ngram_chars = TfidfVectorizer(analyzer='char', token_pattern=r'\w{1,}', ngram_range=(2,3), max_features=5000)
tfidf_vect_ngram_chars.fit(trainDF['text'])
xtrain_tfidf_ngram_chars =  tfidf_vect_ngram_chars.transform(X_train) 
xval_tfidf_ngram_chars =  tfidf_vect_ngram_chars.transform(X_val) 
xtest_tfidf_ngram_chars =  tfidf_vect_ngram_chars.transform(X_test)

#%% Model Building
def train_model(classifier, feature_vector_train, label, feature_vector_valid, is_neural_net=False):
    # fit the training dataset on the classifier
    classifier.fit(feature_vector_train, label)
    
    # predict the labels on validation dataset
    predictions = classifier.predict(feature_vector_valid)
    
    if is_neural_net:
        predictions = predictions.argmax(axis=-1)
    
    return metrics.accuracy_score(predictions, y_val)
#%% Linear Model
    
# Linear Classifier on Count Vectors
accuracy_linear_count = train_model(linear_model.LogisticRegression(), xtrain_count, y_train, xval_count)
print ("LR, Count Vectors: ", accuracy)
#%%
# Linear Classifier on Word Level TF IDF Vectors
accuracy_linear_word = train_model(linear_model.LogisticRegression(), xtrain_tfidf, y_train, xval_tfidf)
print ("LR, WordLevel TF-IDF: ", accuracy_linear_word)
#%%

# Linear Classifier on Ngram Level TF IDF Vectors
accuracy_linear_ngram = train_model(linear_model.LogisticRegression(), xtrain_tfidf_ngram, y_train, xval_tfidf_ngram)
print ("LR, N-Gram Vectors: ", accuracy_linear_ngram)

#%%

# Linear Classifier on Character Level TF IDF Vectors
accuracy_linear_character = train_model(linear_model.LogisticRegression(), xtrain_tfidf_ngram_chars, y_train, xval_tfidf_ngram_chars)
print ("LR, CharLevel Vectors: ", accuracy_linear_character)


#%% Naive Bayes
#On Count Vectors

accuracy_nb_count = train_model(MultinomialNB(), xtrain_count, y_train, xval_count)
print ("NB, Count Vectors: ", accuracy_nb_count)

#%%
# NB on Word Level TF IDF Vectors
accuracy_nb_word = train_model(MultinomialNB(), xtrain_tfidf, y_train, xval_tfidf)
print ("NB, WordLevel TF-IDF: ", accuracy_nb_word)

#%%
# NB on Ngram Level TF IDF Vectors
accuracy_nb_ngram = train_model(MultinomialNB(), xtrain_tfidf_ngram, y_train, xval_tfidf_ngram)
print ("NB, N-Gram Vectors: ", accuracy_nb_ngram)

#%%
# NB on Character Level TF IDF Vectors
accuracy_nb_character = train_model(MultinomialNB(), xtrain_tfidf_ngram_chars, y_train, xval_tfidf_ngram_chars)
print ("NB, CharLevel Vectors: ", accuracy_nb_character)
        



#%% MLP Classifier


accuracy_mlp_count = train_model(MLPClassifier(activation='relu', alpha=1e-2, verbose=2, batch_size=50), xtrain_count, y_train, xval_count)
print ("MLP, Count Vectors: ", accuracy_mlp_count)
#%%
accuracy_mlp_word = train_model(MLPClassifier(activation='relu', alpha=1e-4, verbose=2, batch_size=50), xtrain_tfidf, y_train, xval_tfidf)
print ("NB, WordLevel TF-IDF: ", accuracy_mlp_word)

#%%
accuracy_mlp_ngram = train_model(MLPClassifier(activation='relu', alpha=1e-4, verbose=2, batch_size=50), xtrain_tfidf_ngram, y_train, xval_tfidf_ngram)
print ("NB, N-Gram Vectors: ", accuracy_mlp_ngram)

#%%
accuracy_mlp_character = train_model(MLPClassifier(activation='relu', alpha=1e-4, verbose=2, batch_size=50), xtrain_tfidf_ngram_chars, y_train, xval_tfidf_ngram_chars)
print ("NB, CharLevel Vectors: ", accuracy_mlp_character)
        

        
