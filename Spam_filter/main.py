# -*- coding: utf-8 -*
import nltk
from nltk import word_tokenize,WordNetLemmatizer
from nltk.corpus import stopwords
from collections import Counter
from nltk import NaiveBayesClassifier,classify
import os
import string
import random
# import sys
# reload(sys)
# sys.setdefaultencoding("ISO-8859-1")
# import sys  

# reload(sys)  
# sys.setdefaultencoding('utf8')

stoplist = stopwords.words("english")

def init_lists(folder):
	a_list = []
	file_list = os.listdir(folder)
	for a_file in file_list:
		f = open(folder+a_file,'rb')
		a_list.append(f.read())
	f.close()
	return a_list

lemmatizer = WordNetLemmatizer()

def preprocess(sentence):
    lemmatizer = WordNetLemmatizer()
    return [lemmatizer.lemmatize(word.lower()) for word in word_tokenize(unicode(sentence, errors='ignore'))]

def get_features(text, setting):
    if setting=='bow':
        return {word: count for word, count in Counter(preprocess(text)).items() if not word in stoplist}
    else:
        return {word: True for word in preprocess(text) if not word in stoplist}

spam = init_lists("enron1/spam/") #All Spam emails
ham  = init_lists("enron1/ham/") #All ham emails

spam_emails = [(email,"spam") for email in spam]
ham_emails  = [(email,"ham") for email in ham]
all_emails = spam_emails + ham_emails

			# print(y)
# all_emails.encode("ascii",errors="ignore")
#Size of Dataset
print("Total no. of emails : ",len(all_emails))
print("Spam emails : ",len(spam_emails))
print("Ham emails : ",len(ham_emails))

#Randomly Shuffling the dataset

random.shuffle(all_emails)

#Preprocessing the data

# print(preprocess("THis is 34 newro324nt2#R#$$# awesome bro!!! :)"))

all_features = [(get_features(email,"bow"),label) for (email,label) in all_emails]
# print(all_features)



#Training Classifier

def train(features,samples_proportion):
	train_size = int(len(features)*samples_proportion)
	train_set, test_set = features[:train_size],features[train_size:]

	print("Training size : ",len(train_set))
	print("Test size : ",len(test_set))
	classifier = NaiveBayesClassifier.train(train_set)
	return train_set,test_set,classifier

train_set,test_set,classifier = train(all_features,0.8)

def evaluate(train_set, test_set, classifier):
    print ('Accuracy on the training set = ' + str(classify.accuracy(classifier, train_set)))
    print ('Accuracy of the test set = ' + str(classify.accuracy(classifier, test_set)))
    classifier.show_most_informative_features(20)



print(evaluate(train_set, test_set, classifier))