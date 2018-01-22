
# coding: utf-8

# ## Text Classification and Clustering using kNN and K-means

# In[109]:

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
import string
import re
import numpy as np
from collections import Counter


# In[120]:

stop = set(stopwords.words('english'))
exclude = set(string.punctuation) 
lemma = WordNetLemmatizer()


# ### Cleaning text sentences - removing punctuation,stop words, digits

# In[121]:

def clean(doc):
    stop_free = " ".join([i for i in doc.lower().split() if i not in stop])
    punc_free = ''.join(ch for ch in stop_free if ch not in exclude)
    normalized = " ".join(lemma.lemmatize(word) for word in punc_free.split())
    processed = re.sub(r"\d+","",normalized)
    y = processed.split()
    return y


# In[122]:

#Testing clean()
print(clean("This is 23 2E#~~@!@ amazing! I love it"))


# In[123]:

print("There are 10 sentences of following three classes : \n1)Cricket\n2)Artificial Intelligence\n3)Chemistry")
path = r".\Sentences.txt"


# In[161]:

train_clean_sentences = []
file = open(path,'r')
for line in file:
    line = line.strip()
    cleaned = clean(line)
    cleaned = ' '.join(cleaned)
    train_clean_sentences.append(cleaned)
# print(train_clean_sentences)


# In[162]:

vectorizer = TfidfVectorizer(stop_words='english')
X = vectorizer.fit_transform(train_clean_sentences)


# In[163]:

# Creating labels for training sentences
# 0 - , 1 - , 2 - 
y_train = np.zeros(30)
y_train[10:20] = 1
y_train[20:30] = 2


# ### Training kNN and k-means Models

# In[175]:

kNN_model = KNeighborsClassifier(n_neighbors=5)
kNN_model.fit(X,y_train)

# Clustering the training 30 sentences with K-means technique
kmeans_model = KMeans(n_clusters=3, init='k-means++', max_iter=200, n_init=100)
kmeans_model.fit(X)


# ### Testing on New Data

# In[179]:

test_sentences = ["Deep Learning and Big Data is the future.",                  "Sachin was a great batsman",                  "Machine learning is a area of Artificial intelligence"                 ]

test_clean_sentences = []
for test in test_sentences:
    cleaned_test = clean(test)
    cleaned = ' '.join(cleaned_test)
    cleaned = re.sub(r"\d+","",cleaned)
    test_clean_sentences.append(cleaned)
    
Test = vectorizer.transform(test_clean_sentences) 

true_test_labels = ['Cricket','AI','Chemistry']
predicted_labels_knn = kNN_model.predict(Test)
predicted_labels_kmeans = kmeans_model.predict(Test)

print("\nSentences to be predicted:\n1. ",        test_sentences[0],"\n2. ",test_sentences[1],"\n3. ",test_sentences[2])
print("\n-------------------------------PREDICTIONS BY KNN------------------------------------------")
print("\n",test_sentences[0],":",true_test_labels[np.int(predicted_labels_knn[0])],        "\n",test_sentences[1],":",true_test_labels[np.int(predicted_labels_knn[1])],        "\n",test_sentences[2],":",true_test_labels[np.int(predicted_labels_knn[2])])

print("\n-------------------------------PREDICTIONS BY K-Means--------------------------------------")
print("\nCricket : ",Counter(kmeans_model.labels_[0:10]).most_common(1)[0][0])
print("Artificial Intelligence : ",Counter(kmeans_model.labels_[10:20]).most_common(1)[0][0])
print("Chemistry : ",Counter(kmeans_model.labels_[20:30]).most_common(1)[0][0])

print("\n",test_sentences[0],":",predicted_labels_kmeans[0],        "\n",test_sentences[1],":",predicted_labels_kmeans[1],        "\n",test_sentences[2],":",predicted_labels_kmeans[2])

