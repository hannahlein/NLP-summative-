#!/usr/bin/env python
# coding: utf-8

# In[11]:


#get the dataset from Project Gutenberg through its URL 

from urllib import request
url = "https://www.gutenberg.org/cache/epub/1581/pg1581.txt"
response = request.urlopen(url)
raw = response.read().decode('utf8')


# In[12]:


#create datasets for the individual Gospels 

def get_book(chapter_start, chapter_end): 
    start_pos = raw.index(chapter_start)
    end_pos = raw.index(chapter_end)
    return raw[start_pos:end_pos]


#get Matthew 
HG_S_Matthew = get_book(
    "THE HOLY GOSPEL OF JESUS CHRIST ACCORDING TO SAINT MATTHEW", 
    "THE HOLY GOSPEL OF JESUS CHRIST ACCORDING TO ST. MARK"
    )

#get Mark 
HG_S_Mark = get_book(
    "THE HOLY GOSPEL OF JESUS CHRIST ACCORDING TO ST. MARK", 
    "THE HOLY GOSPEL OF JESUS CHRIST ACCORDING TO ST. LUKE"
    )


#get Luke 

HG_S_Luke = get_book(
    "THE HOLY GOSPEL OF JESUS CHRIST ACCORDING TO ST. LUKE", 
    "THE HOLY GOSPEL OF JESUS CHRIST ACCORDING TO ST. JOHN"
    )


#get John 

HG_S_John = get_book(
    "THE HOLY GOSPEL OF JESUS CHRIST ACCORDING TO ST. JOHN", 
    "THE ACTS OF THE APOSTLES"
    )


# In[40]:


import re
import string
import nltk
from nltk.corpus import stopwords

import spacy
import csv
get_ipython().system('python -m spacy download en_core_web_sm')


# In[14]:


#cleaning Matthew (punctuation etc.)

clean_Matthew = ' '.join([i for i in HG_S_Matthew.splitlines() if i != ''])

clean_Matthew = re.sub(r'\d+', '', clean_Matthew)

text = clean_Matthew 

clean_Matthew = ''.join([char for char in text if char not in string.punctuation])

#stop words 
nltk.download('stopwords')

stop_words = set(stopwords.words('english'))

text = clean_Matthew

words = nltk.word_tokenize(text)

filtered_words = [word for word in words if word.lower() not in stop_words]

filtered_Matthew = ' '.join(filtered_words)

print(filtered_Matthew)


# In[15]:


#cleaning Mark 

clean_Mark = ' '.join([i for i in HG_S_Mark.splitlines() if i != ''])

clean_Mark = re.sub(r'\d+', '', clean_Mark)

text = clean_Mark 

clean_Mark = ''.join([char for char in text if char not in string.punctuation])

#stop words 
nltk.download('stopwords')

stop_words = set(stopwords.words('english'))

text = clean_Mark 

words = nltk.word_tokenize(text)

filtered_words = [word for word in words if word.lower() not in stop_words]

filtered_Mark = ' '.join(filtered_words)

print(filtered_Mark)


# In[16]:


#cleaning Luke 

clean_Luke = ' '.join([i for i in HG_S_Luke.splitlines() if i != ''])

clean_Luke = re.sub(r'\d+', '', clean_Luke)

text = clean_Luke

clean_Luke = ''.join([char for char in text if char not in string.punctuation])

#stop words 
nltk.download('stopwords')

stop_words = set(stopwords.words('english'))

text = clean_Luke

words = nltk.word_tokenize(text)

filtered_words = [word for word in words if word.lower() not in stop_words]

filtered_Luke = ' '.join(filtered_words)

print(filtered_Luke)


# In[17]:


#cleaning John 

clean_John = ' '.join([i for i in HG_S_John.splitlines() if i != ''])

clean_John = re.sub(r'\d+', '', clean_John)

text = clean_John

clean_John = ''.join([char for char in text if char not in string.punctuation])

#stop words 
nltk.download('stopwords')

stop_words = set(stopwords.words('english'))

text = clean_John

words = nltk.word_tokenize(text)

filtered_words = [word for word in words if word.lower() not in stop_words]

filtered_John = ' '.join(filtered_words)

print(filtered_John)


# In[19]:


#sentimenta analysis Matthew 

import pandas as pd
get_ipython().system('pip install plotly')
import plotly.express as px
import nltk.sentiment.util as sentiment_util
from nltk.sentiment import SentimentIntensityAnalyzer

text = filtered_Matthew

#initialise sentiment analyzer
sentiment_analyzer = SentimentIntensityAnalyzer()

#initialise empty lists to store VAD scores for each word
valence_scores = []
arousal_scores = []
dominance_scores = []
words = []  

#loop through each word in text file and perform VAD analysis
for word in nltk.word_tokenize(text):
    scores = sentiment_analyzer.polarity_scores(word)
    valence = scores['pos'] - scores['neg']
    arousal = scores['pos'] + scores['neg'] - scores['neu']
    dominance = scores['pos'] + scores['neg']
    valence_scores.append(valence)
    arousal_scores.append(arousal)
    dominance_scores.append(dominance)
    words.append(word)  # Append each word to the list

#create dataframe with VAD scores and words for each word
data = {
    'valence': valence_scores,
    'arousal': arousal_scores,
    'dominance': dominance_scores,
    'words': words  # Add the new list of words
}
df = pd.DataFrame(data)

#create interactive 3D scatter plot using Plotly
fig = px.scatter_3d(df, x='valence', y='arousal', z='dominance', text='words', opacity=0.7, title='VAD Scores of Words in Text')

fig.update_layout(scene={'xaxis_title': 'Valence', 'yaxis_title': 'Arousal', 'zaxis_title': 'Dominance'})

fig.show()


# In[20]:


#sentimenta analysis Mark 

import plotly.express as px
import nltk.sentiment.util as sentiment_util
from nltk.sentiment import SentimentIntensityAnalyzer

text = filtered_Mark 

#initialise sentiment analyzer
sentiment_analyzer = SentimentIntensityAnalyzer()

#initialise empty lists to store VAD scores for each word
valence_scores = []
arousal_scores = []
dominance_scores = []
words = []  

#loop through each word in text file and perform VAD analysis
for word in nltk.word_tokenize(text):
    scores = sentiment_analyzer.polarity_scores(word)
    valence = scores['pos'] - scores['neg']
    arousal = scores['pos'] + scores['neg'] - scores['neu']
    dominance = scores['pos'] + scores['neg']
    valence_scores.append(valence)
    arousal_scores.append(arousal)
    dominance_scores.append(dominance)
    words.append(word)  # Append each word to the list

#create dataframe with VAD scores and words for each word
data = {
    'valence': valence_scores,
    'arousal': arousal_scores,
    'dominance': dominance_scores,
    'words': words  # Add the new list of words
}
df = pd.DataFrame(data)

#create interactive 3D scatter plot using Plotly
fig = px.scatter_3d(df, x='valence', y='arousal', z='dominance', text='words', opacity=0.7, title='VAD Scores of Words in Text')

fig.update_layout(scene={'xaxis_title': 'Valence', 'yaxis_title': 'Arousal', 'zaxis_title': 'Dominance'})

fig.show()


# In[21]:


#sentimenta analysis Luke 

import plotly.express as px
import nltk.sentiment.util as sentiment_util
from nltk.sentiment import SentimentIntensityAnalyzer

text = filtered_Luke 

#initialise sentiment analyzer
sentiment_analyzer = SentimentIntensityAnalyzer()

#initialise empty lists to store VAD scores for each word
valence_scores = []
arousal_scores = []
dominance_scores = []
words = []  

#loop through each word in text file and perform VAD analysis
for word in nltk.word_tokenize(text):
    scores = sentiment_analyzer.polarity_scores(word)
    valence = scores['pos'] - scores['neg']
    arousal = scores['pos'] + scores['neg'] - scores['neu']
    dominance = scores['pos'] + scores['neg']
    valence_scores.append(valence)
    arousal_scores.append(arousal)
    dominance_scores.append(dominance)
    words.append(word)  # Append each word to the list

#create dataframe with VAD scores and words for each word
data = {
    'valence': valence_scores,
    'arousal': arousal_scores,
    'dominance': dominance_scores,
    'words': words  # Add the new list of words
}
df = pd.DataFrame(data)

#create interactive 3D scatter plot using Plotly
fig = px.scatter_3d(df, x='valence', y='arousal', z='dominance', text='words', opacity=0.7, title='VAD Scores of Words in Text')

fig.update_layout(scene={'xaxis_title': 'Valence', 'yaxis_title': 'Arousal', 'zaxis_title': 'Dominance'})

fig.show()


# In[22]:


#sentimenta analysis John 

import plotly.express as px
import nltk.sentiment.util as sentiment_util
from nltk.sentiment import SentimentIntensityAnalyzer

text = filtered_John

#initialise sentiment analyzer
sentiment_analyzer = SentimentIntensityAnalyzer()

#initialise empty lists to store VAD scores for each word
valence_scores = []
arousal_scores = []
dominance_scores = []
words = []  

#loop through each word in text file and perform VAD analysis
for word in nltk.word_tokenize(text):
    scores = sentiment_analyzer.polarity_scores(word)
    valence = scores['pos'] - scores['neg']
    arousal = scores['pos'] + scores['neg'] - scores['neu']
    dominance = scores['pos'] + scores['neg']
    valence_scores.append(valence)
    arousal_scores.append(arousal)
    dominance_scores.append(dominance)
    words.append(word)  # Append each word to the list

#create dataframe with VAD scores and words for each word
data = {
    'valence': valence_scores,
    'arousal': arousal_scores,
    'dominance': dominance_scores,
    'words': words  # Add the new list of words
}
df = pd.DataFrame(data)

#create interactive 3D scatter plot using Plotly
fig = px.scatter_3d(df, x='valence', y='arousal', z='dominance', text='words', opacity=0.7, title='VAD Scores of Words in Text')

fig.update_layout(scene={'xaxis_title': 'Valence', 'yaxis_title': 'Arousal', 'zaxis_title': 'Dominance'})

fig.show()


# In[24]:


#get American Constitution text file 

with open('american constitution.txt', 'r') as file:
    contents = file.read()
    print(contents)


# In[25]:


#cleaning the data

clean_constitution = ' '.join([i for i in contents.splitlines() if i != ''])

clean_constitution = re.sub(r'\d+', '', clean_constitution)

text = clean_constitution

clean_constitution = ''.join([char for char in text if char not in string.punctuation])

#stop words 
nltk.download('stopwords')

stop_words = set(stopwords.words('english'))

text = clean_constitution 

words = nltk.word_tokenize(text)

filtered_words = [word for word in words if word.lower() not in stop_words]

filtered_constitution = ' '.join(filtered_words)

print(filtered_constitution)


# In[26]:


#make word cloud for consitution 
get_ipython().system('pip install wordcloud')
from wordcloud import WordCloud
import matplotlib.pyplot as plt

text = filtered_constitution

wordcloud = WordCloud(width=800, height=800, background_color='white', min_font_size=10).generate(text)

plt.figure(figsize=(8, 8), facecolor=None)
plt.imshow(wordcloud)
plt.axis("off")
plt.tight_layout(pad=0)

plt.show()


# In[38]:


#topic modelling for Constitution 

import gensim
from gensim.utils import simple_preprocess
from gensim.parsing.preprocessing import STOPWORDS
from gensim import corpora, models

with open('american constitution.txt', 'r') as f:
    text = f.read()

#tokenise and stop words 
def tokenize(text):
    return [token for token in simple_preprocess(text) if token not in STOPWORDS]

#create list of tokenised sentences
sentences = [tokenize(sentence) for sentence in text.split('.')]

#create dictionary from the sentences
dictionary = corpora.Dictionary(sentences)

#create corpus from the sentences
corpus = [dictionary.doc2bow(sentence) for sentence in sentences]

#train the LDA model
lda_model = models.LdaModel(corpus=corpus, num_topics=10, id2word=dictionary)

#print the top topics and words for each topic
for topic in lda_model.show_topics():
    print(topic)


# In[39]:


#topics identified (topic in first row and words in second row for each of the topics)

#Powers and limitations of the government
#congress, power, law, rights, states  

#Legal and judicial system
#court, trial, justice, jury 

#Amendments and rights
#amendments, rights, freedom 

#Executive branch and presidency: 
#president, executive, power, cabinet 

#Legislative branch and Congress  
#congress, bill, representatives 

#Federal government's role in military and defense
#national defense, standing army, war 

#Elections and voting
#elections, suffrage, voting 

#Federal taxes and revenue 
#taxes, revenue, congress, income, treasury 

#Regulation of commerce and trade
#commerce, trade, foreign, imports

#Judicial system and courts 
#judificary, judges, trial, law 


# In[27]:


#print 50 most common words in consitution 

from collections import Counter

words = filtered_constitution.lower().split()

word_counts = Counter(words)

top_words = word_counts.most_common(50)

for word, count in top_words:
    print(word, count)


# In[28]:


#print 50 most common words in Bible 

import pandas as pd
from collections import Counter

df = pd.read_csv('lemmas_dataset_21009210014.csv')

text = ' '.join(df.iloc[:,1].astype(str))

text = ''.join(c for c in text if c.isalnum() or c.isspace())

text = text.lower()

words = text.split()

word_freq = Counter(words)
top_50_words = word_freq.most_common(50)

print('50 most common words:')
for word, freq in top_50_words:
    print(f'{word}: {freq}')


# In[29]:


#generate bar plot of 50 most common words 

import pandas as pd
import string
from collections import Counter
import matplotlib.pyplot as plt

df = pd.read_csv('lemmas_dataset_21009210014.csv')

text = ' '.join(df.iloc[:, 1].astype(str))

text = ''.join(c for c in text if c.isalnum() or c.isspace())

word_counts = Counter(text.split())

top_words = word_counts.most_common(50)

plt.bar(range(50), [count for word, count in top_words])
plt.xticks(range(50), [word for word, count in top_words], rotation=90)
plt.show()


# In[30]:


#generate word cloud for Bible 

import pandas as pd
from wordcloud import WordCloud
import matplotlib.pyplot as plt

df = pd.read_csv('lemmas_dataset_21009210014.csv', header=None, names=['text'])

text = ' '.join(df['text'].astype(str))

wordcloud = WordCloud(width=800, height=800, background_color='white', min_font_size=10).generate(text)

plt.figure(figsize=(8, 8), facecolor=None)
plt.imshow(wordcloud)
plt.axis("off")
plt.tight_layout(pad=0)
plt.show()


# In[31]:


#compare the most common words in both texts 

with open('bible_top50.txt', 'r') as file1:
    file1_words = set(file1.read().split())

with open('constitution_top50.txt', 'r') as file2:
    file2_words = set(file2.read().split())

overlap = file1_words.intersection(file2_words)

print(overlap)


# In[41]:


#using NER to display all locations mentioned in Matthew

#load spaCy model with the NER component
nlp = spacy.load("en_core_web_sm")

#open the CSV file containing the lemmatised version of the Matthew Gospel
with open("lemmatised_Matthew.csv", newline="") as csvfile:
    reader = csv.reader(csvfile)
    next(reader) 
    text = " ".join(row[1] for row in reader)  

#process the text with spaCy to extract named entities
doc = nlp(text)
geographical_locations = set()
for ent in doc.ents:
    if ent.label_ == "GPE":
        geographical_locations.add(ent.text)

#print the extracted geographical locations
print(geographical_locations)


# In[34]:


#using NER to display all locations mentioned in Mark 

#load spaCy model with the NER component
nlp = spacy.load("en_core_web_sm")

#open the CSV file containing the lemmatised version of the Mark Gospel
with open("lemmatised_Mark.csv", newline="") as csvfile:
    reader = csv.reader(csvfile)
    next(reader)  
    text = " ".join(row[1] for row in reader) 

#process the text with spaCy to extract named entities
doc = nlp(text)
geographical_locations = set()
for ent in doc.ents:
    if ent.label_ == "GPE":
        geographical_locations.add(ent.text)

#print the extracted geographical locations
print(geographical_locations)


# In[35]:


#using NER to display all locations mentioned in Luke


#load spaCy model with the NER component
nlp = spacy.load("en_core_web_sm")

#open the CSV file containing the lemmatised version of the Luke Gospel
with open("lemmatised_Luke.csv", newline="") as csvfile:
    reader = csv.reader(csvfile)
    next(reader)  
    text = " ".join(row[1] for row in reader)  

#process the text with spaCy to extract named entities
doc = nlp(text)
geographical_locations = set()
for ent in doc.ents:
    if ent.label_ == "GPE":
        geographical_locations.add(ent.text)

#print the extracted geographical locations
print(geographical_locations)


# In[36]:


#using NER to display all locations mentioned in John

#load spaCy model with the NER component
nlp = spacy.load("en_core_web_sm")

#open the CSV file containing the lemmatised version of the John Gospel
with open("lemmatised_John.csv", newline="") as csvfile:
    reader = csv.reader(csvfile)
    next(reader)  
    text = " ".join(row[1] for row in reader) 

#process the text with spaCy to extract named entities
doc = nlp(text)
geographical_locations = set()
for ent in doc.ents:
    if ent.label_ == "GPE":
        geographical_locations.add(ent.text)

#print the extracted geographical locations
print(geographical_locations)


# In[37]:


#find overlaps between words in all four Gospels 

with open('locations_Matthew.txt', 'r') as f:
    locations_Matthew = set(f.read().split())

with open('locations_Mark.txt', 'r') as f:
    locations_Mark = set(f.read().split())

with open('locations_Luke.txt', 'r') as f:
    locations_Luke = set(f.read().split())

with open('locations_John.txt', 'r') as f:
    locations_John = set(f.read().split())

overlaps = locations_Matthew & locations_Mark & locations_Luke & locations_John

print(overlaps)


# In[ ]:




