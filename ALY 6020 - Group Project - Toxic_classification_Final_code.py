#################################################################################
#######################Part 1 - EDA & Text Mining###############################
##################################################################################

#!/usr/bin/env python
# coding: utf-8

# In[4]:
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
import nltk
import os
import nltk.corpus
from nltk.tokenize import word_tokenize


# In[ ]:
import nltk
nltk.download('punkt')
nltk.download('stopwords')
from nltk import word_tokenize,sent_tokenize

# In[328]:
data = pd.read_csv("C:\\Users\shwet\Documents\shweta\Winter_2020\A\Predictive\Group_Project/train.csv")
data.head(10)


# In[313]:
data.isnull().sum().sum()


# In[314]:
total_rows = data.shape[0]
total_columns = data.shape[1]
data.isnull().sum()
print('Total number of rows : '+str(total_rows))
print('Total number of columns : '+str(total_columns))
print('Missing va lues in the Dataset :' +str(data.isnull().sum().sum()))


# In[305]:
data.describe(include='all')
data['toxic'].unique()


# In[ 306]:
# calculating the count of words in the comments
data['word_count'] = data['comment_text'].apply(lambda x: len(str(x).split(" ")))

# In[332]:

df = data.iloc[ :,2:9]
corrMatrix=df.corr()
corrMatrix

# In[335]:

plt.figure(figsize=(7, 7))
sns.heatmap(corrMatrix, annot=True)

# In[330]:


data['Clean'] =10 # creating the column
data.loc[(data['severe_toxic'] ==0 ) &(data['threat']==0) & (data['identity_hate']==0)
       & (data['insult']==0)& (data['obscene']==0) & (data['toxic']==0) ,
       'Clean'] = 1
x=data.iloc[:,2:].sum()
sns.barplot(x.index,x.values)
plt.title("Category Distribution", fontsize=20)


# In[7]:
pip install wordcloud


# In[337]:
from PIL import Image
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator


# In[338]:
# filtering the comment based on toxicity levels
clean_comment = " ".join(review for review in data[data['Clean']==1]['comment_text'])

negative_comment = " ".join(review for review in data[(data['severe_toxic'] ==1 ) | (data['threat']==1) |(data['identity_hate']==1)
       | (data['insult']==1)|(data['obscene']==1) | (data['toxic']==1)]['comment_text'])

severe_toxic_comment = " ".join(review for review in data[(data['severe_toxic'] ==1 )]['comment_text'])

threat_comment = " ".join(review for review in data[(data['threat']==1)]['comment_text'])

identity_hate_comment = " ".join(review for review in data[(data['identity_hate']==1)]['comment_text'])

insult_comment = " ".join(review for review in data[(data['insult']==1)]['comment_text'])

obscene_comment = " ".join(review for review in data[(data['obscene']==1)]['comment_text'])     

toxic_comment = " ".join(review for review in data[(data['toxic']==1)]['comment_text'])


# In[11]:
stopword=set(STOPWORDS)
wordcloud = WordCloud(max_font_size=50, max_words=3000, background_color="white",stopwords=stopword).generate(clean_comment)
plt.figure()
plt.figure(figsize=(15, 10))
plt.title("Clean Comments", fontsize=20)
plt.axis("off")
plt.imshow(wordcloud, interpolation="bilinear")
plt.show()


# In[12]:


# word cloud for clean and not clean comments
stopword=set(STOPWORDS)
wordcloud = WordCloud(max_font_size=50, max_words=2000, background_color="white",stopwords=stopword).generate(clean_comment)

plt.figure(figsize=(20, 20))
plt.subplot(2,2,1)
plt.title("Common words in Clean Comments", fontsize=30)
plt.axis("off")
plt.imshow(wordcloud, interpolation="bilinear")

wordcloud = WordCloud(max_font_size=50, max_words=2000, background_color="white",stopwords=stopword).generate(negative_comment)
plt.subplot(2,2,2)
plt.title("Common Words in negative comments", fontsize=30)
plt.axis("off")
plt.imshow(wordcloud, interpolation="bilinear")
plt.show()


# In[341]:


# building the word cloud
stopword=set(STOPWORDS)
wordcloud = WordCloud(max_font_size=50, max_words=2000, background_color="white",stopwords=stopword).generate(severe_toxic_comment)

plt.figure(figsize=(15, 15))
plt.subplot(2,2,1)
plt.title(" Common Words of Severe Toxic type", fontsize=10)
plt.axis("off")
plt.imshow(wordcloud, interpolation="bilinear")

wordcloud = WordCloud(max_font_size=50, max_words=2000, background_color="white",stopwords=stopword).generate(threat_comment)
plt.subplot(2,2,2)
plt.title("Common Words of Threat type", fontsize=10)
plt.axis("off")
plt.imshow(wordcloud, interpolation="bilinear")


plt.figure(figsize=(15, 15))

wordcloud = WordCloud(max_font_size=50, max_words=2000, background_color="white",stopwords=stopword).generate(insult_comment)
plt.subplot(2,2,3)
plt.title("Common Words of Insult type", fontsize=10)
plt.axis("off")
plt.imshow(wordcloud, interpolation="bilinear"
          )

wordcloud = WordCloud(max_font_size=50, max_words=2000, background_color="white",stopwords=stopword).generate(identity_hate_comment)
plt.subplot(2,2,4)
plt.title("Commen words of Identity hate type ", fontsize=10)
plt.axis("off")
plt.imshow(wordcloud, interpolation="bilinear")
plt.show()


# In[342]:
stopword=set(STOPWORDS)
wordcloud = WordCloud(max_font_size=50, max_words=2000, background_color="white",stopwords=stopword).generate(obscene_comment)

plt.figure(figsize=(15, 15))
plt.subplot(2,2,1)
plt.title("Commen words of Obscene type", fontsize=15)
plt.axis("off")
plt.imshow(wordcloud, interpolation="bilinear")

wordcloud = WordCloud(max_font_size=50, max_words=2000, background_color="white",stopwords=stopword).generate(toxic_comment)
plt.subplot(2,2,2)
plt.title("Commen words of Toxic type", fontsize=15)
plt.axis("off")
plt.imshow(wordcloud, interpolation="bilinear")


#################################################################################
#######################Part 2 - Vectors & Modelling###############################
##################################################################################


# -*- coding: utf-8 -*-
"""
Created on Sat Jan 25 01:16:54 2020

@author: Nitika
"""

import nltk
nltk.download('punkt')

import pandas as pd
import re
import string
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import numpy as np
from nltk.stem.wordnet import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer
from nltk.tokenize import RegexpTokenizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report


wikitoxic = pd.read_csv('train.csv')

stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()
lem = WordNetLemmatizer()
tokenCV = RegexpTokenizer(r'[a-zA-Z0-9]+')
cv = CountVectorizer(lowercase=True,stop_words='english',ngram_range = (1,1),tokenizer = tokenCV.tokenize)


w, h = 1,len(wikitoxic);
comments_clean = [[0 for x in range(w)] for y in range(h)] 

w, h = 2,len(wikitoxic);
comments_sa = [[0 for x in range(w)] for y in range(h)] 

for i in range(0,len(wikitoxic)):
	comments = wikitoxic.iloc[i,1]
	comments = comments.lower()
	comments = re.sub(r'\d+', '', comments)
	comments = comments.translate(str.maketrans("","", string.punctuation))
	comments = comments.strip()
	comments = lem.lemmatize(comments)
	tokens = word_tokenize(comments)
	comments = [j for j in tokens if not j in stop_words]
	comments_clean[i] = str(comments)

text_counts = cv.fit_transform(comments_clean)

#Splitting the Data
X_train, X_test, y_train, y_test = train_test_split(
    text_counts, wikitoxic.iloc[:,2:8], test_size=0.3, random_state=0)

#prediction matrix for saving prediction from the model for each class 
l, b = len(y_test), 6;
Pred_Matrix = [[0 for x in range(l)] for y in range(b)] 
b = np.array(y_test.values.tolist())    

##########################################################################
#building logistic regression model for each class
for i in range(0,6):
	logreg = LogisticRegression()
	logreg.fit(X_train, y_train.iloc[:,i])
	Pred_Matrix[i] = logreg.predict(X_test)
		
prob = pd.DataFrame(data=Pred_Matrix)
prob_t = prob.T

#Comparing the performance
print(accuracy_score(y_test, prob_t))   
print(f1_score(y_test, prob_t, average="macro"))
print(precision_score(y_test, prob_t, average="macro"))
print(recall_score(y_test, prob_t, average="macro"))    
print(classification_report(y_test, prob_t))    

########################################################################3
# Building Multinomial Naive Bayes Model
for i in range(0,6):
	clf = MultinomialNB().fit(X_train, y_train.iloc[:,i])
	Pred_Matrix[i]= clf.predict(X_test)

prob = pd.DataFrame(data=Pred_Matrix)
prob_t = prob.T

#Comparing the performance
print(accuracy_score(y_test, prob_t))   
print(f1_score(y_test, prob_t, average="macro"))
print(precision_score(y_test, prob_t, average="macro"))
print(recall_score(y_test, prob_t, average="macro"))    
print(classification_report(y_test, prob_t)) 

##########################################################################
#Building Random Forest Model
from sklearn.ensemble import RandomForestClassifier
for i in range(0,6):
	rfmodel = RandomForestClassifier(n_estimators=100)
	rfmodel.fit(X_train, y_train)
	Pred_Matrix[i]= rfmodel.predict(X_test)
	
prob = pd.DataFrame(data=Pred_Matrix)
prob_t = prob.T

#Comparing the performance
print(accuracy_score(y_test, prob_t))   
print(f1_score(y_test, prob_t, average="macro"))
print(precision_score(y_test, prob_t, average="macro"))
print(recall_score(y_test, prob_t, average="macro"))    
print(classification_report(y_test, prob_t)) 

##########################################################################
#Building XGBoost Model
from xgboost import XGBClassifier
xgb_model = XGBClassifier()
for i in range(0,6):
    xgb_model.fit(X_train, y_train.iloc[:,i])
    Pred_Matrix[i]= xgb_model.predict(X_test)
	
prob = pd.DataFrame(data=Pred_Matrix)
prob_t = prob.T

#Comparing the performance
print(accuracy_score(y_test, prob_t))   
print(f1_score(y_test, prob_t, average="macro"))
print(precision_score(y_test, prob_t, average="macro"))
print(recall_score(y_test, prob_t, average="macro"))    
print(classification_report(y_test, prob_t)) 
	  
##########################################################################
##############################TF-IDF 

# loading the data set
toxic_data = pd.read_csv("C:\\Users\shwet\Documents\shweta\Winter_2020\A\Predictive\Group_Project/train.csv")
toxic_data.head()

#Splitting the data set

from sklearn.model_selection import train_test_split
xtrain, xtest, ytrain, ytest = train_test_split(
    toxic_data['clean_comments'], toxic_data.iloc[:,2:8], test_size=0.3, random_state=1)

# performing TF-IDF vaectorization
from sklearn.feature_extraction.text import TfidfVectorizer
# word level tf-idf
tfidf_vect = TfidfVectorizer(analyzer='word', token_pattern=r'\w{1,}', stop_words='english',ngram_range=(2,3),max_features=5000,lowercase=False)
tfidf_vect.fit(toxic_data['comment_text'])
xtrain_tfidf =  tfidf_vect.fit_transform(xtrain)
xtest_tfidf =  tfidf_vect.transform(xtest)

# #characters level tf-idf
tfidf_vect_ngram_chars = TfidfVectorizer(analyzer='char',stop_words='english', token_pattern=r'\w{1,}', ngram_range=(2,3), max_features=5000)
tfidf_vect_ngram_chars.fit(toxic_data['comment_text'])
xtrain_tfidf_chars =  tfidf_vect_ngram_chars.fit_transform(xtrain) 
xtest_tfidf_chars =  tfidf_vect_ngram_chars.transform(xtest)

# stacking the features, wordlevl tf-idf and characters level tf-idf
from scipy import sparse
X = sparse.hstack([xtrain_tfidf, xtrain_tfidf_chars])
x_test = sparse.hstack([xtest_tfidf, xtest_tfidf_chars])


# creating matrix to store predcicted values
l, b = len(ytrain), 6;
Pred_Matrix = [[0 for x in range(l)] for y in range(b)]

#----------------------------------------------------
#---Logistic Regression------------------------------
from sklearn.linear_model import LogisticRegression
#defining logistic regression function
def logreg(train_x,train_y,test_x):
    logreg = LogisticRegression(solver='sag')
    logreg.fit(X, ytrain)
    y_pred = logreg.predict(x_test)
    return y_pred
for i in range(0, 6):
    Pred_Matrix[i] = logreg(X, ytrain.iloc[:,i],x_test)
pred_lr = pd.DataFrame(data=Pred_Matrix)
pred_lr_val =pred_lr.T
print('Accuracy of the Logistic Regression model : '+str(metrics.accuracy_score(ytest, pred_lr_val)))
print('Classification Report for Logistic Regression : ')
print(metrics.classification_report(ytest, pred_lr_val))


# In[ ]:


#---------Naive Baye's------
from sklearn.naive_bayes import MultinomialNB
for i in range(0,6):
    mnb_model = MultinomialNB().fit(X, ytrain.iloc[:,i])
#     Pred_Matrix[i]= mnb_model.predict_proba(x_test)[:,1]
    Pred_Matrix[i]= mnb_model.predict(x_test)
pred_mnb = pd.DataFrame(data=Pred_Matrix)
pred_mnb_val =pred_mnb.T
print('Accuracy of the Logistic Regression model : '+str(metrics.accuracy_score(ytest, pred_mnb_val)))
print('Classification Report for Logistic Regression : ')
print(metrics.classification_report(ytest, pred_mnb_val))


# In[ ]:


#-------Random Forest with tree= 500--------
# libaries for random forest and metrics
from sklearn.ensemble import RandomForestClassifier
rfmodel = RandomForestClassifier(n_estimators=500)
rfmodel.fit(X, ytrain)

#-----predicting values for test-----
Pred_Matrix = rfmodel.predict(x_test)
pred2 = pd.DataFrame(data=Pred_Matrix)
from sklearn import metrics
print('Accuracy of the Random Forest model : '+str(metrics.accuracy_score(ytest, pred2)))
print('Classification Report for Random Forest : ')
print(metrics.classification_report(ytest, pred2))


# In[ ]:


#-------Random Forest with tree= 1000--------
# libaries for random forest and metrics
from sklearn.ensemble import RandomForestClassifier
rfmodel = RandomForestClassifier(n_estimators=1000)
rfmodel.fit(X, ytrain)

#-----predicting values for test-----
Pred_Matrix = rfmodel.predict(x_test)
pred2 = pd.DataFrame(data=Pred_Matrix)
from sklearn import metrics
print('Accuracy of the Random Forest model : '+str(metrics.accuracy_score(ytest, pred2)))
print('Classification Report for Random Forest : ')
print(metrics.classification_report(ytest, pred2))


# In[ ]:


#-------Random Forest with tree= 1500--------
# libaries for random forest and metrics
from sklearn.ensemble import RandomForestClassifier
rfmodel = RandomForestClassifier(n_estimators=1500)
rfmodel.fit(X, ytrain)

#-----predicting values for test-----
Pred_Matrix = rfmodel.predict(x_test)
pred2 = pd.DataFrame(data=Pred_Matrix)
from sklearn import metrics
print('Accuracy of the Random Forest model : '+str(metrics.accuracy_score(ytest, pred2)))
print('Classification Report for Random Forest : ')
print(metrics.classification_report(ytest, pred2))


#-------------XGBoost-------------
# libaries for GBM
from xgboost import XGBClassifier
for i in range(0,6):
    xgb_model = XGBClassifier()
    xgb_model.fit(X, ytrain.iloc[:,i])
    Pred_Matrix[i]= xgb_model.predict(x_test)
#-----predicting values for test
pred_xgb = pd.DataFrame(data=Pred_Matrix)
pred_val_xgb =pred_xgb.T
print('Accuracy of the Random Forest model : '+str(metrics.accuracy_score(ytest, pred_val_xgb)))
print('Classification Report for Random Forest : ')
print(metrics.classification_report(ytest, pred_val_xgb))

# In[ ]:

###################################################Final Model###############################################
#-------Random Forest--------
# libaries for random forest and metrics
from sklearn.ensemble import RandomForestClassifier
rfmodel = RandomForestClassifier(n_estimators=100)
rfmodel.fit(X, ytrain)

#-----predicting values for test-----
Pred_Matrix = rfmodel.predict(x_test)
pred1 = pd.DataFrame(data=Pred_Matrix)
from sklearn import metrics
print('Accuracy of the Random Forest model : '+str(metrics.accuracy_score(ytest, pred1)))
print('Classification Report for Random Forest : ')
print(metrics.classification_report(ytest, pred1))

submid = pd.DataFrame({'comment': testdf["comment"]})
df=pd.concat([submid, pd.DataFrame(pred_mnb_val)], axis=1)
df = df.rename(columns={0:'toxic', 1:'severe_toxic', 2:'obscene', 3:'threat', 4:'insult', 5:'identity_hate'})






