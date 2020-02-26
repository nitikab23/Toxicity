# Toxicity
#Introduction
This challenge was posted in Kaggle by Jigsaw in collaboration with Google’s Alphabet. The reason we chose this dataset is to classify and remove or block rude comments online. All the comments were drawn from Wikipedia talk page where people can post their opinions and experiences online. By implementation of this project, we can reduce online harassment to provide freedom to post publicly.

#Goal
Aim of the project is to correctly classify comments into 6 categories based on level of toxicity. 

#Dataset Info
We have 6 classification output variables:
1.	Toxic
2.	Severe-toxic
3.	Obscene
4.	Threat
5.	Insult
6.	Identity-hate
One ID variable and one input variable i.e. comment_text.

#Data description:
•	Number of variables: 8
•	Number of observations: 159571
•	Missing values:  0
•	Total size: 109.6 mb

 We have used pandas_profiling library to summarize the dataset.
 
#Feature Extraction
We performed feature extraction which is the process of taking out a list of words from the text data and then transforming them into a feature set which would be used as predictors for classification.
There are many ways to extract features using NLP techniques, in this project we have used the following two approach-
1.	Count Vectorization
2.	Term Frequency - Inverse Document Frequency (TF-IDF)

#Classification Models
1. Naive Bayes
2. Logistic Regression
3. Random Forest
4. Gradient Boosting

#Final Model - Random Forest


