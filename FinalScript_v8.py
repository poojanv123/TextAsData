import os
import pandas as pd
import numpy as np
import re
import nltk
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from statistics import mode
from statistics import StatisticsError


# Vectorization
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.multiclass import OneVsOneClassifier
from sklearn.multiclass import OneVsRestClassifier

#Sentiment analysis Lexicon
from textblob import TextBlob
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# Load classifiers
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.svm import SVC

# Load functions for model selection and performance
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from sklearn.model_selection import GridSearchCV

#Import tweets datafile
os.chdir('E:/Study/Text as Data/Project')
tweets = pd.read_csv('demonetization-tweets.csv', encoding = 'utf-8')

#Remove political parties tweets
parties = ['BJP4India', 'INCIndia', 'NCPSpeaks', 'AamAadmiParty', 'PMOIndia']
for i in range(len(tweets)):
    if tweets['name'][i] in parties:
        tweets.drop([0,i])

#Subset the data for tweet text and date of the tweet
tweets = tweets[['text', 'created']]

#Pre-processing of tweets content

tweets['cleantweet'] = ''

for i in range(len(tweets)):
    #Remove callouts
    #tweets['cleantweet'][i] = re.sub(r'@[A-Za-z0-9_]+','', tweets['text'][i])
    #Remove links
    tweets['cleantweet'][i] = re.sub(r'http\S+','', tweets['text'][i])
   # tweets['cleantweet'][4] m = re.search(r'^\d+', tweets['cleantweet'][4])
    tweets['cleantweet'][i] = tweets['cleantweet'][i].lower()
    

#Tokenization
tokens = [RegexpTokenizer(r'\w+').tokenize(tweet) for tweet in tweets['cleantweet']]

#Remove numbers from the text
def remove_number(text):
    return [token for token in text if re.search(r'^\d+$', token) == None]

tokens = [remove_number(doc) for doc in tokens]
# Lemmetize tweet text
lemmatizer =  WordNetLemmatizer()

def lemmatize(text, lemmatizer):
    return [lemmatizer.lemmatize(token) for token in text]

tokens = [lemmatize(doc, lemmatizer) for doc in tokens]
print(tokens[2])

#Put processed tweets text tokens into a 'cleantext' column
tweets['cleantext'] = [' '.join(doc) for doc in tokens]

#Random 1000 tweets as training data
label_data = tweets[['cleantext','created']].sample(1000)
label_data.to_csv('E:/Study/Text as Data/Project/training_data3.csv', encoding = 'utf-8')

#Load annotated data
label_data = pd.read_csv('E:/Study/Text as Data/Project/training_data.csv', encoding = 'utf-8')

#Method 1: Vader sentiment intensity analyser
#Use vader sentiment intensity analyzer
vader = SentimentIntensityAnalyzer()

#Calculate sentiment score of the tweets
label_data['sentiment_scores_pos_vader'] = [vader.polarity_scores(text)['pos'] for text in label_data['cleantext']]
label_data['sentiment_scores_neg_vader'] = [vader.polarity_scores(text)['neg'] for text in label_data['cleantext']]
label_data['sentiment_scores_neu_vader'] = [vader.polarity_scores(text)['neu'] for text in label_data['cleantext']]

#Identify the sentiment type of the tweets
label_data['sentiment_type_vader'] = ''

label_data.loc[label_data['sentiment_scores_pos_vader'] > 0.1, 'sentiment_type_vader'] = 'Positive'
label_data.loc[label_data['sentiment_scores_neg_vader'] > 0.1, 'sentiment_type_vader'] = 'Negative'
label_data.loc[label_data['sentiment_scores_neu_vader'] > 0.7, 'sentiment_type_vader'] = 'Neutral'

#Check F1 score and accuracy of the Lexicon
print('F1 score (macro) = %s' % f1_score(label_data['sentiment_type_training'], label_data['sentiment_type_vader'], average = 'macro'))
print('Accuracy = %s' % accuracy_score(label_data['sentiment_type_training'], label_data['sentiment_type_vader']))

#Method 2: Textblob sentiment intensity analyser

#Calculate sentiment score of the tweets
label_data['sentiment_score_textblob'] = [TextBlob(text).sentiment.polarity for text in label_data['cleantext']]

#Give labels to the tweets based on the sentiment score
label_data['sentiment_type_textblob'] = 'Neutral'
label_data.loc[label_data['sentiment_score_textblob'] > 0.2, 'sentiment_type_textblob'] = 'Positive'
label_data.loc[label_data['sentiment_score_textblob'] < -0.1, 'sentiment_type_textblob'] = 'Negative'

#Check F1 score and accuracy of the Lexicon
print('F1 score (macro) = %s' % f1_score(label_data['sentiment_type_training'], label_data['sentiment_type_textblob'], average = 'macro'))
print('Accuracy = %s' % accuracy_score(label_data['sentiment_type_textblob'], label_data['sentiment_type_vader']))

#method 3: Naive Bayes Classifier

#Vectorizing the training dataset
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(label_data['cleantext'])
Y = np.array(label_data['sentiment_type_training'])

#Checking for class imbalance
def get_freq(obj):
    unique, counts = np.unique(obj, return_counts=True)
    print(np.asarray((unique, counts)).T)
    return unique, counts

print(get_freq(Y))
#it is imbalanced

# Initialize the Naive Bay's classifer
clf_nb = MultinomialNB()

# Fit the model
clf_fit_nb = clf_nb.fit(X, Y)
print("In-sample accuracy = %s" % clf_fit_nb.score(X, Y))
# In-sample accuracy = 94.8

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.30, random_state=1234)
print("Training set size = %s" % y_train.shape)
print("Testing set size = %s" % y_test.shape)

# Fit the model
clf_fit_nb = clf_nb.fit(X_train, y_train)

# Generate predictions
y_predict_nb = clf_fit_nb.predict(X_test)

print('F1 score = %s' % f1_score(y_test, y_predict_nb, average = 'macro'))
print('Accuracy = %s' % accuracy_score(y_test, y_predict_nb))

print(clf_fit_nb.score(X_test, y_test))
#F1 score = 0.4521004093866758
#Accuracy =  0.4521004093866758


#Try another models
#Linear SVC One vs Rest classifier

clf_svc =LinearSVC(class_weight='balanced')

# Fit the model using the training data
# generated above.
clf_fit_svc = clf_svc.fit(X_train, y_train)

# Generate predictions
y_predict_svc = clf_fit_svc.predict(X_test)

# Output performance metrics
print('F1 score (macro) = %s' % f1_score(y_test, y_predict_svc, average = 'macro'))
print('Accuracy = %s' % accuracy_score(y_test, y_predict_svc))
print(clf_fit_svc.score(X_test, y_test))
#F1 score (macro) = 0.5056365489973892
#Accuracy = 0.5733333333333334
#0.5733333333333334

#parameter tuning
param_grid = [
  {'C': [.1, 0.5, 1], 'loss': ['hinge']},
  {'C': [.1, 0.5, 1], 'loss': ['squared_hinge']},
 ]
# Initialize model
lsvc = LinearSVC(class_weight='balanced')

# Intialize grid search
grid = GridSearchCV(lsvc, param_grid=param_grid, cv=5, scoring = 'f1_macro')

# Run grid search
grid.fit(X, Y)

print("The best model based on CV =")
print(grid.best_estimator_)

print("\nThe score of the best model = ")
print(grid.best_score_)

#Final model
clf_svc = LinearSVC(C = 1,class_weight='balanced')

clf_fit_svc = clf_svc.fit(X_train, y_train)

# Generate predictions
y_predict_svc = clf_fit_svc.predict(X_test)

# Output performance metrics
print('F1 score (macro) = %s' % f1_score(y_test, y_predict_svc, average = 'macro'))
print('Accuracy = %s' % accuracy_score(y_test, y_predict_svc))
print(clf_fit_svc.score(X_test, y_test))
#F1 score (macro) = 0.5056365489973892
#Accuracy = 0.5733333333333334
#0.5733333333333334

results = []
for k, (train_idx, test_idx) in enumerate(kf.split(X)):
    # Fit model
    cfit = clf_svc.fit(X[train_idx], Y[train_idx])
    
    # Get predictions
    y_pred = cfit.predict(X[test_idx])
    
    # Write results
    result = {'fold': k,
              'precision': precision_score(Y[test_idx], y_pred, average = 'macro'),
              'recall': recall_score(Y[test_idx], y_pred, average = 'macro'),
              'f1': f1_score(Y[test_idx], y_pred, average = 'macro')}
              
    results.append(result)

print(results[0])
#Cross-validation over

# Get the average F1 score
mean_f1 = np.mean(np.array([row['f1'] for row in results]))

print("Average, cross-validated F1 score = %s" % mean_f1)

#Average, cross-validated F1 score = 0.5535347960152133

#Using TF-IDF Vectorizer
vectorizer = TfidfVectorizer()
X_tfidf = vectorizer.fit_transform(label_data['cleantext'])
X_train_tfidf, X_test_tfidf, y_train, y_test = train_test_split(X_tfidf, Y, test_size=0.30, random_state=1234)

#Naive bayes
clf_fit_nb = clf_nb.fit(X_train_tfidf, y_train)
y_predict_nb_tfidf = clf_fit_nb.predict(X_test_tfidf)
print('F1 score = %s' % f1_score(y_test, y_predict_nb_tfidf, average = 'macro'))
#SVC
clf_fit_svc = clf_svc.fit(X_train_tfidf, y_train)
y_predict_svc_tfidf = clf_fit_svc.predict(X_test_tfidf)
print('F1 score (macro) = %s' % f1_score(y_test, y_predict_svc_tfidf, average = 'macro'))

#Creating final sentiment type by taking the Mode of all models predictions
label_data['sentiment_type_nb'] = clf_fit_nb.predict(X)
label_data['sentiment_type_svc'] = clf_fit_svc.predict(X)

label_data = label_data[['sentiment_type_training',
 'cleantext',
 'sentiment_scores_pos_vader',
 'sentiment_scores_neg_vader',
 'sentiment_scores_neu_vader',
 'sentiment_score_textblob',
 'sentiment_type_vader',
 'sentiment_type_textblob',
 'sentiment_type_nb',
 'sentiment_type_svc']] 

final_pred = np.array([])

for i in range(len(label_data)):
    try:
        final_pred = np.append(final_pred, mode(label_data.iloc[i,6:10]))
    except StatisticsError:
        final_pred = np.append(final_pred,label_data.iloc[i,9])

label_data['final_sentiment_type'] = [row for row in final_pred]

label_data.to_csv('E:/Study/Text as Data/Project/training_output.csv', encoding = 'utf-8')

print(f1_score(Y, final_pred, average = 'macro'))
#93.44486078808587

# Fit the model

tweets['sentiment_scores_pos_vader'] = [vader.polarity_scores(text)['pos'] for text in tweets['cleantext']]
tweets['sentiment_scores_neg_vader'] = [vader.polarity_scores(text)['neg'] for text in tweets['cleantext']]
tweets['sentiment_scores_neu_vader'] = [vader.polarity_scores(text)['neu'] for text in tweets['cleantext']]

#Identify the sentiment type of the tweets
tweets['sentiment_type_vader'] = ''

tweets.loc[tweets['sentiment_scores_pos_vader'] > 0.1, 'sentiment_type_vader'] = 'Positive'
tweets.loc[tweets['sentiment_scores_neg_vader'] > 0.1, 'sentiment_type_vader'] = 'Negative'
tweets.loc[tweets['sentiment_scores_neu_vader'] > 0.7, 'sentiment_type_vader'] = 'Neutral'

tweets['sentiment_score_textblob'] = [TextBlob(text).sentiment.polarity for text in tweets['cleantext']]

#Give labels to the tweets based on the sentiment score
tweets['sentiment_type_textblob'] = 'Neutral'
tweets.loc[tweets['sentiment_score_textblob'] > 0.2, 'sentiment_type_textblob'] = 'Positive'
tweets.loc[tweets['sentiment_score_textblob'] < -0.1, 'sentiment_type_textblob'] = 'Negative'

X_unlabeled = vectorizer.transform(tweets['cleantext'])

tweets['sentiment_type_nb'] = clf_fit_nb.predict(X_unlabeled)
tweets['sentiment_type_svc'] = clf_fit_svc.predict(X_unlabeled)
tweets = tweets[['text',
 'cleantext',
 'created',
 'sentiment_scores_pos_vader',
 'sentiment_scores_neg_vader',
 'sentiment_scores_neu_vader',
 'sentiment_score_textblob',
 'sentiment_type_vader',
 'sentiment_type_textblob',
 'sentiment_type_nb',
 'sentiment_type_svc']] 

final_pred = np.array([])

for i in range(len(tweets)):
    try:
        final_pred = np.append(final_pred, mode(tweets.iloc[i,7:11]))
    except StatisticsError:
        final_pred = np.append(final_pred,tweets.iloc[i,10])

tweets['final_sentiment_type'] = [row for row in final_pred]

tweets.to_csv('E:/Study/Text as Data/Project/model_output.csv', encoding = 'utf-8')
