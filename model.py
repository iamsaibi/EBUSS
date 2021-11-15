# %%
"""
# sentiment-based product recommendation system: 

    Performed following tasks:
    1.Data sourcing and sentiment analysis
    2.Building a recommendation system
    3.Recommending top 5 products
    4.Deploying the end-to-end project with a user interface

"""

# %%
# Importing libraries
import numpy as np
import pandas as pd
import random
import pickle
import pylab
from numpy import *


import matplotlib.pyplot as plt
import seaborn as sns
import time
from wordcloud import WordCloud
from collections import Counter


from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
import re

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection  import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import RandomizedSearchCV

from imblearn.over_sampling import SMOTE
from collections import Counter

from sklearn.metrics import f1_score, classification_report,precision_score,recall_score,confusion_matrix, roc_auc_score, roc_curve

from sklearn.metrics.pairwise import pairwise_distances

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)


# %%
#df = pd.read_csv('/content/gdrive/MyDrive/sample30.csv')
df = pd.read_csv('sample30.csv')


# %%
"""
### Preprocessing:
    Steps followed:
       1. Handling null values
       2. Preprocessing reviews text and visualization 
"""

# %%
"""
#### 1. Handling null values:
    Replaced NaN in reviews_title by empty space and merged reviews and reviews_title.
"""

# %%
df.isnull().sum()

# %%
# only one null value in target change it into 0
# change positive to 1 and negative to 0
df['user_sentiment']= df['user_sentiment'].apply(lambda x:1 if x=='Positive' else 0)


# %%
# Replace nulls 
df['reviews_title'].fillna('',inplace=True)

# %%
# merge reviews columns
df['reviews']=df['reviews_text']+df['reviews_title']
df.drop(['reviews_text','reviews_title'],axis=1,inplace=True)
df.head()

# %%
# df_clean -> cleaned columns for recommendation and sentiment models
df_clean = df[['name','reviews_username','reviews','reviews_rating','user_sentiment']]
df_clean.head()

# %%
df_clean.dropna(inplace=True)

# %%
df_clean.info()

# %%
"""
#### 2. Text preprocessing:
    Removed stops words after converting the text into lowercase. 
"""

# %%
# function to convert text into lowercase, remove stopwords and special characters
def text_process(token):
    tokens = word_tokenize(token)
    words_lower = [word.lower() for word in tokens]
    words_nostop = [word for word in words_lower if word not in stopwords.words('english')]
    text = ' '.join(re.sub('[^a-zA-Z0-9]+', ' ', word) for word in words_nostop)
    return  text


# %%
# text preprocessing
df_clean['reviews'] = df_clean['reviews'].apply(lambda x:text_process(x))


# %%
"""
# Sentiment analysis:

       To build sentiment analysis model, take reviews given by the users. 
       Steps followed:
       1. Feature extraction using tf-idf
       2. Handling imbalance
       3. Build 3 ML models 
"""

# %%
# dataframe for sentiment analysis
Review = df_clean[['name','reviews','user_sentiment']]
Review.head()

# %%
# splitting into test and train

X_train, X_test, y_train, y_test = train_test_split(Review['reviews'], Review['user_sentiment'],test_size=0.30, random_state=42)

# %%
X_train.shape

# %%
"""
#### 1. Feature extarction:
    Used tf-idf vectorizer to extract features from text.
"""

# %%
# tf-idf
vectorizer= TfidfVectorizer(max_features=3000, lowercase=True, analyzer='word', stop_words= 'english')
tf_x_train = vectorizer.fit_transform(X_train).toarray()
tf_x_test = vectorizer.transform(X_test)

# %%
tf_x_train.shape

# %%
"""
#### 2. Handling imbalance:
    Used SMOTE to handle class imbalance.
"""

# %%
# SMOTE
print('Before Sampling')
print(Counter(y_train))

sm = SMOTE(random_state=42)
X_train_sm ,y_train_sm = sm.fit_sample(tf_x_train,y_train)

print('After Sampling')
print(Counter(y_train_sm))

# %%
"""
#### 3. Model building:
"""

# %%
"""
#### Logistic Regression model:
"""

# %%
lr=LogisticRegression()

params={'C':[10, 1, 0.5, 0.1],'penalty':['l1','l2'],'class_weight':['balanced']}


# Create grid search using 4-fold cross validation
grid_search = GridSearchCV(lr, params, cv=4, scoring='roc_auc', n_jobs=-1)
grid_search.fit(X_train_sm, y_train_sm)
model_LR = grid_search.best_estimator_
model_LR.fit(X_train_sm, y_train_sm)

# %%
# Logitic model evalution
y_prob_test=model_LR.predict_proba(tf_x_test)
y_pred_test=model_LR.predict(tf_x_test)

print('Test Score:')
print('Confusion Matrix')
print('='*60)
print(confusion_matrix(y_test,y_pred_test),"\n")
print('Classification Report')
print('='*60)
print(classification_report(y_test,y_pred_test),"\n")
print('AUC-ROC=',roc_auc_score(y_test, y_prob_test[:,1]))
    

fpr_LR, tpr_LR, thresholds_LR = roc_curve(y_test, y_prob_test[:,1])

AUC_ROC_LR = roc_auc_score(y_test, y_prob_test[:,1])


# %%
# store tf-idf model
with open("tfidf_model.pkl", 'wb') as file:
    pickle.dump(vectorizer, file)

# %%
# save logistic regression model
with open('LR_sentiment_model.pkl', 'wb') as file:
    pickle.dump(model_LR, file)

# %%
# save logistic regression model
with open('df_sentiment_model.pkl', 'wb') as file:
    pickle.dump(df_clean, file)

# %%
"""
# Recommendation system:
 
"""

# %%
"""
To build recommendation system taks user name , product name and review ratings.
"""

# %%
# create recommedation data frame
recomm_df = df_clean[['reviews_username','reviews_rating','name']]
recomm_df.head()

# %%
"""
**Create train and test set**
"""

# %%
# Test and Train split of the dataset

train, test = train_test_split(recomm_df, test_size=0.30, random_state=31)

# %%
# Pivot the train dataset into matrix format in which columns are products and the rows are user names.
df_pivot = train.pivot_table(
    index='reviews_username',
    columns='name',
    values='reviews_rating'
).fillna(0)

df_pivot.head()

# %%
df_pivot.shape

# %%
"""
**Creating Dummy train and test**

In the process of building a recommendation system, we do not want to recommend a product that the user has already rated or in some cases has performed some action on it such as view, like, share or comment. To eliminate these products from the recommendation list, you will need to take the help of a ‘dummy data set’.

"""

# %%
# Copy the train dataset into dummy_train
dummy_train = train.copy()

# %%
# The movies not rated by user is marked as 1 for prediction. 
dummy_train['reviews_rating'] = dummy_train['reviews_rating'].apply(lambda x: 0 if x>=1 else 1)

# %%
# Convert the dummy train dataset into matrix format.
dummy_train = dummy_train.pivot_table(
    index='reviews_username',
    columns='name',
    values='reviews_rating'
).fillna(1)

dummy_train.head()

# %%
dummy_train.shape

# %%
"""
#### User Similarity Matrix:
"""

# %%
"""
### Using adjusted Cosine similarity:

 Here, we are not removing the NaN values and calculating the mean only for the movies rated by the user
"""

# %%
df_pivot = train.pivot_table(
    index='reviews_username',
    columns='name',
    values='reviews_rating'
)
df_pivot.head(3)

# %%
#Normalising the rating of the movie for each user around 0 mean
mean = np.nanmean(df_pivot, axis=1)
df_subtracted = (df_pivot.T-mean).T
df_subtracted.head()

# %%
"""
#### Find cosine similarity:
    Used pairwise distance to find similarity.
"""

# %%
# Creating the User Similarity Matrix using pairwise_distance function.
user_correlation = 1 - pairwise_distances(df_subtracted.fillna(0), metric='cosine')
user_correlation[np.isnan(user_correlation)] = 0
print(user_correlation)

# %%
"""
#### Prediction:
"""

# %%
# Ignore the correlation for values less than 0.
user_correlation[user_correlation<0]=0
user_correlation

# %%
"""
Rating predicted by the user is the weighted sum of correlation with the product rating.
"""

# %%
user_predicted_ratings = np.dot(user_correlation, df_pivot.fillna(0))
user_predicted_ratings

# %%
user_predicted_ratings.shape

# %%
# user_final_rating -> this contains predicted ratings for products
user_final_rating = np.multiply(user_predicted_ratings,dummy_train)
user_final_rating.head()

# %%
"""
#### Find the top 5 recommendation for the *user*
"""

# %%
# Take the user ID as input [bob,00sab00]
#user_input = str(input("Enter your user name"))
user_input = str('00sab00') # for checking


# %%
# Recommended products for the selected user based on ratings
out_recommendation = user_final_rating.loc[user_input].sort_values(ascending=False)[:20]
out_recommendation

# %%
"""
#### Evaluation - User User 
"""

# %%
# Find out the common users of test and train dataset.
common = test[test.reviews_username.isin(train.reviews_username)]
common.shape

# %%
# convert into the user-product matrix.
common_user_based_matrix = common.pivot_table(index='reviews_username', columns='name', 
                                              values='reviews_rating')


# %%
# Convert the user_correlation matrix into dataframe.
user_correlation_df = pd.DataFrame(user_correlation)

# %%
user_correlation_df['userId'] = df_subtracted.index
user_correlation_df.set_index('userId',inplace=True)
user_correlation_df.head()

# %%
common.head(1)

# %%
list_name = common.reviews_username.tolist()

user_correlation_df.columns = df_subtracted.index.tolist()


user_correlation_df_1 =  user_correlation_df[user_correlation_df.index.isin(list_name)]
user_correlation_df_1.shape

# %%
user_correlation_df_2 = user_correlation_df_1.T[user_correlation_df_1.T.index.isin(list_name)]

# %%
user_correlation_df_3 = user_correlation_df_2.T
user_correlation_df_3.head()

# %%
user_correlation_df_3.shape

# %%
user_correlation_df_3[user_correlation_df_3<0]=0

common_user_predicted_ratings = np.dot(user_correlation_df_3, common_user_based_matrix.fillna(0))
common_user_predicted_ratings

# %%
dummy_test = common.copy()

dummy_test['reviews_rating'] = dummy_test['reviews_rating'].apply(lambda x: 1 if x>=1 else 0)

dummy_test = dummy_test.pivot_table(index='reviews_username', columns='name', values='reviews_rating').fillna(0)

# %%
common_user_predicted_ratings = np.multiply(common_user_predicted_ratings,dummy_test)
common_user_predicted_ratings

# %%
"""
#### Find rmse :
"""

# %%
X  = common_user_predicted_ratings.copy() 
X = X[X>0]

scaler = MinMaxScaler(feature_range=(1, 5))
print(scaler.fit(X))
y = (scaler.transform(X))



# %%
common_ = common.pivot_table(index='reviews_username', columns='name', values='reviews_rating')

# %%
# Finding total non-NaN value
total_non_nan = np.count_nonzero(~np.isnan(y))

# %%
rmse_user = round((sum(sum((common_ - y )**2))/total_non_nan)**0.5,2)
print(rmse_user)

# %%
"""
### Save model :
"""

# %%
user_final_rating.to_csv('user_based_recomm.csv')

# %%
pickle.dump(user_final_rating, open('user_based_recomm_model.pkl','wb'))

# %%
"""
## Recommendation of Top 20 Products to a Specified User
"""

# %%
# load all pkl files
tfidf_model = pickle.load(open('tfidf_model.pkl', 'rb'))
user_based_recomm_model = pickle.load(open('user_based_recomm_model.pkl', 'rb'))
LR_sentiment_model = pickle.load(open('LR_sentiment_model.pkl', 'rb'))

# %%
# Enter user name

user = str('00sab00')  # for e.g

# %%
# Recommend top 20 products
user_top20 = user_based_recomm_model.loc[user].sort_values(ascending=False)[:20]

# %%
user_top20 = pd.DataFrame(user_top20)  #.to_records())
user_top20.reset_index(inplace = True)
user_top20

# %%
# merge top 20 products and its reviews
top20_products_setiment = pd.merge(user_top20,df_clean,on = ['name'])
top20_products_setiment.head()

# %%
"""
Feed 'top20_products' into tfidf model first and into sentiment model to find sentiment score.
"""

# %%
# convert text to feature
top20_products_tfidf = tfidf_model.transform(top20_products_setiment['reviews'])

# %%
# model prediction
top20_products_pred =LR_sentiment_model.predict(top20_products_tfidf)
top20_products_pred

# %%

top20_products_setiment['top20_products_pred']=top20_products_pred


# %%
"""
 senti_score is given by the percentage of positive reviews to the total reviews for each products.
"""

# %%
senti_score = top20_products_setiment.groupby(['name'])['top20_products_pred'].agg(['sum','count']).reset_index()
senti_score['percent'] = round((100*senti_score['sum'] / senti_score['count']),2)
senti_score.head()

# %%
"""
## Top 5 products:

    Top 5 products based on sentiment score.
"""

# %%
senti_score = senti_score.sort_values(by='percent',ascending=False)
senti_score

# %%
senti_score['name'].head().tolist()

# %%
