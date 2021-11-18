import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, render_template
import pickle


app = Flask(__name__)

# Load models

tfidf_model = pickle.load(open('tfidf_model.pkl', 'rb'))
user_based_recomm_model = pickle.load(open('user_based_recommendation_model.pkl', 'rb'))
LR_sentiment_model = pickle.load(open('LR_sentiment_model.pkl', 'rb'))
df_sentiment_model = pickle.load(open('df_sentiment_model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route("/predict/<user>", methods=['GET'])
def predict(user):        
        #For users not present in the dataset
        if user not in user_based_recomm_model.index:
            return render_template('index.html',prediction_text='Enter valid user name')  
        
        # For the users present in the dataset
        user_recom = user_based_recomm_model.loc[user].sort_values(ascending=False)[:20]
        user_recom = pd.DataFrame(user_recom)  
        user_recom.reset_index(inplace = True)
        # merge products and reviews for sentiment analysis
        top20 = pd.merge(user_recom,df_sentiment_model,on = ['name'])
        
        # tfidf
        products_tfidf = tfidf_model.transform(top20['reviews'])
        #  sentiment prediction
        user_pred =LR_sentiment_model.predict(products_tfidf)
        
        top20['user_pred']=user_pred
        
        # To find percentage of positive sentiments
        senti_score = top20.groupby(['name'])['user_pred'].agg(['sum','count']).reset_index()

        senti_score['percent'] = round((100*senti_score['sum'] / senti_score['count']),2)
        
        # Filter top5 products
        senti_score = senti_score.sort_values(by='percent',ascending=False)
        top5 = senti_score['name'].head().tolist()
   
        return jsonify({'p':top5})


if __name__ == "__main__":
    app.run(debug=True)