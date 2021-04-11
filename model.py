import pandas as pd
from flask import current_app
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
import pickle
import re
import itertools
from nltk.stem import WordNetLemmatizer
from string import punctuation
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
import nltk
import numpy as np
nltk.download("stopwords")
nltk.download('wordnet')



def info(message):
    current_app.logger.info(message)


class ContenEngine(object):
    SIMILAR_KEY = "p:smlr:%s"
    Product_review_df =""
    main_df = ""

    def __init__(self):
        self.dataPereration()


    def dataPereration(self):
        #Load the main data frame
        master_df = pd.read_csv('sample30.csv')
        self.master_df = master_df
        #Preparing relavent small datafram
        Product_review_df = pd.DataFrame()
        Product_review_df['comb_review'] = master_df['reviews_title'].str.lower() + ' ' + master_df[
            'reviews_text'].str.lower()
        # Cleaning comb_review
        Product_review_df['comb_review'] = Product_review_df['comb_review'].map(lambda x: self.cleanData(x))

        Product_review_df['user_sentiment_code'] = master_df['user_sentiment'].str.lower()
        Product_review_df['product_name'] = master_df['name'].str.lower()
        self.Product_review_df = Product_review_df

        self.main_df = pd.read_pickle("./main_df_transf.pkl")

    # Cleaning Data
    def cleanData(self, text, lowercase=True, remove_stops=False, stemming=True, lemmatization=True):
        txt = str(text)

        # Replace apostrophes with standard lexicons
        txt = txt.replace("isn't", "is not")
        txt = txt.replace("aren't", "are not")
        txt = txt.replace("ain't", "am not")
        txt = txt.replace("won't", "will not")
        txt = txt.replace("didn't", "did not")
        txt = txt.replace("shan't", "shall not")
        txt = txt.replace("haven't", "have not")
        txt = txt.replace("hadn't", "had not")
        txt = txt.replace("hasn't", "has not")
        txt = txt.replace("don't", "do not")
        txt = txt.replace("wasn't", "was not")
        txt = txt.replace("weren't", "were not")
        txt = txt.replace("doesn't", "does not")
        txt = txt.replace("'s", " is")
        txt = txt.replace("'re", " are")
        txt = txt.replace("'m", " am")
        txt = txt.replace("'d", " would")
        txt = txt.replace("'ll", " will")

        # More cleaning
        txt = re.sub(r"review", "", txt)
        txt = re.sub(r"Review", "", txt)
        txt = re.sub(r"TripAdvisor", "", txt)
        txt = re.sub(r"reviews", "", txt)
        txt = re.sub(r"Hotel", "", txt)
        txt = re.sub(r"what's", "", txt)
        txt = re.sub(r"What's", "", txt)
        txt = re.sub(r"\'s", " ", txt)
        txt = txt.replace("pic", "picture")
        txt = re.sub(r"\'ve", " have ", txt)
        txt = re.sub(r"can't", "cannot ", txt)
        txt = re.sub(r"n't", " not ", txt)
        txt = re.sub(r"I'm", "I am", txt)
        txt = re.sub(r" m ", " am ", txt)
        txt = re.sub(r"\'re", " are ", txt)
        txt = re.sub(r"\'d", " would ", txt)
        txt = re.sub(r"\'ll", " will ", txt)
        txt = re.sub(r" e g ", " eg ", txt)

        # Remove punctuation from text
        # txt = ''.join([c for c in text if c not in punctuation])
        txt = txt.replace(".", " ")
        txt = txt.replace(":", " ")
        txt = txt.replace("!", " ")
        txt = txt.replace("&", " ")
        txt = txt.replace("#", " ")

        # Remove all symbols
        txt = re.sub(r'[^A-Za-z0-9\s]', r' ', txt)
        txt = re.sub(r'\n', r' ', txt)

        txt = re.sub(r'[0-9]', r' ', txt)

        # Replace words like sooooooo with so
        txt = ''.join(''.join(s)[:2] for _, s in itertools.groupby(txt))

        # Split attached words
        # txt = " ".join(re.findall('[A-Z][^A-Z]*', txt))

        if lowercase:
            txt = " ".join([w.lower() for w in txt.split()])

        #if remove_stops:
            # txt = " ".join([w for w in txt.split() if w not in stops])
        if stemming:
            st = PorterStemmer()
            #         print (len(txt.split()))
            #         print (txt)
            txt = " ".join([st.stem(w) for w in txt.split()])

        if lemmatization:
            wordnet_lemmatizer = WordNetLemmatizer()
            txt = " ".join([wordnet_lemmatizer.lemmatize(w, pos='v') for w in txt.split()])

        return txt


    def getSentiment(self, product_name):
        product_name = product_name.lower()
        nlpmodel = pickle.load(open("nlp_model.pkl", 'rb'))
        vectorizer = pickle.load(open("transform.pkl", 'rb'))
        #print("========Product_review_df===========")
        #print(self.Product_review_df.head())
        fetch_comb_review = self.Product_review_df.loc[self.Product_review_df['product_name'] == product_name]
        movie_vector = vectorizer.transform(fetch_comb_review.comb_review)
        pred = nlpmodel.predict(movie_vector)
        # pred = np.array([1,1,1]) (for testing)
        if len(pd.value_counts(pred).filter(items=[1])) != 0:
            ones = pd.value_counts(pred).filter(items=[1])[1]
        else:
            ones = 0
        if len(pd.value_counts(pred).filter(items=[0])) != 0:
            zeros = pd.value_counts(pred).filter(items=[0])[0]
        else:
            zeros = 0
        positive_percentage = round(ones/(ones+zeros)*100,2)
        return positive_percentage

    def getRecommendation(self, user_id , n_recom):
        # Reading the final recommondation matrix
        user_final_rating = pd.read_pickle("./user_final_rating.pkl")
        # getting top 20 elements
        recommendation = user_final_rating.loc[user_id].sort_values(ascending=False)[0:n_recom]
        # recommendation_df = pd.merge(recommendation, main_df, left_on='product_id', right_on='product_id', how='left')
        recommendation_df = pd.merge(recommendation, self.main_df, left_on='product_id', right_on='product_id', how='left')
        final_df = recommendation_df.drop_duplicates('name',keep='first')
        return final_df

    def predict_final_products(self, reviews_username):
        row = self.main_df.loc[self.main_df['reviews_username'] == reviews_username]
        user_id = row.drop_duplicates('user_id',keep = 'first').user_id.item()
        final_df = self.getRecommendation(user_id, 20)

        #print("========final_df===========")
        #print(final_df.head())
        # Getting Percentage Positive Sentiments in final Dataframe
        # final_df = final_df.assign(Percentage = lambda x: self.getSentiment(x['name']))

        final_df['percentage'] = final_df['name'].apply(lambda x:self.getSentiment(x))
        return final_df


content_engine = ContenEngine()