import pandas as pd
import re

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
nltk.download('stopwords')
nltk.download('punkt')

class ContentFiltering:
    def __init__(self, data_file):
        self.data = pd.read_csv(data_file)
        
    def preprocess_text(self, text):
        text = re.sub(r'<.*?>', '', text)  # Remove HTML tags
        text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
        text = text.lower()  # Convert to lowercase
        return text

    def preprocess(self, text):
        if isinstance(text, str):  # Check if the input is a string
            text = self.preprocess_text(text)
            stop_words = set(stopwords.words('english'))
            words = word_tokenize(text)
            words = [word for word in words if word not in stop_words]
            
            # Stemming using Porter Stemmer
            stemmer = PorterStemmer()
            words = [stemmer.stem(word) for word in words]
            
            return ' '.join(words)
        else:
            return ''  # Return an empty string for non-string inputs
    
    def content_recommender(self, movie_title,n):
        idx = self.data[self.data['title'].str.lower() == movie_title.lower()].index[0]        
        # Apply preprocessing to the overview column
        self.data['cleaned_overview'] = self.data['overview'].apply(self.preprocess)
        
        tfidf = TfidfVectorizer(stop_words='english')
        tfidf_matrix = tfidf.fit_transform(self.data['cleaned_overview'])
        self.data['cleaned_overview'] = self.data['cleaned_overview'].fillna('')
        cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)
        sim_scores = list(enumerate(cosine_sim[idx]))
        # sim_scores = sim_scores[1:11] if n > 10 else sim_scores[1:n+1]
        sim_scores = sorted(sim_scores,key=lambda x: x[1], reverse=True)[:n+1]
        movie_indices = [i[0] for i in sim_scores]
        recommended_movies = self.data.loc[movie_indices,['title','release_date']][1:]
        return recommended_movies
