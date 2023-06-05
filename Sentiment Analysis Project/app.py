#!/usr/bin/env python
# coding: utf-8

# In[12]:


import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from Final_P239 import mnb_model, vectorizer
import streamlit as st
import re
from sklearn.feature_extraction.text import TfidfVectorizer


# In[9]:



# Function to preprocess the tweet
def preprocess_tweet(tweet):
    # Remove any URLs
    tweet = re.sub(r'http\S+', '', tweet)
    # Remove any non-alphanumeric characters
    tweet = re.sub(r'[^\w\s]', '', tweet)
    # Remove any digits
    tweet = re.sub(r'\d+', '', tweet)
    # Remove any mentions
    tweet = re.sub(r'@\w+', '', tweet)
    
    # Tokenization
    tokens = nltk.word_tokenize(tweet)
    
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [token for token in tokens if token.lower() not in stop_words]
    
    # Lemmatization
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(token) for token in tokens]
    
    return ' '.join(tokens)


# In[ ]:





# In[10]:


# Streamlit app
def main():
    # Set the page title
    st.title("Sentiment Analysis App")
    
    # Get user input
    tweet_input = st.text_input("Enter a tweet:")
    
    if st.button("Predict"):
        if tweet_input:
            # Preprocess the input tweet text
            preprocessed_text = preprocess_tweet(tweet_input)
            
            # Apply TF-IDF transformation
            tfidf = vectorizer.transform([preprocessed_text])
            
            # Make prediction using the trained classifier
            prediction = mnb_model.predict(tfidf)[0]
            
            # Display the prediction
            st.write("Prediction:", prediction)
        else:
            st.write("Please enter a tweet.")
    
if __name__ == "__main__":
    main()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




