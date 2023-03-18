import streamlit as st
import pickle
import nltk
from nltk.corpus import stopwords
import string

from nltk.stem.porter import PorterStemmer
stemmer = PorterStemmer()


stop_words = stopwords.words('english')  # it gives list of stopwords in english

punctuations = string.punctuation       # it gives list of all punctuations



def transform_text(text):
    
    text = text.lower()
    
    text = nltk.word_tokenize(text)
    
    text = [word for word in text if word.isalnum()]  # for removing special charecters
    
    text = [word for word in text if word not in stop_words and word not in punctuations]
    
    text = [stemmer.stem(word) for word in text]
    
    return " ".join(text)

tfidf = pickle.load(open('vectorizer.pkl','rb'))
model = pickle.load(open('model.pkl','rb'))

st.title("Email/SMS Spam Classifier")

input_sms = st.text_area("Enter the message")

if st.button('Predict'):

    # 1. preprocess
    transformed_sms = transform_text(input_sms)
    # 2. vectorize
    vector_input = tfidf.transform([transformed_sms])
    # 3. predict
    result = model.predict(vector_input)[0]
    # 4. Display
    if result == 1:
        st.header("Spam")
    else:
        st.header("Not Spam")