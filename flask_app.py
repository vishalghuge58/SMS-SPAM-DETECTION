from flask import Flask,render_template,redirect,request

import pickle
import nltk
from nltk.corpus import stopwords
import string

from nltk.stem.porter import PorterStemmer
stemmer = PorterStemmer()

nltk.download('stopwords')
nltk.download('punkt')
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


app = Flask(__name__)


@app.route('/' )
def home():

    return render_template('index.html')

@app.route('/result',methods = ['GET' , 'POST'])
def result():
    user_text = request.form.get('user_text')

    # 1. preprocess
    transformed_sms = transform_text(str(user_text))
    # 2. vectorize
    vector_input = tfidf.transform([transformed_sms])
    # 3. predict
    result = model.predict(vector_input)[0]

    if result == 1:
        output = "Spam"
    else:
        output = "Not Spam"



    return render_template('index.html' , result = output)

if __name__ == '__main__':

    app.run( port=8080 , host='0.0.0.0')