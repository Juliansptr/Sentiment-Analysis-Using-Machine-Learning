from flask import Flask, request, render_template

app = Flask(__name__)

from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
import string
import pandas as pd
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
import re
nltk.download('punkt')
nltk.download('wordnet')

dictionary = {0 : "Negative", 1 : "Positive"}

lemmer = nltk.stem.WordNetLemmatizer()

def LemTokens(tokens):
  return [lemmer.lemmatize(token) for token in tokens]

remove_punct_dict = dict((ord(punct), None) for punct in string.punctuation)

def LemNormalize(text):
  return LemTokens(nltk.word_tokenize(text.lower().translate(remove_punct_dict)))
  

@app.route("/")
def index():
    return render_template('index.html')
    
@app.route("/result", methods=['POST'])
def result():
    kalimat = request.form["review"]
    tfidfvectorizer = pickle.load(open('tfidfvectorizer_19k.pkl', 'rb'))
    x_transform = tfidfvectorizer.transform([kalimat])
    x_transform = x_transform.toarray()
    model = pickle.load(open('model_sgd.pkl', 'rb'))
    hasil = model.predict(x_transform)

    return render_template('index.html', prediction=dictionary[hasil[0]], kalimat=kalimat)

        
if __name__ == "__main__":
    app.run(debug=True)