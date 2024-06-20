import streamlit as st
import pickle
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

# Ensure NLTK resources are downloaded
nltk.download('punkt')
nltk.download('stopwords')

# Initialize NLTK objects
ps = PorterStemmer()

# Function to preprocess text
def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)
    
    y = []
    for i in text:
        if i.isalnum():
            y.append(i)
    
    text = y[:]
    y.clear()
    
    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(ps.stem(i))
            
    return ' '.join(y)  # Return preprocessed text as a single string

# Load TF-IDF vectorizer and model
tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
model = pickle.load(open('model.pkl','rb'))
# Streamlit app
st.title("Email Spam Detection")
input_sms = st.text_input("Enter the message")
if st.button('Predict'):
    # Preprocess input
    transformed_sms = transform_text(input_sms)
    
    # Vectorize input
    vector_input = tfidf.transform([transformed_sms])  # Pass a list with single transformed message
    
    # Predict
    result = model.predict(vector_input)[0]

    # Display prediction
    if result == 1:
        st.header("Spam")
    else:
        st.header("Legit")
