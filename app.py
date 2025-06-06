import streamlit as st
import pickle
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

ps = PorterStemmer()


def transformation_of_text(text):
    # convert text into lower case
    text = text.lower()

    # create word tokens
    text = word_tokenize(text)  # Now works!

    # remove special characters
    y = []
    for t in text:
        if t.isalnum():
            y.append(t)

    # remove stopwords
    text = y[:]
    y.clear()
    for t in text:
        if t not in stopwords.words('english'):
            y.append(t)

    # now proceed with word stemming
    text = y[:]
    y.clear()
    for t in text:
        y.append(ps.stem(t))

    return " ".join(y)

tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
model = pickle.load(open('model.pkl', 'rb'))

st.title("SMS Spam Classifier")

input_sms = st.text_area("Enter the SMS")

if st.button("Predict"):
    # 1. Preprocess the sms
    transformed_sms = transformation_of_text(input_sms)

    # 2. Vectorization
    vector_input = tfidf.transform([transformed_sms])

    # 3. Predict
    result = model.predict(vector_input)[0]

    # 4. Display the output
    if result == 1:
        st.header("Spam")
    else:
        st.header("Not Spam")
