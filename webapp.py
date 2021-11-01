import streamlit as st
import pandas as pd
import numpy as np
from functions import clean_doc, get_topic
import pickle
import sklearn

st.title('NLP Text Modeling')

prediction = 0

def user_input_features():
    input_text = st.text_area("text input", value='', height=400, max_chars=None, key=None)

    return input_text

input_text = user_input_features()


model = pickle.load( open( "nlp_topic_moeling.pickle", "rb" ) )

input_text = clean_doc(input_text)



prediction = model.predict([input_text])

st.write( get_topic(prediction) )



