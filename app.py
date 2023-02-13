import streamlit as st
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle
import numpy as np

model = tf.keras.models.load_model('model.h5')

with open('tokenizer.pkl', 'rb') as f:
    tokenizer = pickle.load(f)

with open('max_sequence_len.pkl', 'rb') as f:
    max_sequence_len = pickle.load(f)


next_words = 100

st.title("Shakespeare Style Text Generator")

seed_text = st.text_area("Enter the text")


if st.button('Generate'):
    for _ in range(next_words):
        # Convert the text into sequences
        token_list = tokenizer.texts_to_sequences([seed_text])[0]
        # Pad the sequences
        token_list = pad_sequences([token_list], maxlen=max_sequence_len - 1, padding='pre')
        # Get the probabilities of predicting a word
        predicted = model.predict(token_list, verbose=0)
        # Choose the next word based on the maximum probability
        predicted = np.argmax(predicted, axis=-1).item()
        # Get the actual word from the word index
        output_word = tokenizer.index_word[predicted]
        # Append to the current text
        seed_text += " " + output_word
    st.markdown(seed_text)