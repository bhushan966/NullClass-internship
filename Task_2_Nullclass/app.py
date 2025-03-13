import pandas as pd 
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

#load the dataset 
df = pd.read_csv(r'C:\Users\Admin\ncchatbot\processed_medquad.xls')
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(df['Question'])

def get_relevant_answers(query, top_n=3):
    query_vec = tfidf.transform([query])
    similarity = cosine_similarity(query_vec, tfidf_matrix)
    top_indices = similarity[0].argsort()[-top_n:][::-1]
    return df.iloc[top_indices][['Question', 'Answer']]

#streamlit interface 

st.title('Medical QnA Chatbot')
st.write('Ask any medical Question!')

query = st.text_input('Enter your question:')
if st.button('Get Answer'):
    if query:
        results = get_relevant_answers(query)
        for i, row in results.iterrows():
            st.subheader(f"Q: {row['Question']}")
            st.write(f"A: {row['Answer']}")
    else:
        st.error('Please enter question.')
