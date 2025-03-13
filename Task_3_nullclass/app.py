import pandas as pd 
import streamlit as st 
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

med_data = r"C:\Users\Admin\Downloads\processed_medquad.xls"
kb_data = r"C:\Users\Admin\Downloads\senior_health.xls"

@st.cache_resource
def load_data():
    return pd.read_csv(med_data)
df = load_data()

@st.cache_resource
def train_tfidf(data):
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(df['Question'])
    return tfidf, tfidf_matrix

tfidf, tfidf_matrix = train_tfidf(df)

def update_knowledge_base(med_data, kb_data, tfidf, tfidf_matrix):
    global df 
    kb_data = pd.read_csv(kb_data)
    df = pd.concat([df, kb_data], ignore_index=True).drop_duplicates().reset_index(drop=True)
    df.to_csv(med_data, index=False)

    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(df['Question'])

    return tfidf, tfidf_matrix

def get_relevant_answers(query, top_n=3):
    query_vec = tfidf.transform([query])
    similarity = cosine_similarity(query_vec,tfidf_matrix)
    top_indices = similarity[0].argsort()[-top_n:][::-1]
    return df.iloc[top_indices][['Question', 'Answer']]

st.title('Medical QnA Chatbot')
st.write('Ask any Medical Question!')

query = st.text_input('Enter your question:')
if st.button('Get Answer'):
    if query:
        results = get_relevant_answers(query)
        for i, row in results.iterrows():
            st.subheader(f"Q: {row['Question']}")
            st.write(f"A: {row['Answer']}")

    else:
        st.error('Please enter a question.')

st.write('Update knowledge Base')                                                                                    
if st.button('updat knowledge Base'):
    try:
        tfidf, tfidf_matrix = update_knowledge_base(kb_data, med_data, tfidf, tfidf_matrix )
        st.success('knowledge  base updated successfully!')
    except Exception as e:
        st.error(f'Error updating knowledge base: {e}')

        