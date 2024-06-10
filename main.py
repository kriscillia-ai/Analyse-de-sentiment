import pandas as pd
import streamlit as st
from textblob import TextBlob
from polyglot.text import Text

st.header('Analyse de sentiment')
lang = st.selectbox('Sélectionnez la langue', ['en', 'fr'])

with st.expander('Analyse du text'):
    text = st.text_input('Text ici: ')
    if text:
        if lang == 'fr':
            text_polyglot = Text(text, hint_language_code='fr')
            sentiment_score = text_polyglot.polarity
            st.write('Polarité: ', round(sentiment_score, 2))
            st.write('Subjectivité: ', 'Non disponible')
        else:
            blob = TextBlob(text)
            st.write('Polarité: ', round(blob.sentiment.polarity, 2))
            st.write('Subjectivité: ', round(blob.sentiment.subjectivity, 2))

with st.expander('Analyze CSV'):
    upl = st.file_uploader('Téléchargement des fichiers')

    def score(x, lang='en'):
        if lang == 'fr':
            text_polyglot = Text(x, hint_language_code='fr')
            return text_polyglot.polarity
        else:
            blob1 = TextBlob(x)
            return blob1.sentiment.polarity

    def analyze(x):
        if x >= 0.5:
            return 'Positive'
        elif x <= -0.5:
            return 'Negative'
        else:
            return 'Neutral'

    if upl:
        df = pd.read_excel(upl)
        del df['Unnamed: 0']
        df['score'] = df['tweets'].apply(lambda x: score(x, lang))
        df['analysis'] = df['score'].apply(analyze)
        st.write(df.head(10))

        @st.cache
        def convert_df(df):
            return df.to_csv().encode('utf-8')

        csv = convert_df(df)

        st.download_button(
            label="Download data as CSV",
            data=csv,
            file_name='sentiment.csv',
            mime='text/csv',
        )
