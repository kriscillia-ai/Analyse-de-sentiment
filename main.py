from textblob import TextBlob
import pandas as pd
import streamlit as st

st.header('Analyse de sentiment')
with st.expander('Analyse du text'):
    text = st.text_input('Text ici: ')
    if text:
        blob = TextBlob(text)
        st.write('Polarité: ', round(blob.sentiment.polarity,2))
        st.write('Subjectivité: ', round(blob.sentiment.subjectivity,2))

with st.expander('Analyze CSV'):
    upl = st.file_uploader('Téléchargement des fichiers')

    def score(x):
        blob1 = TextBlob(x)
        return blob1.sentiment.polarity

#
    def analyze(x):
        if x >= 0.5:
            return 'Positive'
        elif x <= -0.5:
            return 'Negative'
        else:
            return 'Neutral'

#
    if upl:
        df = pd.read_excel(upl)
        del df['Unnamed: 0']
        df['score'] = df['tweets'].apply(score)
        df['analysis'] = df['score'].apply(analyze)
        st.write(df.head(10))

        @st.cache
        def convert_df(df):
            # IMPORTANT: Cache the conversion to prevent computation on every rerun
            return df.to_csv().encode('utf-8')

        csv = convert_df(df)

        st.download_button(
            label="Download data as CSV",
            data=csv,
            file_name='sentiment.csv',
            mime='text/csv',
        )
