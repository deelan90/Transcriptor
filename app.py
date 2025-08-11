import streamlit as st
import pandas as pd
import json
from pyvis.network import Network
import streamlit.components.v1 as components
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import Counter
import spacy
conn = st.connection("gsheets")


# --- Page Configuration ---
st.set_page_config(page_title="Transcript Analyzer", layout="wide")

# --- Google Sheets Connection ---
conn = st.connection("gsheets")

# --- NLP Model Loading ---
# Load the spaCy model once
@st.cache_resource
def load_spacy_model():
    return spacy.load("en_core_web_sm")

nlp = load_spacy_model()


# --- Data Loading ---
@st.cache_data(ttl=600)  # Cache data for 10 minutes
def load_data():
    """Loads the transcript data from the Google Sheet."""
    try:
        df = conn.read(worksheet="Sheet1", usecols=list(range(9)), ttl=5)
        df = df.dropna(how="all") # Drop empty rows that might be read
        return df
    except Exception as e:
        st.error(f"Failed to load data from Google Sheets: {e}")
        return pd.DataFrame()


# --- Keyword Extraction Logic (Improved) ---
# CORRECTED STOPWORDS LIST: The '...' has been replaced with the full list.
STOPWORDS = set([
    'a', 'about', 'above', 'after', 'again', 'against', 'all', 'am', 'an', 'and', 'any', 'are', "aren't", 'as', 'at',
    'be', 'because', 'been', 'before', 'being', 'below', 'between', 'both', 'but', 'by', 'can', "can't", 'cannot',
    'com', 'could', "couldn't", 'did', "didn't", 'do', 'does', "doesn't", 'doing', 'don', "don't", 'down', 'during',
    'each', 'else', 'ever', 'few', 'for', 'from', 'further', 'get', 'had', "hadn't", 'has', "hasn't", 'have', "haven't",
    'having', 'he', "he'd", "he'll", "he's", 'her', 'here', "here's", 'hers', 'herself', 'him', 'himself', 'his', 'how',
    "how's", 'http', 'i', "i'd", "i'll", "i'm", "i've", 'if', 'in', 'into', 'is', "isn't", 'it', "it's", 'its', 'itself',
    'just', 'k', "let's", 'like', 'me', 'more', 'most', "mustn't", 'my', 'myself', 'no', 'nor', 'not', 'of', 'off', 'on',
    'once', 'only', 'or', 'other', 'ought', 'our', 'ours', 'ourselves', 'out', 'over', 'own', 'r', 'said', 'same', 'say',
    'says', 'shan', "shan't", 'she', "she'd", "she'll", "she's", 'should', "shouldn't", 'so', 'some', 'such', 'than',
    'that', "that's", 'the', 'their', 'theirs', 'them', 'themselves', 'then', 'there', "there's", 'these', 'they',
    "they'd", "they'll", "they're", "they've", 'this', 'those', 'through', 'to', 'too', 'under', 'until', 'up',
    'very', 'was', "wasn't", 'we', "we'd", "we'll", "we're", "we've", 'were', "weren't", 'what', "what's", 'when',
    "when's", 'where', "where's", 'which', 'while', 'who', "who's", 'whom', 'why', "why's", 'with', 'won', "won't",
    'would', "wouldn't", 'www', 'you', "you'd", "you'll", "you're", "you've", 'your', 'yours', 'yourself',
    'yourselves', 'um', 'uh', 'ah', 'er', 'like', 'you know', 'actually', 'basically', 'so', 'yeah', 'okay', 'right',
    'well', 'truly', 'literally', 'honestly', 'gonna', 'wanna', 'gotta', 'kinda', 'sorta'
])

def preprocess_text(text):
    if not isinstance(text, str): return ""
    text = text.lower()
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    tokens = [word for word in text.split() if word not in STOPWORDS and len(word) > 1]
    return " ".join(tokens)

def extract_keywords(text, max_keywords=7):
    processed_text = preprocess_text(text)
    if not processed_text.strip(): return []

    try:
        vectorizer = TfidfVectorizer(max_features=20, stop_words='english', ngram_range=(1, 2))
        tfidf_matrix = vectorizer.fit_transform([processed_text])
        feature_names = vectorizer.get_feature_names_out()
        
        word_counts = Counter(processed_text.split())
        meaningful_words = [word for word, count in word_counts.items() if count > 1]
        
        keywords = [kw for kw in feature_names if kw in meaningful_words]
        return keywords[:max_keywords]
    except ValueError:
        return []

# --- Main Application Logic ---
st.title("Transcript Analysis Dashboard 📊")

# Load data once at the start
df = load_data()

if df.empty:
    st.warning("No data found in the Google Sheet. Please add some transcripts to get started.")
else:
    page = st.sidebar.selectbox("Choose a Page", ["3D Node Cloud", "Transcript Table", "Detailed View"])

    if page == "3D Node Cloud":
        st.header("Transcript Node Cloud")
        st.markdown("Nodes are transcripts, colored by sentiment. They are connected if they share at least two key keywords.")

        net = Network(height="750px", width="100%", bgcolor="#222222", font_color="white", notebook=True, cdn_resources='in_line')
        net.force_atlas_2based(gravity=-45)

        sentiment_color_map = {"Positive": "#2ecc71", "Negative": "#e74c3c", "Neutral": "#3498db"}
        
        df['keywords_list'] = df['keywords'].apply(lambda x: json.loads(x) if isinstance(x, str) and x.startswith('[') else [])

        for index, row in df.iterrows():
            node_title = (
                f"<b>Date:</b> {row['date']}<br>"
                f"<b>Title:</b> {row['short_title']}<br>"
                f"<b>Theme:</b> {row['primary_theme']}<br>"
                f"<b>Sentiment:</b> {row['sentiment']}"
            )
            net.add_node(
                index,
                label=row['short_title'],
                title=node_title,
                color=sentiment_color_map.get(row['sentiment'], 'grey'),
                size=max(10, row['word_count'] / 100) # Ensure a minimum size
            )

        for i in range(len(df)):
            for j in range(i + 1, len(df)):
                shared_keywords = set(df.at[i, 'keywords_list']) & set(df.at[j, 'keywords_list'])
                if len(shared_keywords) >= 2:
                    net.add_edge(i, j, weight=len(shared_keywords), title=", ".join(shared_keywords))
        
        try:
            net.save_graph('pyvis_graph.html')
            with open('pyvis_graph.html', 'r', encoding='utf-8') as f:
                html_file = f.read()
            components.html(html_file, height=800, scrolling=True)
        except Exception as e:
            st.error(f"Could not generate graph: {e}")

    # RESTORED MISSING PAGES
    elif page == "Transcript Table":
        st.header("Transcripts Table")
        st.write("Use the filters below to explore your transcripts.")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            sentiment_filter = st.multiselect("Sentiment", options=df['sentiment'].unique(), default=df['sentiment'].unique())
        with col2:
            theme_filter = st.text_input("Filter by keyword or theme")
        with col3:
            min_words = st.number_input("Minimum Word Count", value=0, step=50)

        filtered_df = df[df['sentiment'].isin(sentiment_filter) & (df['word_count'] >= min_words)]
        if theme_filter:
            filtered_df = filtered_df[
                filtered_df['keywords'].str.contains(theme_filter, na=False, case=False) |
                filtered_df['primary_theme'].str.contains(theme_filter, na=False, case=False)
            ]
        st.dataframe(filtered_df.sort_values(by='date', ascending=False))

    elif page == "Detailed View":
        st.header("Transcript Detail")
        if not df.empty:
            # Use a unique identifier for the selectbox key
            selected_date = st.selectbox("Select a transcript by date", options=df['date'].unique(), key="transcript_selector")
            record = df[df['date'] == selected_date].iloc[0]

            st.subheader(record['short_title'])
            st.markdown(f"**Date:** {record['date']} | **Word Count:** {record['word_count']} | **WPM:** {record['wpm']:.2f} | **Sentiment:** {record['sentiment']}")

            try:
                keywords = json.loads(record['keywords'])
                st.markdown(f"**Keywords:** {', '.join(keywords)}")
            except (json.JSONDecodeError, TypeError):
                st.markdown("**Keywords:** Not available")

            st.markdown("---")
            st.markdown("#### Action Items")
            try:
                action_items = json.loads(record['action_items'])
                if action_items:
                    for item in action_items:
                        st.markdown(f"- {item}")
                else:
                    st.write("No action items were extracted.")
            except (json.JSONDecodeError, TypeError):
                st.write("Could not parse action items.")

            st.markdown("---")
            st.markdown("#### Full Transcript")
            st.text_area("Transcript", value=record['full_text'], height=400, key="full_transcript_text")
        else:
            st.warning("No transcripts available to display.")
