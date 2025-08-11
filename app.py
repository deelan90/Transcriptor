import streamlit as st
import pandas as pd
import json
from pyvis.network import Network
import streamlit.components.v1 as components
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import Counter
import spacy
from streamlit_gsheets import GSheetsConnection

# --- Page Configuration ---
st.set_page_config(page_title="Transcript Analyzer", layout="wide")

# --- Google Sheets Connection ---
# Note: You'll set up the connection details in Streamlit's secrets management
conn = st.connection("gsheets", type=GSheetsConnection)

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
STOPWORDS = set(['a', 'about', 'above', ...]) # Add all your stopwords here

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
        
        # Count frequencies of non-stop words to find meaningful terms
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

    # ... (rest of the pages: Transcript Table and Detailed View - no changes needed for these)
