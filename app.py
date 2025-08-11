import streamlit as st
import pandas as pd
import gspread
from google.oauth2.service_account import Credentials
from pyvis.network import Network
import re
from collections import Counter
import tempfile
import string

# ---------- CONFIG ----------
STOPWORDS = set([
    # Common stopwords
    "a","about","above","after","again","against","all","am","an","and","any","are","as","at",
    "be","because","been","before","being","below","between","both","but","by",
    "can","did","do","does","doing","down","during","each","few","for","from","further",
    "had","has","have","having","he","her","here","hers","herself","him","himself","his",
    "how","i","i'd","i'll","i'm","i've","if","in","into","is","it","it's","its","itself",
    "let's","me","more","most","my","myself","nor","of","on","once","only","or","other",
    "ought","our","ours","ourselves","out","over","own","same","she","she'd","she'll",
    "she's","should","so","some","such","than","that","that's","the","their","theirs",
    "them","themselves","then","there","there's","these","they","they'd","they'll",
    "they're","they've","this","those","through","to","too","under","until","up","very",
    "was","we","we'd","we'll","we're","we've","were","what","what's","when","when's",
    "where","where's","which","while","who","who's","whom","why","why's","with","would",
    "you","you'd","you'll","you're","you've","your","yours","yourself","yourselves",
    # Filler words
    "um","uh","like","you know","sort of","kind of","actually","basically","literally","gonna"
])

# ---------- GOOGLE SHEETS CONNECTION ----------
@st.cache_data(ttl=600)
def load_data():
    creds = Credentials.from_service_account_info(
        st.secrets["gcp_service_account"],
        scopes=["https://www.googleapis.com/auth/spreadsheets"]
    )
    client = gspread.authorize(creds)
    sheet = client.open_by_url(st.secrets["sheet_url"])
    df = pd.DataFrame(sheet.sheet1.get_all_records())
    return df

@st.cache_data
def save_data(df):
    creds = Credentials.from_service_account_info(
        st.secrets["gcp_service_account"],
        scopes=["https://www.googleapis.com/auth/spreadsheets"]
    )
    client = gspread.authorize(creds)
    sheet = client.open_by_url(st.secrets["sheet_url"])
    sheet.sheet1.clear()
    sheet.sheet1.update([df.columns.values.tolist()] + df.values.tolist())

# ---------- KEYWORD EXTRACTION ----------
def extract_keywords(text, top_n=10):
    text = text.lower()
    text = re.sub(f"[{string.punctuation}]", " ", text)
    words = [w for w in text.split() if w not in STOPWORDS and len(w) > 2]
    counter = Counter(words)
    return [w for w, _ in counter.most_common(top_n)]

# ---------- NODE CLOUD ----------
def generate_node_cloud(df):
    net = Network(height="750px", width="100%", bgcolor="#222222", font_color="white")
    theme_groups = {theme: idx for idx, theme in enumerate(df['primary_theme'].unique())}
    colors = ["#FF6666", "#66FF66", "#6666FF", "#FFFF66", "#FF66FF", "#66FFFF"]

    for idx, row in df.iterrows():
        color = colors[theme_groups[row['primary_theme']] % len(colors)]
        net.add_node(
            idx,
            label=row['short_title'],
            title=f"Date: {row['date']}<br>Theme: {row['primary_theme']}<br>Sentiment: {row['sentiment']}",
            color=color
        )

    # Connect nodes if they share >= 2 keywords
    for i, row_i in df.iterrows():
        for j, row_j in df.iterrows():
            if i >= j:
                continue
            if len(set(row_i['keywords']).intersection(set(row_j['keywords']))) >= 2:
                net.add_edge(i, j)

    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
        net.show(tmp_file.name)
        return tmp_file.name

# ---------- STREAMLIT UI ----------
st.set_page_config(page_title="Transcript Analysis Dashboard", layout="wide")

st.title("📜 Transcript Analysis Dashboard")

# Load data
df = load_data()

# Ensure keywords column exists and is cleaned
if 'keywords' not in df.columns or any(isinstance(k, str) for k in df['keywords']):
    df['keywords'] = df['full_text'].apply(lambda txt: extract_keywords(txt))

# Page selector
page = st.sidebar.radio("Select a page", ["Node Cloud", "Transcript Table", "Transcript Detail"])

if page == "Node Cloud":
    st.subheader("3D Draggable Node Cloud (Grouped by Theme, Colored by Sentiment)")
    html_file = generate_node_cloud(df)
    st.components.v1.html(open(html_file, 'r', encoding='utf-8').read(), height=750)

elif page == "Transcript Table":
    st.subheader("Transcript Table View")
    st.dataframe(df[["date", "short_title", "word_count", "wpm", "sentiment", "keywords", "primary_theme"]])

elif page == "Transcript Detail":
    st.subheader("Transcript Detail View")
    selected_title = st.selectbox("Select Transcript", df['short_title'])
    row = df[df['short_title'] == selected_title].iloc[0]
    st.markdown(f"**Date:** {row['date']}")
    st.markdown(f"**Primary Theme:** {row['primary_theme']}")
    st.markdown(f"**Sentiment:** {row['sentiment']}")
    st.markdown(f"**Word Count:** {row['word_count']} | **WPM:** {row['wpm']}")
    st.markdown("### Keywords")
    st.write(", ".join(row['keywords']))
    st.markdown("### Summary")
    st.write(row['summary'])
    st.markdown("### Full Transcript")
    st.write(row['full_text'])
