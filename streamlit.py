import streamlit as st
import PyPDF2
import re
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize
from textblob import TextBlob
from wordcloud import WordCloud
from sklearn.feature_extraction.text import TfidfVectorizer
from gensim import corpora, models
import matplotlib.pyplot as plt
import seaborn as sns
import pypdf
import io
import base64
from collections import Counter
import os

# Set page configuration
st.set_page_config(
    page_title="NLP PDF Analyzer",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #ff7f0e;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    </style>
""", unsafe_allow_html=True)

# Download NLTK data
@st.cache_resource
def download_nltk_data():
    try:
        nltk.data.find('tokenizers/punkt')
        nltk.data.find('corpora/stopwords')
    except LookupError:
        nltk.download('punkt', quiet=True)
        nltk.download('punkt_tab', quiet=True)
        nltk.download('stopwords', quiet=True)

download_nltk_data()

# Initialize session state
if 'processed' not in st.session_state:
    st.session_state.processed = False
if 'df' not in st.session_state:
    st.session_state.df = None
if 'sent_df' not in st.session_state:
    st.session_state.sent_df = None
if 'word_freq' not in st.session_state:
    st.session_state.word_freq = None
if 'topic_df' not in st.session_state:
    st.session_state.topic_df = None

# Text preprocessing function
@st.cache_data
def clean_text(text):
    stop_words = set(stopwords.words('english'))
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    tokens = [w for w in text.split() if w not in stop_words and len(w) > 2]
    return " ".join(tokens)

# PDF extraction function
@st.cache_data
def extract_pdf_text(pdf_path):
    try:
        reader = pypdf.PdfReader(pdf_path)
        text_pages = []
        
        for i, page in enumerate(reader.pages):
            text = page.extract_text()
            if text:
                text_pages.append({'page_num': i+1, 'text': text})
        
        return pd.DataFrame(text_pages)
    except Exception as e:
        st.error(f"Error reading PDF: {str(e)}")
        return None

# Sentiment analysis function
@st.cache_data
def analyze_sentiment(df):
    sentences = []
    for page in df['text']:
        for sent in sent_tokenize(page):
            blob = TextBlob(sent)
            sentences.append({
                'sentence': sent,
                'polarity': blob.sentiment.polarity,
                'subjectivity': blob.sentiment.subjectivity
            })
    return pd.DataFrame(sentences)

# Topic modeling function
@st.cache_data
def perform_topic_modeling(texts, num_topics=10):
    tokenized_docs = [doc.split() for doc in texts]
    dictionary = corpora.Dictionary(tokenized_docs)
    corpus = [dictionary.doc2bow(text) for text in tokenized_docs]
    
    lda_model = models.LdaModel(
        corpus=corpus,
        id2word=dictionary,
        num_topics=num_topics,
        random_state=42,
        passes=10,
        update_every=1,
        chunksize=100,
        alpha='auto',
        eta='auto'
    )
    
    topics = lda_model.show_topics(num_topics=num_topics, num_words=8, formatted=False)
    topic_data = []
    for idx, topic in topics:
        topic_words = ", ".join([w for w, _ in topic])
        topic_data.append({"Topic": f"Topic {idx + 1}", "Words": topic_words})
    
    return pd.DataFrame(topic_data)

# Auto-process PDF on first load
def auto_process_pdf():
    pdf_path = "report.pdf"
    
    if not os.path.exists(pdf_path):
        st.error("❌ Error: 'report.pdf' not found in the application directory.")
        st.stop()
    
    # Extract text
    st.session_state.df = extract_pdf_text(pdf_path)
    
    if st.session_state.df is not None:
        # Clean text
        st.session_state.df['clean_text'] = st.session_state.df['text'].apply(clean_text)
        
        # Sentiment analysis
        st.session_state.sent_df = analyze_sentiment(st.session_state.df)
        
        # Word frequency
        all_words = []
        for t in st.session_state.df['clean_text']:
            all_words.extend(t.split())
        st.session_state.word_freq = pd.Series(all_words).value_counts().head(20)
        
        # Topic modeling
        st.session_state.topic_df = perform_topic_modeling(
            st.session_state.df['clean_text'].tolist(), 
            num_topics=10
        )
        
        st.session_state.processed = True

# Main app
def main():
    st.markdown('<h1 class="main-header">📄 NLP PDF Analysis Dashboard</h1>', unsafe_allow_html=True)
    
    # Auto-process PDF if not already processed
    if not st.session_state.processed:
        with st.spinner("🔄 Analyzing PDF... Please wait..."):
            auto_process_pdf()
    
    # Main content - Tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Overview", 
        "😊 Sentiment Analysis", 
        "📝 Word Analysis", 
        "🎯 Topic Modeling",
        "💾 Export Data"
    ])
    
    with tab1:
        st.markdown('<h2 class="sub-header">Document Overview</h2>', unsafe_allow_html=True)
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Pages", len(st.session_state.df))
        with col2:
            st.metric("Total Sentences", len(st.session_state.sent_df))
        with col3:
            total_words = sum([len(t.split()) for t in st.session_state.df['clean_text']])
            st.metric("Total Words", f"{total_words:,}")
        with col4:
            unique_words = len(set([w for t in st.session_state.df['clean_text'] for w in t.split()]))
            st.metric("Unique Words", f"{unique_words:,}")
        
        st.markdown("---")
        
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Sample Text (First Page)")
            st.text_area("", st.session_state.df['text'].iloc[0][:500] + "...", height=200, key="sample1")
        
        with col2:
            st.subheader("Cleaned Text (First Page)")
            st.text_area("", st.session_state.df['clean_text'].iloc[0][:500] + "...", height=200, key="sample2")
    
    with tab2:
        st.markdown('<h2 class="sub-header">Sentiment Analysis</h2>', unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns(3)
        with col1:
            avg_polarity = st.session_state.sent_df['polarity'].mean()
            st.metric("Average Polarity", f"{avg_polarity:.4f}")
        with col2:
            avg_subjectivity = st.session_state.sent_df['subjectivity'].mean()
            st.metric("Average Subjectivity", f"{avg_subjectivity:.4f}")
        with col3:
            positive_pct = (st.session_state.sent_df['polarity'] > 0).sum() / len(st.session_state.sent_df) * 100
            st.metric("Positive Sentences", f"{positive_pct:.1f}%")
        
        st.markdown("---")
        
        # Sentiment distribution plot
        fig, ax = plt.subplots(figsize=(10, 5))
        sns.histplot(st.session_state.sent_df['polarity'], bins=30, kde=True, color='skyblue', ax=ax)
        ax.set_title("Sentiment Polarity Distribution", fontsize=16, fontweight='bold')
        ax.set_xlabel("Polarity", fontsize=12)
        ax.set_ylabel("Sentence Count", fontsize=12)
        plt.tight_layout()
        st.pyplot(fig)
        
        # Sentiment statistics
        st.subheader("Sentiment Statistics")
        st.dataframe(st.session_state.sent_df['polarity'].describe(), use_container_width=True)
        
        # Most positive and negative sentences
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Most Positive Sentences")
            top_positive = st.session_state.sent_df.nlargest(5, 'polarity')
            for idx, row in top_positive.iterrows():
                st.info(f"**Polarity: {row['polarity']:.3f}**\n\n{row['sentence'][:200]}...")
        
        with col2:
            st.subheader("Most Negative Sentences")
            top_negative = st.session_state.sent_df.nsmallest(5, 'polarity')
            for idx, row in top_negative.iterrows():
                st.warning(f"**Polarity: {row['polarity']:.3f}**\n\n{row['sentence'][:200]}...")
    
    with tab3:
        st.markdown('<h2 class="sub-header">Word Analysis</h2>', unsafe_allow_html=True)
        
        # Word frequency bar chart
        fig, ax = plt.subplots(figsize=(12, 6))
        sns.barplot(
            x=st.session_state.word_freq.values, 
            y=st.session_state.word_freq.index, 
            palette='viridis',
            ax=ax,
            hue=st.session_state.word_freq.index,
            legend=False
        )
        ax.set_title("Top Frequent Words", fontsize=16, fontweight='bold')
        ax.set_xlabel("Frequency", fontsize=12)
        ax.set_ylabel("Words", fontsize=12)
        plt.tight_layout()
        st.pyplot(fig)
        
        st.markdown("---")
        
        # Word cloud
        st.subheader("Word Cloud Visualization")
        all_words = []
        for t in st.session_state.df['clean_text']:
            all_words.extend(t.split())
        
        wc = WordCloud(
            width=1200, 
            height=600, 
            background_color='white',
            colormap='viridis'
        ).generate(" ".join(all_words))
        
        fig, ax = plt.subplots(figsize=(15, 8))
        ax.imshow(wc, interpolation='bilinear')
        ax.axis('off')
        ax.set_title("Word Cloud - Document Analysis", fontsize=18, fontweight='bold', pad=20)
        plt.tight_layout()
        st.pyplot(fig)
        
        # Word frequency table
        st.subheader("Word Frequency Table")
        st.dataframe(
            st.session_state.word_freq.reset_index().rename(columns={'index': 'Word', 'count': 'Frequency'}),
            use_container_width=True
        )
    
    with tab4:
        st.markdown('<h2 class="sub-header">Topic Modeling (LDA)</h2>', unsafe_allow_html=True)
        
        # Topic visualization
        fig, ax = plt.subplots(figsize=(12, 6))
        topic_nums = [f"T{i+1}" for i in range(len(st.session_state.topic_df))]
        word_counts = [len(words.split(",")) for words in st.session_state.topic_df["Words"]]
        
        sns.barplot(
            x=topic_nums, 
            y=word_counts, 
            palette="rocket",
            ax=ax,
            hue=topic_nums,
            legend=False
        )
        ax.set_title("LDA Topic Word Distribution", fontsize=16, fontweight='bold')
        ax.set_xlabel("Topics", fontsize=12)
        ax.set_ylabel("Number of Words", fontsize=12)
        plt.tight_layout()
        st.pyplot(fig)
        
        st.markdown("---")
        
        # Topic details
        st.subheader("Discovered Topics")
        for idx, row in st.session_state.topic_df.iterrows():
            with st.expander(f"🎯 {row['Topic']}", expanded=(idx < 3)):
                st.write(f"**Keywords:** {row['Words']}")
    
    with tab5:
        st.markdown('<h2 class="sub-header">Export Data</h2>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Download CSV Files")
            
            # Sentiment data
            csv1 = st.session_state.sent_df.to_csv(index=False)
            st.download_button(
                label="📥 Download Sentiment Analysis",
                data=csv1,
                file_name="sentence_sentiment.csv",
                mime="text/csv",
                use_container_width=True
            )
            
            # Cleaned text
            csv2 = st.session_state.df.to_csv(index=False)
            st.download_button(
                label="📥 Download Cleaned Text",
                data=csv2,
                file_name="cleaned_text.csv",
                mime="text/csv",
                use_container_width=True
            )
            
            # Word frequency
            csv3 = st.session_state.word_freq.to_csv()
            st.download_button(
                label="📥 Download Word Frequency",
                data=csv3,
                file_name="word_frequency.csv",
                mime="text/csv",
                use_container_width=True
            )
            
            # Topics
            csv4 = st.session_state.topic_df.to_csv(index=False)
            st.download_button(
                label="📥 Download Topics",
                data=csv4,
                file_name="lda_topics.csv",
                mime="text/csv",
                use_container_width=True
            )
        
        with col2:
            st.subheader("Data Preview")
            data_choice = st.selectbox(
                "Select data to preview:",
                ["Sentiment Analysis", "Cleaned Text", "Word Frequency", "Topics"]
            )
            
            if data_choice == "Sentiment Analysis":
                st.dataframe(st.session_state.sent_df.head(10), use_container_width=True)
            elif data_choice == "Cleaned Text":
                st.dataframe(st.session_state.df.head(10), use_container_width=True)
            elif data_choice == "Word Frequency":
                st.dataframe(st.session_state.word_freq.head(10), use_container_width=True)
            else:
                st.dataframe(st.session_state.topic_df, use_container_width=True)

if __name__ == "__main__":
    main()