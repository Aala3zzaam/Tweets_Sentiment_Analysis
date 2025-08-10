import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline

st.set_page_config(layout="wide")
st.markdown("""
    <style>
    div[data-testid="stApp"] {
        background-color: #061729;
        color: white;
    }
    label, span, p {
        color: white !important;
    }
    textarea {
        color: white !important;
        background-color: #0b2239 !important;
        border: 1.5px solid #0d3b66;
        border-radius: 8px;
        padding: 10px;
        font-size: 16px;
    }
    ::placeholder {
        color: #cccccc !important;
        opacity: 1;
    }
    div.stButton > button {
        background-color: #0d3b66;
        color: white;
        border: none;
        padding: 0.5em 1.5em;
        border-radius: 8px;
        font-weight: bold;
        cursor: pointer;
        transition: background-color 0.3s ease;
    }
    div.stButton > button:hover {
        background-color: #092b4c;
        color: white;
    }
    </style>
""", unsafe_allow_html=True)


model_path = "./models/Twitter-Sentiment-Analysis-BERT"
token_path = './models/Twitter-Sentiment-Analysis-BERT-Tokenizer'


@st.cache_resource
def load_model():
    model = AutoModelForSequenceClassification.from_pretrained(model_path, local_files_only=True)
    tokenizer = AutoTokenizer.from_pretrained(token_path)
    return model, tokenizer


model, tokenizer = load_model()

sentiment_pipeline = pipeline(
    "sentiment-analysis",
    model=model,
    tokenizer=tokenizer,
    device=-1  # use CPU
)

st.markdown(
    """
    <h1 style='color: #ffffff; text-align: center; 
               text-shadow: 2px 2px 4px #000000;'>Tweets Sentiment Analyzer</h1>
    """,
    unsafe_allow_html=True
)

col1, col2, col3 = st.columns([1, 2, 1])

with col2:
    st.markdown("🖋 Enter a tweet or short sentence:")
    tweet = st.text_area(
        "",
        height=150,
        placeholder="Type your tweet here...",
        help="Enter a tweet or a short sentence to analyze its sentiment."
    )

    if st.button("🔍 Analyze") and tweet.strip():
        with st.spinner("Analyzing..."):
            result = sentiment_pipeline(tweet)[0]

        label_map = {
            "LABEL_0": "Irrelevant",
            "LABEL_1": "Negative",
            "LABEL_2": "Neutral",
            "LABEL_3": "Positive"
        }
        color_map = {
            "Irrelevant": "gray",
            "Negative": "red",
            "Neutral": "orange",
            "Positive": "green"
        }

        prediction = label_map.get(result['label'], result['label'])
        score = result['score']

        st.markdown(
            f"<h2 style='color: {color_map.get(prediction, 'black')}; text-align:center;'>{prediction}</h2>",
            unsafe_allow_html=True
        )
        st.write(f"Confidence: {score:.2%}")
