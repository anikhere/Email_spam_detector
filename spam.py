import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import string

# Download NLTK data (runs only once per session)
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)
nltk.download('stopwords', quiet=True)

# Preprocessing function (fixed to remove unnecessary '==> ' prefix)
def preprocessing(text):
    text = text.lower()
    tokens = word_tokenize(text)
    final = []
    for i in tokens:
        if i.isalnum() and i not in stopwords.words('english') and i not in string.punctuation:
            final.append(i)
    text = final[:]
    final.clear()
    ps = PorterStemmer()
    for j in text:
        final.append(ps.stem(j))
    return ' '.join(final)

# Load and preprocess data (assuming spam.csv is in the same directory)
@st.cache_data  # Cache to avoid reloading on every interaction
def load_and_prepare_data():
    df = pd.read_csv('spam.csv', encoding="ISO-8859-1")
    df.drop(columns=['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'], inplace=True)
    df.rename(columns={'v1': 'Result', 'v2': 'Message'}, inplace=True)
    le = LabelEncoder()
    df['Result'] = le.fit_transform(df['Result'])
    df = df.drop_duplicates(keep='first')
    df['Alphabets'] = df['Message'].apply(lambda x: len(x))
    df['words'] = df['Message'].apply(lambda x: len(nltk.word_tokenize(x)))
    df['sentences'] = df['Message'].apply(lambda x: len(nltk.sent_tokenize(x)))
    df['Transformed_text'] = df['Message'].apply(preprocessing)
    return df, le

# Train models and get metrics (cached)
@st.cache_resource  # Cache models and vectorizer
def train_models(df):
    texts = df['Transformed_text'].astype(str)
    tfidf = TfidfVectorizer()
    X = tfidf.fit_transform(texts)
    y = df['Result'].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    models = {
        'MultinomialNB': MultinomialNB(),
        'BernoulliNB': BernoulliNB()
    }
    
    final_dict = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        final_dict[f'Accuracy of {name}'] = accuracy_score(y_test, y_pred)
        final_dict[f'Precision of {name}'] = precision_score(y_test, y_pred, average="weighted", zero_division=0)
        final_dict[f'F1 of {name}'] = f1_score(y_test, y_pred, average='weighted')
        final_dict[f'Recall of {name}'] = recall_score(y_test, y_pred, average='weighted')
        final_dict[f'Confusion_matrix of {name}'] = confusion_matrix(y_test, y_pred).tolist()  # Convert to list for display
    
    # Return the best model (MultinomialNB as default, based on common performance)
    best_model = models['MultinomialNB']
    return tfidf, best_model, final_dict

# Generate WordCloud for spam
def generate_spam_wordcloud(df):
    spam_messages = df[df['Result'] == 1]['Transformed_text']
    if spam_messages.empty:
        return None
    spam_text = spam_messages.str.cat(sep=" ")
    wc = WordCloud(width=500, height=500, min_font_size=10, background_color='white')
    spam_wc = wc.generate(spam_text)
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.imshow(spam_wc)
    ax.axis("off")
    return fig

# Generate histogram for ham messages
def generate_ham_histogram(df):
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.histplot(df[df['Result'] == 0]['Alphabets'], ax=ax)
    ax.set_title('Histogram of Alphabet Counts in Ham Messages')
    return fig

# Main Streamlit app
st.title("SMS Spam Detector")

# Load data
df, le = load_and_prepare_data()

# Train models
tfidf, model, metrics = train_models(df)

# Sidebar for stats and visualizations
with st.sidebar:
    st.header("Model Performance")
    for key, value in metrics.items():
        if 'matrix' in key:
            st.subheader(key)
            st.write(value)  # Displays as list
        else:
            st.write(f"{key}: {value:.4f}")
    
    st.header("Visualizations")
    if st.button("Show Spam WordCloud"):
        wc_fig = generate_spam_wordcloud(df)
        if wc_fig:
            st.pyplot(wc_fig)
    
    if st.button("Show Ham Alphabet Histogram"):
        hist_fig = generate_ham_histogram(df)
        st.pyplot(hist_fig)
    
    st.header("Dataset Stats")
    st.write(df.describe())

# Main prediction interface
st.header("Predict Spam or Ham")
user_input = st.text_area("Enter the SMS message here:")
if st.button("Predict"):
    if user_input:
        transformed = preprocessing(user_input)
        vectorized = tfidf.transform([transformed])
        prediction = model.predict(vectorized)[0]
        result = "Spam" if prediction == 1 else "Ham"
        st.success(f"The message is classified as: **{result}**")
    else:
        st.warning("Please enter a message to predict.")

# Example prediction (from your code)
st.header("Example Prediction")
example_email = "Hello this Fraudster calling can u pay me 1000$"
st.write(f"Example Message: {example_email}")
transformed_ex = preprocessing(example_email)
vectorized_ex = tfidf.transform([transformed_ex])
pred_ex = model.predict(vectorized_ex)[0]
result_ex = "Spam" if pred_ex == 1 else "Ham"
st.write(f"Prediction: **{result_ex}**")