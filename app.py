import streamlit as st
import pickle
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import matplotlib.pyplot as plt

nltk.download('punkt')
nltk.download('stopwords')

ps = PorterStemmer()

def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)

    y = [i for i in text if i.isalnum()]

    text = [i for i in y if i not in stopwords.words('english') and i not in string.punctuation]

    y = [ps.stem(i) for i in text]

    return " ".join(y)

tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
model = pickle.load(open('model.pkl', 'rb'))

# Define accuracy for the model
accuracy = 97

st.markdown(
    """
    <div style="text-align: center;">
        <h1 style="color: #FF6347;">📧 Email Spam Detection 📧</h1>
    </div>
    """,
    unsafe_allow_html=True
)

st.markdown(
    """
    <div style="font-size: 20px;color: orange;font-family: Georgia, serif; text-align: center;">
        📬 This app uses a Machine Learning model to predict whether a given message is spam or not. 
        📧 Email spam, also known as junk email, refers to unsolicited messages sent in bulk by email (spamming). 
        🚫 These spam messages can be harmful, containing phishing links or malware. 
        💡 Using Machine Learning models like Multinomial Naive Bayes, we can classify emails as spam or not spam to help keep your inbox safe.
    </div>
    """,
    unsafe_allow_html=True
)

st.markdown("---")

# Video and image section
st.markdown(
    """
    <div style="text-align: center;">
        <h2 style="color: #4682B4;">🎥 Learn More About Email Spam 🎥</h2>
    </div>
    """,
    unsafe_allow_html=True
)

col1, col2 = st.columns([2, 1])  # Adjust the width ratio as needed

# Video in col1
with col1:
    st.video("scam.mp4", format="video/mp4", end_time=None, loop=True, autoplay=True, muted=True)

# Image in col2
with col2:
    st.image('email.jpeg', caption='Spam Filter', width=500)

st.markdown("---")

# Top horizontal images
st.markdown(
    """
    <div style="text-align: center;">
        <h2 style="color: #4682B4;">🔒 Protect Your Inbox 🔒</h2>
    </div>
    """,
    unsafe_allow_html=True
)
col1, col2 = st.columns(2)
with col1:
    st.image('banner1.jpeg', caption='Secure Email', use_column_width=True)
with col2:
    st.image('lock.png', caption='Stay Safe!', use_column_width=True)

st.markdown("---")

# Prediction system
st.markdown(
    """
    <div style="text-align: center;">
        <h1 style="color: #9400d3;">🔍 Prediction System 🔍</h1>
    </div>
    """,
    unsafe_allow_html=True
)
st.markdown(
    """
    <div style="text-align: ;">
        <h3 style="color: #cae00d;">✉️ Enter Your Message Below ✉️</h3>
    </div>
    """,
    unsafe_allow_html=True
)

input_sms = st.text_area("", height=150)

if st.button('Predict'):
    st.markdown("---")
    st.subheader("🔍 Prediction Result")
    st.write("Analyzing your message...")

    transformed_sms = transform_text(input_sms)
    vector_input = tfidf.transform([transformed_sms])
    result = model.predict(vector_input)[0]

    if result == 1:
        st.error("🚫 This message is classified as **SPAM**.")
    else:
        st.success("✅ This message is classified as **NOT SPAM**.")

st.markdown("---")

# Model accuracy
st.markdown(
    """
    <div style="text-align: center;">
        <h2 style="color: #4682B4;">📊 Model Accuracy 📊</h2>
        <p style="font-size: 18px; color: red; font-family: Georgia, serif;">
            The accuracy of the Multinomial Naive Bayes (MNB) model is 97%.
        </p>
    </div>
    """,
    unsafe_allow_html=True
)
fig, ax = plt.subplots(figsize=(6, 3))
ax.barh(['MultinomialNB Model'], [accuracy], color='#4682B4')
ax.set_xlim(0, 100)
ax.set_xlabel('Accuracy (%)')
ax.set_title('Model Accuracy')
st.pyplot(fig)

st.markdown(
    """
    <div style="text-align: center;">
        <p style="font-size: 18px; color: #4682B4; font-family: Georgia, serif;">
            The Multinomial Naive Bayes (MultinomialNB) model is a probabilistic learning method commonly used in text classification.
            It leverages the Bayes' theorem with strong independence assumptions between features.
            This model is particularly effective for problems with discrete features, such as word counts in documents, making it ideal for spam detection.
        </p>
    </div>
    """,
    unsafe_allow_html=True
)
st.markdown("---")
st.text("By Vaishnavi ❤️")
