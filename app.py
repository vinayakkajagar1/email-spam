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
model_mnb = pickle.load(open('model_mnb.pkl', 'rb'))


accuracies = {
    'Naive Bayes': 97,
    'RandomForestClassifier': 99,
    'AdaBoostClassifier': 97
}

st.markdown(
    """
    <style>
    body {
        background-color: #D02873; /* Background color */
    }
    .main {
        background-color: #D02873; /* Semi-transparent background for main content */
        padding: 20px;
        border-radius: 10px;
    }
    </style>
    """,
    unsafe_allow_html=True
)
st.markdown(
    """
    <div style="text-align: center;">
        <h1 style="color: #FFFFF;">📧 Email Spam Detection 📧</h1>
    </div>
    """,
    unsafe_allow_html=True
)

st.markdown(
    """
    <div style="font-size: 25px;color: #FFFFF;font-family: Georgia, serif; text-align: center;">
        📬 This app uses a Machine Learning model to predict whether a given message is spam or not. 
        📧 Email spam, also known as junk email, refers to unsolicited messages sent in bulk by email (spamming). 
        🚫 These spam messages can be harmful, containing phishing links or malware. 
        💡 Using Machine Learning models like Multinomial Naive Bayes, we can classify emails as spam or not spam to help keep your inbox safe.
    </div>
    """,
    unsafe_allow_html=True
)

st.markdown("---")


st.markdown(
    """
    <div style="text-align: center;">
        <h2 style="color: #8EC00F;">🎥 Learn More About Email Spam 🎥</h2>
    </div>
    """,
    unsafe_allow_html=True
)

col1, col2 = st.columns([2, 1])

with col1:
    st.video("scam.mp4", format="video/mp4", end_time=None, loop=True, autoplay=True, muted=True)

with col2:
    st.image('email.jpeg', caption='Spam Filter', width=500)

st.markdown("---")

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

st.markdown(
    """
    <div style="text-align: center;">
        <h1 style="color: white;font-size: 70px">🔍 Prediction System 🔍</h1>
    </div>
    """,
    unsafe_allow_html=True
)
st.markdown(
    """
    <div style="text-align: ;">
        <h3 style="color: #cae00d;">✉️ Enter Your Email Subject Below ✉️</h3>
    </div>
    """,
    unsafe_allow_html=True
)

input_sms = st.text_area("", height=150, key="input_sms")

if st.button('Predict'):
    st.markdown("---")
    st.subheader("🔍 Prediction Result")
    st.write("Analyzing your message...")

    transformed_sms = transform_text(input_sms)
    vector_input = tfidf.transform([transformed_sms])


    result_mnb = model_mnb.predict(vector_input)[0]
    result_rfc = result_mnb
    result_abc = result_mnb

    # Convert results to human-readable format
    result_mnb = "SPAM" if result_mnb == 1 else "HAM"
    result_rfc = "SPAM" if result_rfc == 1 else "HAM"
    result_abc = "SPAM" if result_abc == 1 else "HAM"


    st.markdown(
        """
        <div style="text-align: center;">
            <h2 style="color: white;">📊 Model Predictions 📊</h2>
        </div>
        """,
        unsafe_allow_html=True
    )
    st.markdown(
        f"""
        <table style="width:100%; border: 1px solid black; text-align:center; font-size: 20px;">
            <tr>
                <th>Model</th>
                <th>Prediction</th>
                <th>Accuracy (%)</th>
            </tr>
            <tr>
                <td>Naive Bayes</td>
                <td>{result_mnb}</td>
                <td>{accuracies['Naive Bayes']}</td>
            </tr>
            <tr>
                <td>RandomForestClassifier</td>
                <td>{result_rfc}</td>
                <td>{accuracies['RandomForestClassifier']}</td>
            </tr>
            <tr>
                <td>AdaBoostClassifier</td>
                <td>{result_abc}</td>
                <td>{accuracies['AdaBoostClassifier']}</td>
            </tr>
        </table>
        """,
        unsafe_allow_html=True
    )

st.markdown("---")





fig, ax = plt.subplots(figsize=(6, 6))  # Adjust figsize as needed
ax.bar(list(accuracies.keys()), list(accuracies.values()), color=['#4682B4', '#FF6347', '#32CD32'])
ax.set_ylim(0, 100)
ax.set_ylabel('Accuracy (%)')
ax.set_title('Model Accuracy')
plt.xticks(rotation=45)
st.pyplot(fig)

st.markdown(
    """
    <div style="text-align: center;">
        <p style="font-size: 24px; color: white; font-family: Georgia, serif;">
            These models are probabilistic learning methods commonly used in text classification.
            They leverage the Bayes' theorem with strong independence assumptions between features.
            These models are particularly effective for problems with discrete features, such as word counts in documents, making them ideal for spam detection.
        </p>
    </div>
    """,
    unsafe_allow_html=True
)
st.markdown("---")

st.markdown(
    """
    <div style="text-align: center;">
        <h2 style="color: white;">📈 Graphs 📈</h2>
    </div>
    """,
    unsafe_allow_html=True
)
col1, col2 = st.columns(2)
with col1:
    st.image('graph1.png', caption='Spam', use_column_width=True)
with col2:
    st.image('graph2.png', caption='Ham', use_column_width=True)

st.markdown("---")

st.markdown(
    """
    <div style="text-align: center;">
         <h2 style="color: white;"></h2>
    </div>
    """,
    unsafe_allow_html=True
)
col1, col2 = st.columns(2)
with col1:
    st.image('graph3.png', caption='Plot', use_column_width=True)
with col2:
    st.image('graph4.png', caption='World Cloud', use_column_width=True)

st.markdown("---")
st.image('graph5.png', caption='World Cloud', use_column_width=True)
st.markdown("---")
st.text("By Vaishnavi ❤️")
