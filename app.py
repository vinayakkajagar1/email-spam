import streamlit as st
import pickle
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer


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


st.markdown(
    """
    <div style="text-align: center;">
        <h1>📧 Email Spam Detection 📧</h1>
    </div>
    """,
    unsafe_allow_html=True
)
st.markdown("---")

col1, col2 = st.columns([2, 1])

with col1:
    st.write("""
    <div style="font-size: 20px;color: orange;font-family: Georgia, serif; ">
            This app uses a Machine Learning model to predict whether a given message is spam or not.
            Please enter your message in the text area below and click the 'Predict' button to see the prediction.
        </div>
        <div style="margin-top: 60px;"></div>
    """,
        unsafe_allow_html=True)
    st.markdown(
        """
        <div style="font-size: 23px; margin-bottom: 10px;font-family: Georgia, serif;">Enter your message ✉️</div>
        """,
        unsafe_allow_html=True
    )
    input_sms = st.text_area("", height=150)

    if st.button('Predict'):
        st.markdown("---")
        st.subheader("Prediction Result")
        st.write("Analyzing your message...")

        transformed_sms = transform_text(input_sms)
        vector_input = tfidf.transform([transformed_sms])
        result = model.predict(vector_input)[0]

        if result == 1:
            st.error("This message is classified as **SPAM**.")
        else:
            st.success("This message is classified as **NOT SPAM**.")

with col2:
    st.image('image.jpeg', caption='Email spam', use_column_width=True)

st.markdown("---")
st.text("By Vaishnavi ❤️")