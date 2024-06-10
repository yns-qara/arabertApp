import streamlit as st
from transformers import pipeline

# Function to initialize the text classification pipeline with the specified model
@st.cache_resource
def get_pipeline(model_name):
    return pipeline("text-classification", model=model_name, top_k=None)

# Available models
models = {
    "Medical BERT": "achDev/medicalBert",
    "Medecal Roberta": "achDev/medidalRoberta",
    "General": "achDev/reberta"
}

# Streamlit app
st.title("Text Classification")

st.write("""
    This is a text classification app. 
    Enter text in Arabic and the app will classify it into one of the predefined categories.
""")

# Select a model
model_name = st.selectbox("Select Model", list(models.keys()))
pipe = get_pipeline(models[model_name])

text_input = st.text_area("Enter text in Arabic:")

if st.button("Classify"):
    if text_input:
        classification_results = pipe(text_input)

        # st.write("Raw classification results:")
        # st.write(classification_results)

        results = classification_results[0]

        st.subheader("Classification Results")

        for result in results:
            label = result['label']
            score = result['score']
            st.write(f"{label}: {score:.3f}")
            st.progress(score)
    else:
        st.write("Please enter some text to classify.")
