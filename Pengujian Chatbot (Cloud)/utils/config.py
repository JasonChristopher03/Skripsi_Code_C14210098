import json
import streamlit as st
from google.cloud import storage

@st.cache_resource
def load_modules(bucket_name="skripsi-rag", file_name="modules.json"):
    client = storage.Client()
    bucket = client.get_bucket(bucket_name)
    file = bucket.blob(file_name)
    content = file.download_as_text()
    return json.loads(content)

@st.cache_resource
def load_questions(bucket_name="skripsi-rag", file_name="questions.json"):
    client = storage.Client()
    bucket = client.get_bucket(bucket_name)
    file = bucket.blob(file_name)
    content = file.download_as_text()
    return json.loads(content)
