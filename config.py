# Imports
import os

from langchain_community.embeddings import HuggingFaceEmbeddings
from openai import OpenAI
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings

# Loading environment variables
load_dotenv(override = True)
api_key = os.getenv("GROQ_API_KEY") # groq for now (might change to openai later)

# Setting up client
base_url="https://api.groq.com/openai/v1"
client = OpenAI(base_url = base_url, api_key = api_key)

# models
model = "openai/gpt-oss-120b"
embedding_model = "all-MiniLM-L6-v2"
embedding = HuggingFaceEmbeddings(model = embedding_model)