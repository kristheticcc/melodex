# Imports
import os
from openai import OpenAI
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_chroma import Chroma

# Loading environment variables
load_dotenv(override = True)
api_key = os.getenv("GROQ_API_KEY") # groq for now (might change to openai later)

# Setting up client
base_url="https://api.groq.com/openai/v1"
client = OpenAI(base_url = base_url, api_key = api_key)

# models
chat_model = "openai/gpt-oss-120b"
embedding_model = "all-MiniLM-L6-v2"

# Embeddings
embedding = HuggingFaceEmbeddings(model_name = embedding_model)

# Vector store
DB_NAME = "vector_db"
vector_store = Chroma(persist_directory = DB_NAME, embedding_function = embedding)

# Retriever and generator
llm = ChatGroq(model = chat_model, temperature = 0, api_key = api_key)
retriever = vector_store.as_retriever()

# Judge LLM for evaluation
judge = "groq/llama-3.3-70b-versatile"