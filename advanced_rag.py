# Imports

from langchain_classic.retrievers import MultiQueryRetriever
from langchain_classic.retrievers import ContextualCompressionRetriever
from langchain_classic.retrievers.document_compressors import CrossEncoderReranker
from config import retriever, rewriter
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from config import cross_encoding_model
# from langchain_classic.retrievers.document_compressors import LLMChainExtractor

def get_pro_retriever():

    # LLM generates 3 different versions of the query. Retriever retrieves documents for each query and returns the union of all retrieved documents (excluding duplicates).
    multi_query_retriever = MultiQueryRetriever.from_llm(
        llm = rewriter,
        retriever = retriever,
        include_original=True,
    )

    # Re-ranker: Uses a cross-encoder to re-rank the retrieved documents based on their relevance to the query
    cross_encoder = HuggingFaceCrossEncoder(model_name=cross_encoding_model)
    reranker = CrossEncoderReranker(model=cross_encoder, top_n=5)

    # Compressor: Uses LLM to remove irrelevant information from the retrieved content (EXCLUDED)
    # compressor = LLMChainExtractor.from_llm(rewriter)

    # Chaining the two objects together: Multi query retrieves, and re-ranker ranks and filters
    pro_retriever = ContextualCompressionRetriever(
        base_compressor = reranker,
        base_retriever = multi_query_retriever
    )


    return pro_retriever


# Initialize the object for it to be used in rag.py
pro_retriever = get_pro_retriever()


