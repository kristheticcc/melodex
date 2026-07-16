# Imports

from langchain_classic.retrievers import MultiQueryRetriever
from langchain_classic.retrievers import ContextualCompressionRetriever
from langchain_classic.retrievers.document_compressors import LLMChainExtractor
from config import retriever, rewriter

def get_pro_retriever():

    # LLM generates 3 different versions of the query. Retriever retrieves documents for each query and returns the union of all retrieved documents (excluding duplicates).
    multi_query_retriever = MultiQueryRetriever.from_llm(
        llm = rewriter,
        retriever = retriever,
        include_original=True,
    )

    print(multi_query_retriever.llm_chain)

    # Compressor: Uses LLM to remove irrelevant information from the retrieved content
    compressor = LLMChainExtractor.from_llm(rewriter)

    # Chaining the two objects together
    pro_retriever = ContextualCompressionRetriever(
        base_compressor = compressor,
        base_retriever = multi_query_retriever
    )

    return pro_retriever


# Initialize the object for it to be used in rag.py
pro_retriever = get_pro_retriever()


