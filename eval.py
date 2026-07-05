from pydantic import BaseModel, Field
from rag import question_answer
from litellm import Completions
from config import judge
import math

class RetrievalEval(BaseModel):

    # Object to hold retrieval evaluation metrics
    mrr: float = Field(description = "Mean Reciprocal Rank- average across all keywords")
    ndcg: float = Field(description = "Normalized Discounted Cumulative Gain (binary)")

class AnswerEval(BaseModel):

    # Object to hold answer evaluation metrics (LLM as a judge)
    feedback: str = Field(description = "Concise feedback on the answer quality, comparing it to the reference answer and evaluating based on the retrieved context")
    accuracy: float = Field(description = "How factually correct is the answer compared to the reference answer? Score from 1 (wrong. Any wrong answer must score 1) to 5 (ideal. Perfect answer). Score 3 for acceptable")
    completeness: float = Field(description = "How complete is the answer in addressing all aspects of the question? Score from 1 (very poor. Missing key information) to 5 (ideal. answer contains all key information), only give 5 if answer addresses all parts. Score of  3 if acceptable")
    relevance: float = Field(description = "How relevant is the answer to the specific question asked? Score from 1 (very poor. Off-topic) to 5 (ideal. Contains all relevant information and almost no irrelevant information).Score 3 if acceptable- Contains most relevant information with some irrelevant information)")

# Function to calculate Mean Reciprocal Rank
def calculate_mrr():
    return