import time

from pydantic import BaseModel, Field
from rag import question_answer, fetch_context
from litellm import completion
from config import judge
import math
from tests import TestQuestion, load_tests


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
def calculate_mrr(keyword: str, retrieved_docs: list) -> float:
    # Calculate the MRR for a single keyword based on the retrieved documents
    keyword = keyword.lower()
    for rank, doc in enumerate(retrieved_docs, start = 1):
        if keyword in doc.page_content.lower():
            return 1/rank
    return 0.0

# Function to calculate dcg for NDCG
def calculate_dcg(relevances: list[int], k: int) -> float:
    dcg = 0.0
    for i in range(min(k, len(relevances))):
        dcg += relevances[i] / math.log2(i + 2)
    return dcg

# Function to calculate NDCG (Generated using an LLM)
def calculate_ndcg(keyword: str, retrieved_docs: list, k: int) -> float:
    """Calculate nDCG for a single keyword (binary relevance, case-insensitive)."""
    keyword_lower = keyword.lower()
    # Binary relevance: 1 if keyword found, 0 otherwise
    relevances = [1 if keyword_lower in doc.page_content.lower() else 0 for doc in retrieved_docs[:k]]

    # DCG
    dcg = calculate_dcg(relevances, k)

    # Ideal DCG (best case: keyword in first position)
    ideal_relevances = sorted(relevances, reverse=True)
    idcg = calculate_dcg(ideal_relevances, k)

    return dcg / idcg if idcg > 0 else 0.0

# Function to return a RetrievalEval object with MRR and NDCG metrics
def evaluate_retrieval(test: TestQuestion, k: int = 10) -> RetrievalEval:
    # Extracting the relevant content from vector store
    retrieved_docs = fetch_context(test.question)

    # Calculating average MRR for all keywords
    mrr_score = [calculate_mrr(keyword, retrieved_docs) for keyword in test.keywords]
    avg_mrr = sum(mrr_score)/len(mrr_score)

    # Calculating average NDCG
    ndcg_score = [calculate_ndcg(keyword, retrieved_docs, k) for keyword in test.keywords]
    avg_ndcg = sum(ndcg_score)/len(ndcg_score)

    return RetrievalEval(mrr = avg_mrr, ndcg = avg_ndcg)

# Function to evaluate the answer quality using an LLM as a judge
def evaluate_answer(test: TestQuestion):

    generated_ans = question_answer(test.question, []) # Getting the answer from the RAG system
    judge_messages = [
        {"role": "system", "content": "You are an expert evaluator assessing the quality of answers. Evaluate the generated answer by comparing it to the reference answer. Only give 5/5 scores for perfect answers."},
        {"role": "user", "content": f"""Question: {test.question}\n Answer: {generated_ans}\n Reference Answer: {test.reference_answer}
Please evaluate the generated answer on three dimensions:
1. Accuracy: How factually correct is it compared to the reference answer? Only give 5/5 scores for perfect answers.
2. Completeness: How thoroughly does it address all aspects of the question, covering all the information from the reference answer?
3. Relevance: How well does it directly answer the specific question asked, giving no additional information?

Provide detailed feedback and scores from 1 (very poor) to 5 (ideal) for each dimension. If the answer is wrong, then the accuracy score must be 1.
"""}
    ]
    judge_response = completion(model = judge, messages = judge_messages, response_format = AnswerEval)

    ans_eval = AnswerEval.model_validate_json(judge_response.choices[0].message.content)

    return ans_eval

# Function to evaluate all retrieval
def evaluate_all_retrieval():
    tests = load_tests()
    total = len(tests)

    for index, test in enumerate(tests):
        result = evaluate_retrieval(test)
        progress = (index+1)/total
        yield test, result, progress

# Function to evaluate all answers
def evaluate_all_answers():
    tests = load_tests()
    total = len(tests)

    for index, test in enumerate(tests):
        result = evaluate_answer(test)
        progress = (index + 1) / total
        yield test, result, progress
        time.sleep(10)


# For personal testing
if __name__ == "__main__":
    tests = load_tests()
    sample_test = tests[0]

    print("Evaluating retrieval results: ")
    retrieval_results = evaluate_retrieval(sample_test)
    print(f"MRR: {retrieval_results.mrr} | NDCG: {retrieval_results.ndcg}")

    print("Evaluating the answer: ")
    answer_results = evaluate_answer(sample_test)
    print(f"Accuracy: {answer_results.accuracy} | Feedback: {answer_results.feedback}")