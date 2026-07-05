# Imports
import json
from pydantic import BaseModel, Field
from pathlib import Path

TEST_FILE = Path("tests.jsonl")

class TestQuestion(BaseModel):
    # A test question with expected structure

    question: str = Field(description = "The question to ask the RAG system")
    key_words: list[str] = Field(description = "The list of keywords that must appear in retrieved content")
    reference_answer: str = Field(description = "The reference answer to the question")
    category: str = Field(description = "The category of the question (eg. direct_fact, temporal, comparative)")

def load_tests() -> list[TestQuestion]:
    # To validate the structure

    tests = []

    with open(TEST_FILE, "r", encoding = 'utf-8') as f:
        for line in f:
            data = json.loads(line)
            tests.append(TestQuestion(**data)) # step 1: data spreads as named arguments, step 2: validates if follows structure

    return tests

