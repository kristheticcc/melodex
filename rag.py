from config import llm, retriever
from langchain_core.messages import SystemMessage, HumanMessage
from advanced_rag import pro_retriever

SYSTEM_PROMPT_TEMPLATE = """
    You are Melodex, a friendly music assistant that knows everything about artists and albums.
    Answer conversationally in plain prose, like you're talking to a friend. 
    Do not use bullet points, headers, or markdown formatting.
    Keep answers concise — 2-4 sentences unless more detail is specifically asked for.
    IF YOU DO NOT KNOW THE ANSWER, JUST SAY SO.
    Here is some relevant information:\n{context}
    """


def fetch_context(question):
    docs = pro_retriever.invoke(question)
    return docs

def question_answer(message, history):
    docs = fetch_context(message)
    context = "\n".join(doc.page_content for doc in docs)

    system_prompt = SYSTEM_PROMPT_TEMPLATE.format(context = context)

    response = llm.invoke([
        SystemMessage(content = system_prompt),
        HumanMessage(content = message),
    ])

    return response.content


