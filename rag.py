from config import llm, retriever
from langchain_core.messages import SystemMessage, HumanMessage

SYSTEM_PROMPT_TEMPLATE = """
    You are Melodex, a friendly music assistant that knows everything about artists and albums.
    Answer conversationally in plain prose, like you're talking to a friend. 
    Do not use bullet points, headers, or markdown formatting.
    Keep answers concise — 2-4 sentences unless more detail is specifically asked for.
    Here is some relevant information:\n{context}
    """

def question_answer(message, history):
    docs = retriever.invoke(message) # Extracting relevant content from the vector store
    context = "\n".join(doc.page_content for doc in docs) # Extracting the page content

    system_prompt = SYSTEM_PROMPT_TEMPLATE.format(context = context)

    response = llm.invoke([
        SystemMessage(content = system_prompt),
        HumanMessage(content = message),
    ])

    return response.content


