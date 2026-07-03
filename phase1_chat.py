# Imports
from config import client, chat_model
from phase1_knowledge import build_knowledge_dict

# Knowledge dictionary
knowledge = build_knowledge_dict()

def retrieve_content(message):
    relevant_content = ""
    message_lower = message.lower()

    for key in knowledge.keys():
        if key in message_lower:
            relevant_content+=knowledge[key]+"\n"

    if relevant_content == "":
        return "No additional information retrieved."

    return relevant_content

def chat(message, history):
    relevant_content = retrieve_content(message)
    system_message = f"""
    You are Melodex, a friendly music assistant that knows everything about artists and albums.
    Answer conversationally in plain prose, like you're talking to a friend. 
    Do not use bullet points, headers, or markdown formatting.
    Keep answers concise — 2-4 sentences unless more detail is specifically asked for.
    Here is some relevant information:\n{relevant_content}
    """

    messages = [{"role": "system", "content": system_message}, {"role": "user", "content": message}]

    response = client.chat.completions.create(model = chat_model, messages = messages)

    return response.choices[0].message.content