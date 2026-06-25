import os
from llm.gemini_client import model

def generate_answer(question, context):
    prompt = f"""
    You are a helpful assistant.

    Answer the user's question by summarizing the relevant information found inside the provided context. Always state the name of the document or source where you found the information so the user knows where it came from.

    If the answer cannot be determined from the context, say:
    "I could not find the answer in the provided information."

    The context may come from:
    - Database query results
    - Retrieved documents
    - Both

    If the answer cannot be determined from the context, say:
    "I could not find the answer in the provided information."

    Context:
    {context}

    Question:
    {question}

    Answer:
    """

    response = model.invoke(prompt)
    
    # --- FIX: Extract content text safely if it is returned as a list ---
    content = response.content
    if isinstance(content, list):
        # Join pieces together if it's a list of blocks, extracting the 'text' key if present
        content = "".join([chunk if isinstance(chunk, str) else chunk.get("text", "") for chunk in content])
    
    return content.strip()