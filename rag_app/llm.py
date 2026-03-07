import os
from dotenv import load_dotenv
from google import genai

load_dotenv()


def llm_response(context, query):
    """
    Generate an answer to a user query using a Large Language Model (Gemini).

    This function constructs a prompt that instructs the LLM to answer a
    question strictly using the provided context. If the answer cannot be
    found in the context, the model is instructed to return
    "Answer not found in context".

    Args:
        context (str): Retrieved contextual information (e.g., from a vector
                       database or RAG pipeline) used to answer the query.
        query (str): The user's question.

    Returns:
        str: The generated response from the Gemini model based on the
             provided context and query.
    """

    prompt = f"""
    You are a helpful AI assistant.

    Answer the question using ONLY the provided context. give a detailed answer.
    If the answer is not in the context, say "Answer not found in context".

    Context:
    {context}

    Question:
    {query}

    Answer:
    """
    client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

    response_llm = client.models.generate_content(
        model="gemini-2.5-flash", contents=prompt
    )
    return response_llm.candidates[0].content.parts[0].text
