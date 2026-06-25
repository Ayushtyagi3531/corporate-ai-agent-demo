import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

# The model name must match the exact API string registry identifier
model = ChatGoogleGenerativeAI(
    model="gemma-4-26b-a4b-it",  # Correct string for the 26B MoE variant
    temperature=0
)

print("LangChain Gemini client setup done.")