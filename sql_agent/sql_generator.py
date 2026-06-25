import sys
import os
import re
from langchain_google_genai._common import GoogleGenerativeAIError

# Path alignment setups
ROOT_DIR = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)
sys.path.append(ROOT_DIR)

from llm.gemini_client import model

def generate_sql(question):
    schema = """
    Table: sales
    Columns:
    - customer
    - amount
    - product
    """

    prompt = f"""
    You are an expert SQLite query generator.
    If the user asks for an aggregation (like SUM, AVG, COUNT), you MUST always provide a clean column alias using the 'AS' keyword.
    For example, instead of SELECT SUM(amount), use SELECT SUM(amount) AS total_revenue.

    Schema:
    {schema}

    Rules:
    1. Return ONLY the SQL query.
    2. Do not explain anything.
    3. Do not use markdown.
    4. Do not write 'sqlite' or 'SQL Query:'.
    5. Generate valid SQLite syntax.

    Question:
    {question}
    """
    try:
        response = model.invoke(prompt)

        # Safe extraction for LangChain multi-part structured content lists
        content = response.content
        if isinstance(content, list):
            content = "".join([chunk if isinstance(chunk, str) else chunk.get("text", "") for chunk in content])
            
        text = content.strip()

        # Clean markdown wrappers if returned
        text = text.replace("```sql", "")
        text = text.replace("```", "")
        text = text.strip()

        # Regex extract query expression
        match = re.search(
            r"(SELECT|WITH).*?(;|$)",
            text,
            re.IGNORECASE | re.DOTALL
        )

        if match:
            return match.group(0).strip()

        return text

    except GoogleGenerativeAIError as e:
        print(f"Gemini API Error in SQL Generation: {e}")
        return "ERROR: API limit or failure"
    except Exception as e:
        print(f"Unexpected Error in SQL Generator: {e}")
        return "ERROR: System failure"