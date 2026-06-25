import os
import sys
from llm.gemini_client import model

# Path alignment setups
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_DIR)

def route_question(question):
    prompt = f"""
    You are an expert query router for a corporate data system. Your job is to classify the user's question into exactly one category based on structural intent.

    Available Categories:
    - sql : The user wants raw metrics, calculations, listings, or totals from the 'sales' table. (e.g., "What is total revenue?", "List sales over 1000")
    - rag : The user wants to know company rules, corporate guidelines, compliance policies, or HR instructions. (e.g., "What is the dual-factor authentication rule?")
    - both : The user is asking you to check real data for violations against corporate rules, or cross-reference database logs with written policies. (e.g., "Did any transactions in our sales logs violate policy?")

    CRITICAL RULE:
    If a question asks you to look at data or logs AND evaluate whether they violate a policy, rule, or boundary, you MUST choose 'both'. Do not look at keywords alone; analyze the structural intent.

    Return ONLY one word: sql, rag, or both. Do not include punctuation or explanations.

    Question:
    {question}
    """
    try:
        response = model.invoke(prompt)
        
        # Safe extraction wrapper for LangChain multi-part structured content structures
        content = response.content
        if isinstance(content, list):
            text = "".join([
                chunk if isinstance(chunk, str) else 
                (chunk.get("text", "") if isinstance(chunk, dict) else getattr(chunk, "text", "")) 
                for chunk in content
            ])
        else:
            text = str(content)
            
        route = text.strip().lower()
        
        # Fallback routing cleanups
        if "both" in route or "hybrid" in route:
            return "both"
        elif "sql" in route:
            return "sql"
        elif "rag" in route:
            return "rag"
            
        return "rag"

    except Exception as e:
        print(f"Routing Pipeline Failure: {e}")
        return "both"  # Defensive fallback: route to both so all context is pulled if router fails