import os
import sys
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# Ensure local modules are discoverable
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

from llm.router import route_question
from rag.retriever import retrieve_context
from sql_agent.sql_executor import execute_sql
from llm.answer_generator import generate_answer

# Define a Lifespan function to defer initialization until AFTER port binding
@asynccontextmanager
async def lifespan(app: FastAPI):
    print("🚀 Port bound! Starting background asset initialization...")
    # Trigger a dummy call or initialization if needed, or let them lazy load safely
    yield
    print("🛑 Shutting down application...")

# Initialize FastAPI App with Lifespan
app = FastAPI(
    title="Corporate Intelligence Router API",
    description="Unified API interface for RAG, SQL Agents, and Hybrid data lookups.",
    version="1.0.0",
    lifespan=lifespan # Added here!
)
# Define request body structure
class QueryRequest(BaseModel):
    question: str

# Define response payload structure
class QueryResponse(BaseModel):
    query: str
    route: str
    sql_generated: str | None = None
    database_raw_results: str | None = None
    retrieved_documents_context: str | None = None
    answer: str

@app.post("/api/query", response_model=QueryResponse)
async def process_corporate_query(payload: QueryRequest):
    question = payload.question.strip()
    
    if not question:
        raise HTTPException(status_code=400, detail="Question string cannot be empty.")
        
    try:
        # 1. Determine the appropriate execution path
        route = route_question(question)
        
        # Initialize tracking storage variables
        sql_query = None
        sql_result = None
        rag_context = None
        combined_context = ""
        
        # 2. Execute Branching Logic based on Router Engine decision
        if route == "sql":
            sql_result = execute_sql(question)
            
            # Use defensive row list conversions 
            if isinstance(sql_result, list) and len(sql_result) > 0:
                formatted_context = "Database Records Found:\n" + "\n".join([f"- {row}" for row in sql_result])
            else:
                formatted_context = f"Database Query Output: {sql_result}"
                
            combined_context = formatted_context

        elif route == "rag":
            rag_context = retrieve_context(question)
            combined_context = rag_context

        elif route in ["both", "hybrid"]:
            import asyncio
            
            print("⏳ Running SQL and RAG concurrently to minimize latency...")
            
            # This runs both operations at the exact same time!
            sql_task = asyncio.to_thread(execute_sql, question)
            rag_task = asyncio.to_thread(retrieve_context, question)
            
            # Wait for both tasks to complete together
            sql_result, rag_context = await asyncio.gather(sql_task, rag_task)
            
            combined_context = f"""
            SQL Result:
            {sql_result}

            Document Context:
            {rag_context}
            """
        else:
            raise HTTPException(status_code=500, detail=f"Invalid internally generated routing token: {route}")

        # 3. Pass accumulated analytical content to final Answer Generator
        answer = generate_answer(question, combined_context)
        
        # 4. Construct structural JSON tracking breakdown
        return QueryResponse(
            query=question,
            route=route,
            sql_generated=None, # Optional: populate if execute_sql lets you inspect the query string
            database_raw_results=str(sql_result) if sql_result is not None else None,
            retrieved_documents_context=rag_context,
            answer=answer
        )

    except Exception as e:
        # Catch unexpected pipeline script exceptions safely
        raise HTTPException(status_code=500, detail=f"Internal agent pipeline execution error: {str(e)}")

@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "corporate-ai-agent"}
