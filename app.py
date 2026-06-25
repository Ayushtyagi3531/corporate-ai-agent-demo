from llm.router import route_question
from rag.retriever import retrieve_context
from sql_agent.sql_executor import execute_sql
from llm.answer_generator import generate_answer


while True:
    question = input("\nAsk Question: ")

    if question.lower() == "exit":
        break

    route = route_question(question)

    print("\nRoute:", route)

    if route == "sql":
        sql_result = execute_sql(question) 

        # Let's cleanly check if it's a dictionary, a tuple, or a raw list
        columns = ['customer', 'amount', 'product'] # Fallback default schema columns
        rows = []

        if isinstance(sql_result, dict):
            # If your sql_executor returns a dictionary structure
            rows = sql_result.get("data", sql_result.get("rows", []))
            columns = sql_result.get("columns", columns)
        elif isinstance(sql_result, tuple) and len(sql_result) == 2:
            # If your sql_executor returns (columns, rows)
            columns, rows = sql_result
        elif isinstance(sql_result, list):
            # If it's just a raw list of row tuples
            rows = sql_result

        # --- CONVERT TO HUMAN-READABLE TABLE CONTEXT ---
        if rows:
            formatted_context = "SQL Database Results:\n"
            formatted_context += f"Columns: {', '.join(columns)}\n"
            for i, row in enumerate(rows, 1):
                # Map individual tuple values cleanly to help Gemini read them textually
                formatted_context += f"Row {i}: {dict(zip(columns, row)) if len(row) == len(columns) else row}\n"
        else:
            formatted_context = f"SQL Database returned an empty result or error: {sql_result}"

        # Send the descriptive text table to your answer generator
        answer = generate_answer(
            question,
            formatted_context
        )
        print("\nAnswer:")
        print(answer)

    elif route == "rag":
        rag_context = retrieve_context(question)

        answer = generate_answer(
            question,
            rag_context
        )

        print("\nAnswer:")
        print(answer)

    elif route in ["both", "hybrid"]:
        sql_result = execute_sql(question)

        rag_context = retrieve_context(question)

        combined_context = f"""
        SQL Result:
        {sql_result}

        Document Context:
        {rag_context}
        """

        answer = generate_answer(
            question,
            combined_context
        )

        print("\nAnswer:")
        print(answer)

    else:
        print("Unable to determine route.")