import sqlite3
from sql_agent.sql_generator import generate_sql

# Connect database
conn = sqlite3.connect("database/company.db")


def execute_sql(question):
    sql_query = generate_sql(question)

    print("\nGenerated SQL:")
    print(sql_query)

    # Prevent dangerous queries
    forbidden = ["DROP", "DELETE", "UPDATE", "INSERT", "ALTER"]

    if any(word in sql_query.upper() for word in forbidden):
        return "Unsafe SQL query blocked."

    try:
        cursor = conn.execute(sql_query)

        rows = cursor.fetchall()
        print(rows)
        if cursor.description:
            columns = [col[0] for col in cursor.description]

            print("\nColumns:")
            print(columns)

        return rows

    except Exception as e:
        return f"Error: {e}"

