import pandas as pd 
import sqlite3 
df = pd.read_csv("data/sales.csv") 
conn = sqlite3.connect( "database/company.db" ) 
df.to_sql( "sales", conn, if_exists="replace", index=False ) 
conn.commit() 
conn.close() 
print("Database Created")