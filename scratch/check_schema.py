import sqlite3
import os

db_path = 'data/la_liga.db'
if os.path.exists(db_path):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT sql FROM sqlite_master WHERE type='table' AND name='matches';")
    print(cursor.fetchone()[0])
    conn.close()
else:
    print(f"Database {db_path} not found.")
