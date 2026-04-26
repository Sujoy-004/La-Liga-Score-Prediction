import sqlite3
import os

db_path = 'data/la_liga.db'
if os.path.exists(db_path):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("PRAGMA index_list('matches');")
    indexes = cursor.fetchall()
    for idx in indexes:
        print(f"Index: {idx}")
        cursor.execute(f"PRAGMA index_info('{idx[1]}');")
        print(f"Details: {cursor.fetchall()}")
    conn.close()
else:
    print(f"Database {db_path} not found.")
