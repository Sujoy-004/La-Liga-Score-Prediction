import sqlite3
import os

db_path = 'data/la_liga.db'
if os.path.exists(db_path):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    print("Dropping existing non-unique index...")
    cursor.execute("DROP INDEX IF EXISTS idx_team_date;")
    
    print("Creating unique index on (date, team)...")
    try:
        cursor.execute("CREATE UNIQUE INDEX idx_date_team_unique ON matches (date, team);")
        print("Success.")
    except sqlite3.IntegrityError as e:
        print(f"Error: {e}")
        print("Duplicates found. Cleaning duplicates before creating unique index...")
        # Optional: Clean duplicates if they exist
        cursor.execute("""
            DELETE FROM matches 
            WHERE rowid NOT IN (
                SELECT MIN(rowid) 
                FROM matches 
                GROUP BY date, team
            );
        """)
        cursor.execute("CREATE UNIQUE INDEX idx_date_team_unique ON matches (date, team);")
        print("Cleaned and created unique index.")
    
    conn.commit()
    conn.close()
else:
    print(f"Database {db_path} not found.")
