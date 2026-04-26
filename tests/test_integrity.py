import sqlite3
import os

db_path = 'data/la_liga.db'
def test_upsert_integrity():
    print("[TEST] Verifying SQLite Unique Constraint and Upsert Logic...")
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    test_date = '2099-01-01'
    test_team = 'Test FC'
    
    # 1. Insert Initial
    cursor.execute("""
        INSERT INTO matches (date, team, opponent, venue, gf, ga, result)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    """, (test_date, test_team, 'Opponent FC', 'Home', 1, 0, 'W'))
    
    # 2. Attempt Upsert (Update)
    cursor.execute("""
        INSERT INTO matches (date, team, opponent, venue, gf, ga, result)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(date, team) DO UPDATE SET
        gf = excluded.gf,
        result = excluded.result
    """, (test_date, test_team, 'Opponent FC', 'Home', 5, 0, 'W'))
    
    # 3. Verify
    cursor.execute("SELECT gf FROM matches WHERE date = ? AND team = ?", (test_date, test_team))
    res = cursor.fetchone()[0]
    
    # Cleanup
    cursor.execute("DELETE FROM matches WHERE date = ? AND team = ?", (test_date, test_team))
    conn.commit()
    conn.close()
    
    if res == 5:
        print("[PASS] Upsert logic validated. Unique constraint active.")
    else:
        print(f"[FAIL] Expected gf=5, got {res}")
        exit(1)

if __name__ == "__main__":
    test_upsert_integrity()
