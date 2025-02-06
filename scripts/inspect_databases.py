import sqlite3
import sys
from pathlib import Path

def inspect_database(db_path):
    """Inspect the structure and content of a SQLite database."""
    try:
        print(f"\nInspecting database: {db_path}")
        print("=" * 50)
        
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Get list of all tables
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = cursor.fetchall()
        
        print(f"Tables found: {[table[0] for table in tables]}")
        
        # For each table, show structure and row count
        for table in tables:
            table_name = table[0]
            print(f"\nTable: {table_name}")
            print("-" * 30)
            
            # Get table schema
            cursor.execute(f"SELECT sql FROM sqlite_master WHERE type='table' AND name='{table_name}'")
            schema = cursor.fetchone()[0]
            print("Schema:")
            print(schema)
            
            # Get column info
            cursor.execute(f"PRAGMA table_info({table_name})")
            columns = cursor.fetchall()
            print("\nColumns:")
            for col in columns:
                print(f"  {col[1]} ({col[2]})")
            
            # Get row count
            cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
            count = cursor.fetchone()[0]
            print(f"\nTotal rows: {count}")
            
            # Show a sample row if table is not empty
            if count > 0:
                cursor.execute(f"SELECT * FROM {table_name} LIMIT 1")
                sample = cursor.fetchone()
                print("\nSample row:")
                for col, val in zip([c[1] for c in columns], sample):
                    print(f"  {col}: {val}")
        
        conn.close()
        
    except sqlite3.Error as e:
        print(f"Error inspecting database {db_path}: {e}")

def main():
    if len(sys.argv) < 2:
        print("Usage: python inspect_databases.py <db_path1> [db_path2 ...]")
        sys.exit(1)
    
    for db_path in sys.argv[1:]:
        path = Path(db_path)
        if not path.exists():
            print(f"Error: Database not found: {path}")
            continue
        inspect_database(path)

if __name__ == "__main__":
    main() 