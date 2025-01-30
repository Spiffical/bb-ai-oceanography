import sqlite3
import os
import sys
import argparse
from pathlib import Path

def create_merged_database(output_path, schema_source_path):
    """Create a new database with the same schema as the source databases."""
    conn = sqlite3.connect(output_path)
    cursor = conn.cursor()
    
    # Get schema from the first database
    source_conn = sqlite3.connect(schema_source_path)
    source_cursor = source_conn.cursor()
    source_cursor.execute("SELECT sql FROM sqlite_master WHERE type='table' AND name='articles'")
    create_table_sql = source_cursor.fetchone()[0]
    
    cursor.execute(create_table_sql)
    source_conn.close()
    return conn

def get_all_entries(db_path):
    """Get all entries from a database with Reference as key."""
    entries = {}
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Get column names
        cursor.execute("PRAGMA table_info(articles)")
        columns = [col[1] for col in cursor.fetchall()]
        
        # Get all rows
        cursor.execute("SELECT * FROM articles")
        rows = cursor.fetchall()
        
        for row in rows:
            # Create dictionary for each row using column names
            entry_dict = dict(zip(columns, row))
            reference = entry_dict.get('Reference')
            if reference:  # Only include entries with a reference
                entries[reference] = entry_dict
        
        conn.close()
        return entries
    except sqlite3.Error as e:
        print(f"Error accessing database {db_path}: {e}")
        return {}

def parse_args():
    parser = argparse.ArgumentParser(description='Merge multiple SQLite databases with priority based on argument order (first has highest priority)')
    parser.add_argument('databases', nargs='+', type=Path, help='Paths to SQLite databases (in priority order, highest first)')
    parser.add_argument('-o', '--output', type=Path, default=Path("paperetlmodels/models/pdfxml-oceanai/articles.sqlite"),
                      help='Output path for merged database (default: paperetlmodels/models/pdfxml-oceanai/articles.sqlite)')
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Create output directory if it doesn't exist
    output_dir = args.output.parent
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Verify all input databases exist
    for db_path in args.databases:
        if not db_path.exists():
            print(f"Error: Database not found: {db_path}")
            sys.exit(1)
    
    print("\nReading databases in priority order:")
    # Read all databases and track their entries
    all_entries = []
    for i, db_path in enumerate(args.databases, 1):
        entries = get_all_entries(db_path)
        all_entries.append(entries)
        print(f"{i}. {db_path.name}: {len(entries)} entries")
    
    # Merge entries with priority (reverse order - lowest priority first)
    merged_entries = {}
    for entries in reversed(all_entries):
        merged_entries.update(entries)
    
    print(f"\nTotal unique entries after merge: {len(merged_entries)}")
    
    # Create and populate new database
    print("\nCreating new database...")
    merged_conn = create_merged_database(args.output, args.databases[0])
    cursor = merged_conn.cursor()
    
    # Get column names (excluding rowid if it exists)
    columns = list(next(iter(merged_entries.values())).keys())
    if 'rowid' in columns:
        columns.remove('rowid')
    
    # Insert all entries
    placeholders = ','.join(['?' for _ in columns])
    insert_sql = f"INSERT INTO articles ({','.join(columns)}) VALUES ({placeholders})"
    
    for entry in merged_entries.values():
        # Extract values in the same order as columns
        values = [entry[col] for col in columns]
        cursor.execute(insert_sql, values)
    
    merged_conn.commit()
    merged_conn.close()
    
    print(f"\nMerged database created successfully at: {args.output}")
    print("\nPriority order used:")
    for i, db_path in enumerate(args.databases, 1):
        print(f"{i}. {db_path.name}")

if __name__ == "__main__":
    main() 