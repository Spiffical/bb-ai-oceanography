import sqlite3
import os
import sys
import argparse
from pathlib import Path

def create_merged_database(output_path, schema_source_path):
    """Create a new database with the same schema as the source databases."""
    conn = sqlite3.connect(output_path)
    cursor = conn.cursor()
    
    # Get schema from the first database for both tables
    source_conn = sqlite3.connect(schema_source_path)
    source_cursor = source_conn.cursor()
    
    # Copy schema for both tables
    for table in ['articles', 'sections']:
        source_cursor.execute(f"SELECT sql FROM sqlite_master WHERE type='table' AND name='{table}'")
        create_table_sql = source_cursor.fetchone()[0]
        cursor.execute(create_table_sql)
    
    source_conn.close()
    return conn

def get_all_entries(db_path, table_name):
    """Get all entries from a specified table in the database."""
    entries = {}
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Get column names
        cursor.execute(f"PRAGMA table_info({table_name})")
        columns = [col[1] for col in cursor.fetchall()]
        
        # Get all rows
        cursor.execute(f"SELECT * FROM {table_name}")
        rows = cursor.fetchall()
        
        for row in rows:
            # Create dictionary for each row using column names
            entry_dict = dict(zip(columns, row))
            if table_name == 'articles':
                key = entry_dict.get('Reference')
                if key:  # Only include entries with a reference
                    entries[key] = entry_dict
            else:  # sections table
                # Each section should be treated as unique, use Article + Id as key
                article = entry_dict.get('Article')
                section_id = entry_dict.get('Id')
                if article and section_id is not None:
                    entries[f"{article}|{section_id}"] = entry_dict
        
        conn.close()
        return entries
    except sqlite3.Error as e:
        print(f"Error accessing {table_name} in database {db_path}: {e}")
        return {}

def parse_args():
    parser = argparse.ArgumentParser(description='Merge multiple SQLite databases with priority based on argument order (first has highest priority)')
    parser.add_argument('databases', nargs='+', type=Path, help='Paths to SQLite databases (in priority order, highest first)')
    parser.add_argument('-o', '--output', type=Path, default=Path("paperetlmodels/pdfxml-oceanai/articles.sqlite"),
                      help='Output path for merged database (default: paperetlmodels/pdfxml-oceanai/articles.sqlite)')
    return parser.parse_args()

def merge_table_entries(db_paths, table_name):
    """Merge entries from a specific table across all databases."""
    print(f"\nReading {table_name} table from databases in priority order:")
    all_entries = []
    for i, db_path in enumerate(db_paths, 1):
        entries = get_all_entries(db_path, table_name)
        all_entries.append(entries)
        print(f"{i}. {db_path.name}: {len(entries)} entries")
    
    # Merge entries with priority (reverse order - lowest priority first)
    merged_entries = {}
    for entries in reversed(all_entries):
        merged_entries.update(entries)
    
    print(f"Total unique entries after merge: {len(merged_entries)}")
    return merged_entries

def insert_entries(conn, table_name, entries):
    """Insert entries into the specified table."""
    if not entries:
        return
    
    cursor = conn.cursor()
    
    # Get the actual columns from the database
    cursor.execute(f"PRAGMA table_info({table_name})")
    db_columns = [col[1] for col in cursor.fetchall()]
    
    # For sections table, we need to reset the Id column as it's an auto-incrementing primary key
    if table_name == 'sections':
        if 'Id' in db_columns:
            db_columns.remove('Id')
    
    # Insert all entries
    placeholders = ','.join(['?' for _ in db_columns])
    insert_sql = f"INSERT INTO {table_name} ({','.join(db_columns)}) VALUES ({placeholders})"
    
    for entry in entries.values():
        # Extract values in the same order as database columns
        values = [entry.get(col) for col in db_columns]  # Use get() to handle missing columns
        cursor.execute(insert_sql, values)

def main():
    args = parse_args()
    
    # Create output directory if it doesn't exist
    output_dir = args.output.parent
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Remove existing database if it exists
    if args.output.exists():
        print(f"\nRemoving existing database at {args.output}")
        args.output.unlink()
    
    # Verify all input databases exist
    for db_path in args.databases:
        if not db_path.exists():
            print(f"Error: Database not found: {db_path}")
            sys.exit(1)
    
    # Create new database with schema from first database
    print("\nCreating new database...")
    merged_conn = create_merged_database(args.output, args.databases[0])
    
    # Merge and insert articles
    print("\nProcessing articles table...")
    articles_entries = merge_table_entries(args.databases, 'articles')
    insert_entries(merged_conn, 'articles', articles_entries)
    
    # Merge and insert sections
    print("\nProcessing sections table...")
    sections_entries = merge_table_entries(args.databases, 'sections')
    insert_entries(merged_conn, 'sections', sections_entries)
    
    merged_conn.commit()
    merged_conn.close()
    
    print(f"\nMerged database created successfully at: {args.output}")
    print("\nPriority order used:")
    for i, db_path in enumerate(args.databases, 1):
        print(f"{i}. {db_path}")

if __name__ == "__main__":
    main() 