import sqlite3
import argparse
from pathlib import Path

def get_unique_references(db_path):
    """Get unique references from a SQLite database."""
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Get all unique references
        cursor.execute("SELECT DISTINCT Reference FROM articles WHERE Reference IS NOT NULL")
        references = {row[0] for row in cursor.fetchall()}
        
        conn.close()
        return references
    except sqlite3.Error as e:
        print(f"Error accessing database {db_path}: {e}")
        return set()

def parse_args():
    parser = argparse.ArgumentParser(description='Analyze references across multiple SQLite databases')
    parser.add_argument('databases', nargs='+', type=Path, help='Paths to SQLite databases to analyze')
    parser.add_argument('-o', '--output', type=Path, default=Path("reference_analysis_results.txt"),
                      help='Output path for analysis results (default: reference_analysis_results.txt)')
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Verify all databases exist
    for db_path in args.databases:
        if not db_path.exists():
            print(f"Error: Database not found: {db_path}")
            return
    
    # Get references from each database
    print("\nAnalyzing databases...")
    db_refs = {}
    for db_path in args.databases:
        refs = get_unique_references(db_path)
        db_refs[db_path.name] = refs
        print(f"{db_path.name}: {len(refs)} unique references")
    
    # Combine all references
    all_refs = set().union(*db_refs.values())
    print(f"\nTotal unique references across all databases: {len(all_refs)}")
    
    # Analyze overlaps between all pairs
    print("\nOverlap Analysis:")
    db_names = list(db_refs.keys())
    for i in range(len(db_names)):
        for j in range(i + 1, len(db_names)):
            name1, name2 = db_names[i], db_names[j]
            common = db_refs[name1].intersection(db_refs[name2])
            print(f"References in both {name1} and {name2}: {len(common)}")
    
    # Find references common to all databases
    common_all = set.intersection(*db_refs.values())
    print(f"\nReferences in all databases: {len(common_all)}")
    
    # Save results to file
    print(f"\nSaving results to {args.output}")
    with open(args.output, "w") as f:
        f.write("=== Reference Analysis Results ===\n\n")
        for name, refs in db_refs.items():
            f.write(f"{name}: {len(refs)} unique references\n")
        f.write(f"\nTotal unique references: {len(all_refs)}\n")
        
        f.write("\n=== Overlap Analysis ===\n")
        for i in range(len(db_names)):
            for j in range(i + 1, len(db_names)):
                name1, name2 = db_names[i], db_names[j]
                common = db_refs[name1].intersection(db_refs[name2])
                f.write(f"References in {name1} and {name2}: {len(common)}\n")
        f.write(f"\nReferences in all databases: {len(common_all)}\n")

if __name__ == "__main__":
    main()