import os
import sys
import urllib.parse
import argparse

# Add the project root directory to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from utils.file_utils import sanitize_filename

def rename_pdfs(directory):
    """Rename all PDFs in the directory from URL-encoded to sanitized names."""
    print(f"Processing PDFs in {directory}...")
    
    # Keep track of how many files we process
    count = 0
    errors = 0
    
    for filename in os.listdir(directory):
        if filename.endswith('.pdf'):
            old_path = os.path.join(directory, filename)
            
            # Decode the URL-encoded filename and sanitize it
            decoded_name = urllib.parse.unquote(filename)
            base_name = os.path.splitext(decoded_name)[0]
            new_filename = sanitize_filename(base_name) + '.pdf'
            new_path = os.path.join(directory, new_filename)
            
            try:
                if old_path != new_path:  # Only rename if the name would actually change
                    os.rename(old_path, new_path)
                    print(f"Renamed: {filename} -> {new_filename}")
                    count += 1
            except OSError as e:
                print(f"Error renaming {filename}: {e}")
                errors += 1
    
    print(f"\nComplete! Renamed {count} files with {errors} errors.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Rename PDF files from URL-encoded to sanitized names')
    parser.add_argument('directory', help='Directory containing PDFs to rename')
    args = parser.parse_args()
    
    if not os.path.isdir(args.directory):
        print(f"Error: {args.directory} is not a valid directory")
        exit(1)
        
    rename_pdfs(args.directory) 