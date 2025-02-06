import argparse
import os
import sys
import pandas as pd
import logging
import requests
from tqdm import tqdm
import sqlite3

# Add the project root directory to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from utils.file_utils import generate_output_path

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def get_publisher_from_doi(doi):
    """Get publisher information from Crossref API"""
    base_url = "https://api.crossref.org/works/"
    headers = {
        "User-Agent": "YourApp/1.0 (mailto:your@email.com)"
    }
    
    try:
        response = requests.get(f"{base_url}{doi}", headers=headers)
        response.raise_for_status()
        data = response.json()
        
        if 'message' in data:
            return data['message'].get('publisher', 'Unknown')
        return 'Unknown'
    except Exception as e:
        logger.error(f"Error getting publisher for DOI {doi}: {str(e)}")
        return 'Unknown'

def check_paper_exists(doi, papers_dir):
    """
    Check if either PDF or XML file exists for a given DOI
    """
    if not doi or pd.isna(doi):
        return False
        
    pdf_path = generate_output_path(doi, 'pdf', papers_dir)
    xml_path = generate_output_path(doi, 'xml', papers_dir)
    
    return os.path.exists(pdf_path) or os.path.exists(xml_path)

def check_paper_exists_in_db(doi, db_path):
    """
    Check if paper exists in SQLite database
    """
    if not doi or pd.isna(doi):
        return False
        
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Extract just the DOI part from URLs like https://doi.org/10.1007/s00376-015-5190-8
        doi_suffix = doi.split('doi.org/')[-1] if 'doi.org' in doi else doi
        
        # Check if DOI exists in Reference column
        cursor.execute("SELECT 1 FROM articles WHERE Reference LIKE ?", (f'%{doi_suffix}%',))
        exists = cursor.fetchone() is not None
        
        conn.close()
        return exists
    except sqlite3.Error as e:
        logger.error(f"Database error when checking DOI {doi}: {str(e)}")
        return False

def find_missing_papers(input_csv, papers_source, output_csv, use_db=False):
    """
    Find papers that don't have either PDF/XML files or database entries
    
    Args:
        input_csv: Path to input CSV file
        papers_source: Path to either papers directory or SQLite database
        output_csv: Path to output CSV file
        use_db: Boolean indicating whether to check against database instead of files
    """
    # Read CSV and select only the desired columns
    desired_columns = ['Title', 'DOI', 'Year', 'Journal', 'Citation Count']
    
    try:
        df = pd.read_csv(input_csv)
        # Check if all desired columns exist
        missing_cols = [col for col in desired_columns if col not in df.columns]
        if missing_cols:
            logger.warning(f"Missing columns in input CSV: {missing_cols}")
            # Only select columns that exist
            existing_cols = [col for col in desired_columns if col in df.columns]
            df = df[existing_cols]
        else:
            df = df[desired_columns]

        # Create mask for missing papers
        if use_db:
            df['exists'] = df['DOI'].apply(lambda x: check_paper_exists_in_db(x, papers_source))
        else:
            df['exists'] = df['DOI'].apply(lambda x: check_paper_exists(x, papers_source))
            
        missing_papers = df[~df['exists']].drop(columns=['exists'])
        
        # Add publisher information for missing papers
        logger.info("Fetching publisher information for missing papers...")
        tqdm.pandas(desc="Fetching publisher information")
        missing_papers['Publisher'] = missing_papers['DOI'].progress_apply(get_publisher_from_doi)
        
        # Save to output CSV
        missing_papers.to_csv(output_csv, index=False)
        
        logger.info(f"Total papers in input: {len(df)}")
        logger.info(f"Missing papers: {len(missing_papers)}")
        logger.info(f"Missing papers list saved to: {output_csv}")
        
        return len(df), len(missing_papers)
        
    except Exception as e:
        logger.error(f"Error processing data: {str(e)}")
        raise

def main():
    parser = argparse.ArgumentParser(description="Find papers missing from the papers directory or database")
    parser.add_argument("input_csv", help="Path to input CSV file containing paper information")
    parser.add_argument("papers_source", help="Path to either papers directory or SQLite database")
    parser.add_argument("output_csv", help="Path to save the CSV file containing missing papers")
    parser.add_argument("--use-db", action="store_true", help="Use SQLite database instead of papers directory")
    
    args = parser.parse_args()
    
    # Verify input CSV exists
    if not os.path.exists(args.input_csv):
        logger.error(f"Input CSV file does not exist: {args.input_csv}")
        return
        
    if not os.path.exists(args.papers_source):
        logger.error(f"Source path does not exist: {args.papers_source}")
        return
        
    if not args.use_db:
        # Verify pdf and xml subdirectories exist
        pdf_dir = os.path.join(args.papers_source, 'pdf')
        xml_dir = os.path.join(args.papers_source, 'xml')
        
        if not os.path.exists(pdf_dir) or not os.path.exists(xml_dir):
            logger.error(f"Papers directory must contain 'pdf' and 'xml' subdirectories")
            return
    
    try:
        total, missing = find_missing_papers(args.input_csv, args.papers_source, args.output_csv, args.use_db)
        logger.info(f"Successfully processed {total} papers and found {missing} missing papers")
    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main() 