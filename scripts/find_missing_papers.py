import argparse
import os
import sys
import pandas as pd
import logging
import requests
from tqdm import tqdm

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

def find_missing_papers(input_csv, papers_dir, output_csv):
    """
    Find papers that don't have either PDF or XML files
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
        df['exists'] = df['DOI'].apply(lambda x: check_paper_exists(x, papers_dir))
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
        logger.error(f"Error reading CSV file: {str(e)}")
        raise

def main():
    parser = argparse.ArgumentParser(description="Find papers missing from the papers directory")
    parser.add_argument("input_csv", help="Path to input CSV file containing paper information")
    parser.add_argument("papers_dir", help="Path to directory containing pdf and xml folders")
    parser.add_argument("output_csv", help="Path to save the CSV file containing missing papers")
    
    args = parser.parse_args()
    
    # Verify input CSV exists
    if not os.path.exists(args.input_csv):
        logger.error(f"Input CSV file does not exist: {args.input_csv}")
        return
        
    if not os.path.exists(args.papers_dir):
        logger.error(f"Papers directory does not exist: {args.papers_dir}")
        return
        
    # Verify pdf and xml subdirectories exist
    pdf_dir = os.path.join(args.papers_dir, 'pdf')
    xml_dir = os.path.join(args.papers_dir, 'xml')
    
    if not os.path.exists(pdf_dir) or not os.path.exists(xml_dir):
        logger.error(f"Papers directory must contain 'pdf' and 'xml' subdirectories")
        return
    
    try:
        total, missing = find_missing_papers(args.input_csv, args.papers_dir, args.output_csv)
        logger.info(f"Successfully processed {total} papers and found {missing} missing papers")
    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main() 