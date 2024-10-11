import sys
import os
import pandas as pd
import argparse
import logging

# Add the project root directory to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from utils.download_utils import get_doi_from_title

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def main(input_file, output_file):
    # Read the CSV file
    df = pd.read_csv(input_file)

    # Filter out papers marked as "Excluded"
    df_included = df[df['Include / Exclude'] != 'Excluded'].copy()
    
    logging.info(f"Total papers: {len(df)}")
    logging.info(f"Included papers: {len(df_included)}")
    logging.info(f"Excluded papers: {len(df) - len(df_included)}")

    # Create a new column for DOI right after the title column
    df_included.insert(df_included.columns.get_loc('Title') + 1, 'DOI', '')

    # Iterate through the rows and get the DOI for each paper
    total_papers = len(df_included)
    for index, row in df_included.iterrows():
        title = row['Title']
        try:
            doi = get_doi_from_title(title)
            if doi:
                df_included.at[index, 'DOI'] = doi
                logging.info(f"Processed {index+1}/{total_papers}: DOI found for '{title[:50]}...'")
            else:
                df_included.at[index, 'DOI'] = 'Not Found'
                logging.warning(f"Processed {index+1}/{total_papers}: No DOI found for '{title[:50]}...'")
        except Exception as e:
            logging.error(f"Error processing {index+1}/{total_papers}: '{title[:50]}...': {str(e)}")
            df_included.at[index, 'DOI'] = 'Error: Unable to retrieve DOI'

    # Save the updated DataFrame back to a CSV file
    df_included.to_csv(output_file, index=False)
    logging.info(f"Updated CSV with included papers saved to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Add DOI column to CSV file based on paper titles, excluding marked papers.")
    parser.add_argument("input_file", help="Path to the input CSV file")
    parser.add_argument("output_file", help="Path to save the output CSV file")
    args = parser.parse_args()

    main(args.input_file, args.output_file)