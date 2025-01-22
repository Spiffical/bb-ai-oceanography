import os
import logging
import argparse
import requests
import sys
import openai
import base64
from habanero import Crossref
import re
from shutil import move

# Add the project root directory to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from utils.download_utils import get_doi_from_title

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def get_paper_title_from_doi(doi):
    """Retrieve the official paper title from CrossRef using the DOI"""
    try:
        headers = {'Accept': 'application/json'}
        response = requests.get(f'https://api.crossref.org/works/{doi}', headers=headers)
        response.raise_for_status()
        data = response.json()
        return data['message']['title'][0]
    except Exception as e:
        logger.error(f"Error retrieving title for DOI {doi}: {str(e)}")
        return None

def extract_title_or_doi_with_gpt(filename):
    """Use GPT-4-mini to extract either a paper title or DOI from filename"""
    try:
        # Create a prompt that asks the model to identify if it's a DOI or title
        response = openai.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{
                "role": "system",
                "content": "You are a helpful assistant that extracts paper titles or DOIs from filenames."
            }, {
                "role": "user", 
                "content": f"""Analyze this filename: "{filename}"
                If this is a DOI (e.g., "10.1038/s41467-019-12808-z"), return exactly: DOI: <the doi>
                If this contains a paper title, extract and return only the paper title, removing any file extensions, version numbers, or download markers.
                Return only the extracted information with no additional text. Titles will contain full sentences, not things
                like omar.202 or arma-2022-441 etc."""
            }]
        )
        
        extracted_text = response.choices[0].message.content.strip()
        
        if extracted_text.startswith("DOI:"):
            # Return tuple (None, doi) if DOI found
            return None, extracted_text.replace("DOI:", "").strip()
        else:
            # Return tuple (title, None) if title found
            return extracted_text, None
            
    except Exception as e:
        logger.error(f"Error in GPT extraction for {filename}: {str(e)}")
        return None, None

def extract_title_from_pdf(filepath, max_pages=3):
    """Extract title from PDF using GPT through Assistants API"""
    try:
        # Create an assistant for PDF processing
        pdf_assistant = openai.beta.assistants.create(
            model="gpt-4o-mini",
            description="An assistant to extract paper titles from PDF files.",
            tools=[{"type": "file_search"}],
            name="PDF assistant"
        )

        # Create thread
        thread = openai.beta.threads.create()

        # Upload file
        with open(filepath, "rb") as file:
            uploaded_file = openai.files.create(
                file=file,
                purpose="assistants"
            )

        # Create message with file attachment
        openai.beta.threads.messages.create(
            thread_id=thread.id,
            role="user",
            attachments=[{
                "file_id": uploaded_file.id,
                "tools": [{"type": "file_search"}]
            }],
            content="Extract only the academic paper title from this PDF. Return only the title with no additional text."
        )

        # Run thread and wait for completion
        run = openai.beta.threads.runs.create_and_poll(
            thread_id=thread.id,
            assistant_id=pdf_assistant.id,
            timeout=300
        )

        if run.status != "completed":
            raise Exception("Run failed:", run.status)

        # Get response
        messages = openai.beta.threads.messages.list(thread_id=thread.id)
        extracted_title = messages.data[0].content[0].text.value.strip()
        
        logger.info(f"Extracted title from PDF: {extracted_title}")
        return extracted_title

    except Exception as e:
        logger.error(f"Error extracting title from PDF: {str(e)}")
        return None

def get_doi_from_title(title):
    """Simple function to find DOI using paper title via Crossref"""
    try:
        # Initialize Crossref client
        cr = Crossref()
        
        # Search for the paper
        results = cr.works(query=title, limit=1)
        
        # Check if we got any results
        if results['message']['total-results'] > 0:
            # Get the first result
            paper = results['message']['items'][0]
            
            # Get DOI and title for verification
            found_doi = paper.get('DOI')
            found_title = paper.get('title', [None])[0]
            
            if found_doi and found_title:
                logger.info(f"Found potential match:")
                logger.info(f"Input title: {title}")
                logger.info(f"Found title: {found_title}")
                return found_doi
            
        logger.warning(f"No DOI found for title: {title}")
        return None
        
    except Exception as e:
        logger.error(f"Error searching for DOI: {str(e)}")
        return None

def sanitize_filename(filename):
    """Remove or replace characters that are unsafe for filenames."""
    logger.debug(f"Sanitizing filename: {repr(filename)}")
    # Remove null characters
    filename = filename.replace('\0', '')
    # Replace other unsafe characters with underscore
    sanitized = re.sub(r'[<>:"/\\|?*]', '_', filename)
    logger.debug(f"Sanitized filename: {repr(sanitized)}")
    return sanitized

def process_directory(input_dir):
    """Process all PDF files in the input directory and attempt to find their DOIs."""
    matches = []
    no_doi_found = []
    retrieval_errors = []
    
    # Process all PDF files
    for filename in os.listdir(input_dir):
        if not filename.lower().endswith('.pdf'):
            continue
            
        filepath = os.path.join(input_dir, filename)
        
        logger.info(f"\nProcessing: {filename}")
        
        try:
            # Comment out filename-based extraction for now
            # extracted_text, is_doi = extract_title_or_doi_with_gpt(filename)
            # 
            # if is_doi:
            #     doi = extracted_text
            #     logger.info(f"DOI found directly in filename: {doi}")
            # else:
            #     if extracted_text:
            #         logger.info(f"Extracted title from filename: {extracted_text}")
            #         doi = get_doi_from_title(extracted_text)
            #     else:
            #         doi = None
            
            # Try extracting from PDF directly
            pdf_title = extract_title_from_pdf(filepath)
            if pdf_title:
                logger.info(f"Attempting to find DOI using title from PDF: {pdf_title}")
                doi = get_doi_from_title(pdf_title)
                if doi:
                    extracted_text = pdf_title
            else:
                doi = None
            
            if doi:
                official_title = get_paper_title_from_doi(doi)
                if official_title:
                    # Create new filename with DOI
                    safe_doi = sanitize_filename(doi)
                    new_filename = f"{safe_doi}.pdf"
                    new_filepath = os.path.join(input_dir, new_filename)
                    
                    # Rename the file
                    try:
                        move(filepath, new_filepath)
                        logger.info(f"Renamed file to: {new_filename}")
                    except Exception as e:
                        logger.error(f"Error renaming file {filename}: {str(e)}")
                    
                    matches.append({
                        'original_filename': filename,
                        'new_filename': new_filename,
                        'extracted_text': extracted_text,
                        'doi': doi,
                        'official_title': official_title
                    })
                    logger.info(f"Found DOI: {doi}")
                    logger.info(f"Official title: {official_title}")
                else:
                    retrieval_errors.append((filename, doi))
            else:
                logger.warning(f"No DOI found for: {filename}")
                no_doi_found.append(filename)
                
        except Exception as e:
            logger.error(f"Error processing {filename}: {str(e)}")
            retrieval_errors.append((filename, str(e)))

    # Print summary
    logger.info("\n" + "="*80)
    logger.info("SUMMARY")
    logger.info("="*80)
    
    if matches:
        logger.info("\nSuccessful matches:")
        logger.info("-"*80)
        for match in matches:
            logger.info(f"\nOriginal file: {match['original_filename']}")
            logger.info(f"New filename: {match['new_filename']}")
            logger.info(f"DOI: {match['doi']}")
            logger.info(f"Extracted text: {match['extracted_text']}")
            logger.info(f"Official title: {match['official_title']}")
            logger.info("-"*80)
    
    if no_doi_found:
        logger.info("\nFiles with no DOI found:")
        for filename in no_doi_found:
            logger.info(f"- {filename}")
    
    if retrieval_errors:
        logger.info("\nFiles with errors:")
        for filename, error in retrieval_errors:
            logger.info(f"- {filename}: {error}")
    
    logger.info(f"\nTotal files processed: {len(matches) + len(no_doi_found) + len(retrieval_errors)}")
    logger.info(f"Successful matches: {len(matches)}")
    logger.info(f"No DOI found: {len(no_doi_found)}")
    logger.info(f"Errors: {len(retrieval_errors)}")

def main():
    parser = argparse.ArgumentParser(description="Verify DOIs and titles for PDF files")
    parser.add_argument("input_dir", help="Directory containing PDF files")
    args = parser.parse_args()
    
    process_directory(args.input_dir)

if __name__ == "__main__":
    main() 