import os
import logging
import csv
import sys
from dotenv import load_dotenv
import argparse

# Add the project root directory to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from utils.download_utils import (
    get_full_text_elsevier,
    get_full_text_springer,
    get_full_text_wiley,
    get_full_text_arxiv,
    get_full_text_unpaywall,
    get_pdf_url,
    extract_pdf_url_from_xml,
    download_pdf_from_url,
    is_xml_only_pdf_link,
    extract_redirect_url,
    construct_full_url,
)

from utils.api_utils import rate_limited_api_call
from utils.file_utils import generate_output_path
from utils.pdf_utils import is_valid_pdf

# Load environment variables
load_dotenv()

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def download_full_text(doi, title, base_output_dir):
    """
    Attempt to download both XML and PDF of a paper given its DOI and title using various methods.
    
    Args:
    doi (str): The DOI of the paper.
    title (str): The title of the paper.
    base_output_dir (str): The base output directory.
    
    Returns:
    tuple: (bool, bool) indicating success of (XML, PDF) downloads.
    """
    logger.debug(f"Entering download_full_text with DOI: {repr(doi)}, title: {repr(title)}, base_output_dir: {repr(base_output_dir)}")
    
    # Check if files already exist
    xml_output_path = generate_output_path(doi, 'xml', base_output_dir)
    pdf_output_path = generate_output_path(doi, 'pdf', base_output_dir)
    
    xml_exists = os.path.exists(xml_output_path)
    pdf_exists = os.path.exists(pdf_output_path)
    
    if xml_exists and pdf_exists:
        logger.info(f"Both XML and PDF already exist for DOI {doi}. Skipping download.")
        return True, True
    
    xml_success = xml_exists
    pdf_success = pdf_exists
    
    if xml_exists:
        logger.info(f"XML already exists for DOI {doi}. Skipping XML download.")
    if pdf_exists:
        logger.info(f"PDF already exists for DOI {doi}. Skipping PDF download.")

    # Check availability of each method
    method_availability = {
        "Elsevier API": bool(os.getenv('ELSEVIER_API_KEY')),
        "Springer API": bool(os.getenv('SPRINGER_API_KEY')),
        "Wiley API": bool(os.getenv('WILEY_API_KEY')),
        "arXiv API": True,  # arXiv doesn't require an API key
        "Unpaywall API": bool(os.getenv('UNPAYWALL_EMAIL')),
        "DOI resolution": True
    }
    
    methods = [
        (get_full_text_elsevier, "Elsevier API"),
        (get_full_text_springer, "Springer API"),
        (get_full_text_wiley, "Wiley API"),
        (lambda d, t: get_full_text_arxiv(d, t), "arXiv API"),
        (lambda d: get_full_text_unpaywall(d, base_output_dir), "Unpaywall API"),
        (get_pdf_url, "DOI resolution")
    ]
    
    for method, source in methods:
        if xml_success and pdf_success:
            break
        
        if not method_availability[source]:
            logger.info(f"Skipping {source} as the required API key or email is not set.")
            continue
        
        try:
            logger.info(f"Attempting to retrieve content from {source}")
            if source == "arXiv API":
                success, content = rate_limited_api_call(method, doi, title)
            else:
                success, content = rate_limited_api_call(method, doi)
            logger.debug(f"Result from {source}: success={success}, content type={type(content)}")
            
            if success:
                if isinstance(content, str):
                    # Handle case where content is a string (possibly XML, HTML, or PDF link)
                    if content.lower().endswith('.pdf') or 'pdf' in content.lower() or is_xml_only_pdf_link(content):
                        if not pdf_success:
                            pdf_url = construct_full_url(content, source)
                            pdf_output_path = generate_output_path(doi, 'pdf', base_output_dir)
                            pdf_success, message = download_pdf_from_url(pdf_url, pdf_output_path)
                            logger.info(message)
                            if is_xml_only_pdf_link(content):
                                logger.info(f"Content for DOI {doi} only contained a PDF link. Not saving XML.")
                            else:
                                logger.info(f"Downloaded PDF from direct link for DOI {doi}")
                    elif content.strip().startswith('<!DOCTYPE HTML'):
                        # This might be HTML content with a redirect
                        redirect_url = extract_redirect_url(content)
                        if redirect_url and not pdf_success:
                            logger.info(f"Extracted redirect URL: {redirect_url}")
                            pdf_output_path = generate_output_path(doi, 'pdf', base_output_dir)
                            pdf_success, message = download_pdf_from_url(redirect_url, pdf_output_path)
                            logger.info(message)
                        else:
                            logger.warning("Failed to extract redirect URL from HTML content")
                    elif not xml_success:
                        # Treat as XML content
                        xml_output_path = generate_output_path(doi, 'xml', base_output_dir)
                        with open(xml_output_path, 'w', encoding='utf-8') as f:
                            f.write(content)
                        logger.info(f"XML saved successfully to {xml_output_path}")
                        xml_success = True
                        
                        # Check if XML contains a PDF link
                        pdf_url = extract_pdf_url_from_xml(content)
                        if pdf_url and not pdf_success:
                            pdf_url = construct_full_url(pdf_url, source)
                            pdf_output_path = generate_output_path(doi, 'pdf', base_output_dir)
                            pdf_success, message = download_pdf_from_url(pdf_url, pdf_output_path)
                            logger.info(message)
                            logger.info(f"XML for DOI {doi} contained a PDF link. Attempted to download PDF.")

                elif isinstance(content, dict):
                    # Handle PDF content
                    if 'pdf' in content and not pdf_success:
                        pdf_content = content['pdf']
                        if isinstance(pdf_content, bytes) and is_valid_pdf(pdf_content):
                            pdf_output_path = generate_output_path(doi, 'pdf', base_output_dir)
                            with open(pdf_output_path, 'wb') as f:
                                f.write(pdf_content)
                            logger.info(f"PDF saved successfully to {pdf_output_path}")
                            pdf_success = True
                        else:
                            logger.warning(f"Invalid PDF content for DOI {doi} from {source}")
                    
                    # Handle XML content
                    if 'xml' in content and not xml_success:
                        xml_content = content['xml']
                        if isinstance(xml_content, str):
                            pdf_url = extract_pdf_url_from_xml(xml_content)
                            if pdf_url and not pdf_success:
                                pdf_url = construct_full_url(pdf_url, source)
                                pdf_output_path = generate_output_path(doi, 'pdf', base_output_dir)
                                pdf_success, message = download_pdf_from_url(pdf_url, pdf_output_path)
                                logger.info(message)
                                logger.info(f"XML for DOI {doi} contained a PDF link. Attempted to download PDF.")
                            
                            # Save XML content even if it contains a PDF link
                            xml_output_path = generate_output_path(doi, 'xml', base_output_dir)
                            with open(xml_output_path, 'w', encoding='utf-8') as f:
                                f.write(xml_content)
                            logger.info(f"XML saved successfully to {xml_output_path}")
                            xml_success = True
                        else:
                            logger.warning(f"Invalid XML content for DOI {doi} from {source}")
                
                elif isinstance(content, bytes) and not pdf_success:
                    if is_valid_pdf(content):
                        pdf_output_path = generate_output_path(doi, 'pdf', base_output_dir)
                        with open(pdf_output_path, 'wb') as f:
                            f.write(content)
                        logger.info(f"PDF saved successfully to {pdf_output_path}")
                        pdf_success = True
                    else:
                        logger.warning(f"Invalid PDF content for DOI {doi} from {source}")
                
                if xml_success and pdf_success:
                    logger.info(f"Both XML and PDF successfully downloaded for DOI {doi}")
                    break
            
            else:
                logger.info(f"Failed to retrieve content from {source}")
        
        except Exception as e:
            logger.error(f"Error occurred while trying to retrieve content from {source}: {str(e)}", exc_info=True)

    logger.debug(f"Exiting download_full_text with results: xml_success={xml_success}, pdf_success={pdf_success}")
    return xml_success, pdf_success

def process_csv(csv_path, base_output_dir):
    """
    Process the input CSV file and attempt to download XML and PDF for each DOI.
    
    Args:
    csv_path (str): Path to the input CSV file.
    base_output_dir (str): The base output directory.
    """
    with open(csv_path, 'r') as csvfile:
        reader = csv.DictReader(csvfile)
        if 'DOI' not in reader.fieldnames or 'Title' not in reader.fieldnames:
            logger.error("CSV file does not contain both 'DOI' and 'Title' columns")
            return

        for row in reader:
            doi = row['DOI']
            title = row['Title']
            logger.info(f"Processing DOI: {doi}, Title: {title}")
            xml_success, pdf_success = download_full_text(doi, title, base_output_dir)
            
            if xml_success and pdf_success:
                logger.info(f"Successfully downloaded both XML and PDF for DOI {doi}")
            elif xml_success:
                logger.info(f"Successfully downloaded XML, but failed to download PDF for DOI {doi}")
            elif pdf_success:
                logger.info(f"Successfully downloaded PDF, but failed to download XML for DOI {doi}")
            else:
                logger.error(f"Failed to download both XML and PDF for DOI {doi}")

def parse_arguments():
    parser = argparse.ArgumentParser(description="Download scientific papers.")
    parser.add_argument("--csv", help="Path to CSV file containing DOIs")
    parser.add_argument("--doi", help="Single DOI to download")
    parser.add_argument("--title", help="Single title to search for")
    parser.add_argument("output_path", help="Path to save downloaded papers")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_arguments()
    
    if args.csv and not os.path.exists(args.csv):
        print(f"Error: CSV file '{args.csv}' does not exist.")
        sys.exit(1)
    
    if args.csv:
        process_csv(args.csv, args.output_path)
        print("Finished processing all DOIs in the CSV file.")
    else:
        if args.doi:
            print(f"Processing single DOI: {args.doi}")
            xml_success, pdf_success = download_full_text(args.doi, args.title, args.output_path)
        elif args.title:
            print(f"Processing single title: {args.title}")
            xml_success, pdf_success = download_full_text(None, args.title, args.output_path)
        
        if xml_success and pdf_success:
            print(f"Successfully downloaded both XML and PDF for {'DOI ' + args.doi if args.doi else 'title: ' + args.title}")
        elif xml_success:
            print(f"Successfully downloaded XML, but failed to download PDF for {'DOI ' + args.doi if args.doi else 'title: ' + args.title}")
        elif pdf_success:
            print(f"Successfully downloaded PDF, but failed to download XML for {'DOI ' + args.doi if args.doi else 'title: ' + args.title}")
        else:
            print(f"Failed to download both XML and PDF for {'DOI ' + args.doi if args.doi else 'title: ' + args.title}")