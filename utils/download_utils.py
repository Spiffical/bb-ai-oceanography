import os
from habanero import Crossref
import glob
import shutil
import requests
from utils.pdf_utils import is_valid_pdf
from utils.file_utils import clean_title
from bs4 import BeautifulSoup
import time
import xml.etree.ElementTree as ET
import logging
import random
from dotenv import load_dotenv
import xml.etree.ElementTree as ET
from functools import wraps
from unpywall import Unpywall
from unpywall.utils import UnpywallCredentials
import re
from urllib.parse import urljoin, unquote, urlparse, urljoin

from utils.file_utils import sanitize_filename, titles_are_similar

# Load environment variables
load_dotenv()

# Set up Unpaywall credentials
UnpywallCredentials(os.getenv('UNPAYWALL_EMAIL'))

ELSEVIER_API_KEY = os.getenv('ELSEVIER_API_KEY')
SPRINGER_API_KEY = os.getenv('SPRINGER_API_KEY')
WILEY_API_KEY = os.getenv('WILEY_API_KEY')

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def retry_on_exception(max_retries=3, delay=1):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    if attempt == max_retries - 1:
                        raise
                    logger.warning(f"Attempt {attempt + 1} failed: {str(e)}. Retrying in {delay} seconds...")
                    time.sleep(delay)
        return wrapper
    return decorator

class NotFoundError(Exception):
    pass

def find_doi_with_habanero(title):
    """
    Search for a DOI using the Crossref API via the habanero library.

    This function queries the Crossref API with a given title and attempts to find a matching DOI.
    It retrieves up to 5 results and checks if any of them match the cleaned version of the input title.

    Args:
        title (str): The title of the paper to search for.

    Returns:
        tuple: A tuple containing three elements:
            - doi (str or None): The DOI of the matching paper, if found.
            - url (str or None): The URL constructed from the DOI, if found.
            - t (str or None): The title of the matching paper from the Crossref results, if found.

    Note:
        This function uses the clean_title() function to normalize titles for comparison.
    """
    cr = Crossref()
    results = cr.works(query=title, limit=5)  # Retrieve more results to find the exact match
    
    clean_search_title = clean_title(title)
    
    for item in results['message']['items']:
        if item.get('title'):
            for t in item.get('title'):
                clean_item_title = clean_title(t)
                if clean_search_title in clean_item_title:
                    doi = item.get('DOI')
                    url = f"https://doi.org/{doi}"
                    return doi, url, t
    
    logging.info("No DOI found with habanero.")
    return None, None, None

def find_doi_with_openalex(title):
    """Returns metadata of a paper with http://openalex.org/"""
    api_res = requests.get(
        f"https://api.openalex.org/works?search={title}&per-page=1&page=1&sort=relevance_score:desc"
    )

    if api_res.status_code != 200:
        raise NotFoundError("Paper not found.")

    metadata = api_res.json()
    if metadata.get("results") and len(metadata["results"]) > 0:
        metadata = metadata["results"][0]

        if metadata.get("doi"):
            doi = metadata["doi"][len("https://doi.org/"):]
            url = f"https://doi.org/{doi}"
            found_title = metadata.get("title", "")
            return doi, url, found_title

    logging.info(f"No results or DOI found for the title: '{title}'")
    return None, None, None

def download_pdf_directly(pdf_url, pdf_path):
    try:
        response = requests.get(pdf_url, stream=True)
        if response.status_code == 200:
            with open(pdf_path, 'wb') as file:
                file.write(response.content)
            print(f"PDF downloaded successfully to {pdf_path}")
        else:
            print(f"Failed to download PDF. Status code: {response.status_code}")
    except Exception as e:
        print(f"Error during PDF download: {e}")


def handle_downloaded_file(temp_dir, download_path):
    # Find the downloaded PDF file in the temp directory
    pdf_files = glob.glob(os.path.join(temp_dir, "*.pdf"))
    if pdf_files:
        downloaded_pdf = pdf_files[0]
        shutil.move(downloaded_pdf, download_path)
        if is_valid_pdf(download_path):
            print(f"Downloaded paper saved to {download_path}")
            return True, ""
        else:
            print(f"Downloaded PDF is invalid or corrupted")
            os.remove(download_path)  # Remove the invalid file
            return False, "Downloaded PDF is invalid or corrupted"
    else:
        print(f"No PDF found in {temp_dir}")
        return False, "No PDF found in directory (did not download)."
    
    
def get_full_text_elsevier(doi):
    """
    Retrieve the full text content for a given DOI using the Elsevier API.

    This function attempts to fetch both XML and PDF content for the specified article.

    Args:
        doi (str): The Digital Object Identifier (DOI) of the article.

    Returns:
        tuple: A tuple containing two elements:
            - bool: True if the retrieval was successful, False otherwise.
            - dict or str: If successful, a dictionary containing 'xml' and optionally 'pdf' keys.
                           If unsuccessful, an error message string.
    """
    base_url = "https://api.elsevier.com/content/article/doi/"
    headers = {
        'X-ELS-APIKey': ELSEVIER_API_KEY,
        'Accept': 'text/xml'
    }
    
    # Request XML content
    xml_url = f"{base_url}{doi}?httpAccept=text/xml&view=FULL"
    xml_response = requests.get(xml_url, headers=headers)
    
    if xml_response.status_code == 200:
        xml_content = xml_response.text
        
        # Try to get PDF
        pdf_url = f"{base_url}{doi}?httpAccept=application/pdf"
        pdf_headers = headers.copy()
        pdf_headers['Accept'] = 'application/pdf'
        pdf_response = requests.get(pdf_url, headers=pdf_headers)
        
        if pdf_response.status_code == 200:
            return True, {"xml": xml_content, "pdf": pdf_response.content}
        else:
            logger.warning(f"PDF retrieval failed for DOI {doi}. Status code: {pdf_response.status_code}")
            return True, {"xml": xml_content}
    else:
        error_message = f"Failed to retrieve the article. Status code: {xml_response.status_code}"
        logger.error(error_message)
        return False, error_message

def save_full_text_to_file(full_text, file_path):
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, 'w', encoding='utf-8') as file:
            file.write(full_text)
        logger.info(f"\n========================\nFull text saved successfully to {file_path}\n========================\n")
    except Exception as e:
        logger.error(f"Failed to save full text to {file_path}. Error: {e}")


def get_full_text_unpaywall(doi, base_output_dir):
    """
    Attempt to retrieve the full text PDF for a given DOI using Unpaywall.

    This function tries to find a PDF link for the given DOI using Unpaywall,
    downloads the PDF if found, and saves it to the specified output directory.

    Args:
        doi (str): The Digital Object Identifier (DOI) of the paper.
        base_output_dir (str): The base directory where the PDF should be saved.

    Returns:
        tuple: A tuple containing two elements:
            - bool: True if the retrieval was successful, False otherwise.
            - dict or str: If successful, a dictionary containing the 'pdf' key with the PDF content.
                           If unsuccessful, an error message string.

    Raises:
        Exception: Any unexpected errors during the process are caught and logged.
    """
    logger.debug(f"Entering get_full_text_unpaywall with DOI: {repr(doi)}, base_output_dir: {repr(base_output_dir)}")
    try:
        # Get the PDF link from Unpywall
        pdf_link = Unpywall.get_pdf_link(doi=doi)
        logger.debug(f"PDF link from Unpywall: {repr(pdf_link)}")
        
        if pdf_link:
            logger.info(f"Found PDF link for DOI {doi}: {pdf_link}")
            
            # Create the pdf folder if it doesn't exist
            pdf_folder = os.path.join(base_output_dir, 'pdf')
            logger.debug(f"PDF folder path: {repr(pdf_folder)}")
            os.makedirs(pdf_folder, exist_ok=True)
            
            # Sanitize the DOI for use in filename
            safe_doi = sanitize_filename(doi)
            
            # Use our new download_pdf_from_url function
            output_file = os.path.join(pdf_folder, f"{safe_doi}.pdf")
            logger.debug(f"Output file path: {repr(output_file)}")
            
            success, message = download_pdf_from_url(pdf_link, output_file)
            
            if success and is_valid_pdf(output_file):
                logger.info(f"Successfully downloaded valid PDF for DOI {doi}")
                with open(output_file, 'rb') as f:
                    content = f.read()
                return True, {"pdf": content}
            else:
                logger.warning(f"Failed to download PDF for DOI {doi} or PDF is invalid. Error: {message}")
                if os.path.exists(output_file):
                    os.remove(output_file)
                return False, "Failed to download valid PDF"
        else:
            logger.info(f"No PDF link found for DOI {doi} in Unpaywall")
    except Exception as e:
        logger.error(f"Unexpected error in get_full_text_unpaywall for DOI {doi}: {str(e)}", exc_info=True)
    
    return False, "Full text not found or invalid in Unpaywall"

def get_pdf_url(doi):
    url = f"https://doi.org/{doi}"
    response = requests.get(url)
    if response.status_code == 200:
        soup = BeautifulSoup(response.text, 'html.parser')
        # print('\n =======================\n')
        # print(soup)
        # print('\n =======================\n')
        pdf_link = soup.find('a', href=lambda href: href and href.endswith('.pdf'))
        if pdf_link:
            return True, pdf_link['href']
    return False, "PDF URL not found"

def download_pdf(url):
    response = requests.get(url)
    if response.status_code == 200:
        return response.content
    return None


def get_full_text_springer(doi):
    """
    Retrieve full text content from Springer API for a given DOI.

    This function attempts to fetch JATS XML content for the specified article using the Springer API.

    Args:
        doi (str): The Digital Object Identifier (DOI) of the article.

    Returns:
        tuple: A tuple containing two elements:
            - bool: True if the retrieval was successful, False otherwise.
            - dict or str: If successful, a dictionary containing 'xml' key with the JATS XML content.
                           If unsuccessful, an error message string.

    Raises:
        requests.exceptions.RequestException: For network-related errors during the API request.
        Exception: For any unexpected errors during the API request.
    """
    base_url = 'https://api.springernature.com/openaccess/jats'
    params = {
        "q": f'doi:"{doi}"',
        "api_key": SPRINGER_API_KEY,
        "p": 1  # Limit to 1 result since we're querying a specific DOI
    }
    
    try:
        response = requests.get(base_url, params=params)
        response.raise_for_status()
        
        if response.status_code == 200:
            xml_content = response.text
            
            # Check if the response actually contains JATS content
            if '<article' in xml_content:
                return True, {"xml": xml_content}
            else:
                return False, f"No open access JATS content found for DOI: {doi}"
        else:
            return False, f"No records found for DOI: {doi}"
        
    except requests.exceptions.RequestException as e:
        logger.error(f"Network error when querying Springer API: {str(e)}")
        return False, f"Network error: {str(e)}"
    except Exception as e:
        logger.error(f"Unexpected error querying Springer API: {str(e)}")
        return False, f"Unexpected error: {str(e)}"

def get_full_text_wiley(doi):
    """
    Retrieve full text content from Wiley API for a given DOI.

    This function attempts to fetch both XML and PDF content for the specified article.

    Args:
        doi (str): The Digital Object Identifier (DOI) of the article.

    Returns:
        tuple: A tuple containing two elements:
            - bool: True if the retrieval was successful, False otherwise.
            - dict or str: If successful, a dictionary containing 'xml' and optionally 'pdf' keys with their respective content.
                           If unsuccessful, an error message string.

    Raises:
        Exception: Any unexpected errors during the API request are caught and logged.
    """
    base_url = f"https://api.wiley.com/onlinelibrary/tdm/v1/articles/{doi}"
    
    try:
        headers = {
            "Wiley-TDM-Client-Token": WILEY_API_KEY,
            "Accept": "application/xml"
        }

        response = requests.get(base_url, headers=headers)

        if response.status_code == 200:
            xml_content = response.content
            
            # Try to get PDF
            pdf_headers = {
                "Wiley-TDM-Client-Token": WILEY_API_KEY,
                "Accept": "application/pdf"
            }
            pdf_response = requests.get(base_url, headers=pdf_headers)
            
            if pdf_response.status_code == 200:
                return True, {"xml": xml_content, "pdf": pdf_response.content}
            else:
                return True, {"xml": xml_content}
        
        return False, "Full text not found in Wiley"
    except Exception as e:
        logger.error(f"Error querying Wiley API: {str(e)}")
        return False, f"Error: {str(e)}"

def get_full_text_arxiv(doi=None, title=None, max_retries=5, initial_wait=1, max_wait=60):
    """
    Retrieve full text content from arXiv for a given DOI or title.

    This function attempts to fetch PDF content for the specified article from arXiv.
    It uses an exponential backoff strategy for retrying in case of network errors.

    Args:
        doi (str, optional): The Digital Object Identifier (DOI) of the article.
        title (str, optional): The title of the article.
        max_retries (int, optional): Maximum number of retry attempts. Defaults to 5.
        initial_wait (int, optional): Initial wait time in seconds before first retry. Defaults to 1.
        max_wait (int, optional): Maximum wait time in seconds between retries. Defaults to 60.

    Returns:
        tuple: A tuple containing two elements:
            - bool: True if the retrieval was successful, False otherwise.
            - dict or str: If successful, a dictionary containing 'pdf' key with the PDF content.
                           If unsuccessful, an error message string.

    Raises:
        Exception: Any unexpected errors during the API request are caught and logged.
    """
    import arxiv
    
    if doi:
        search_query = f'doi:{doi}'
    elif title:
        search_query = f'ti:"{title}"'
    else:
        return False, "Either DOI or title must be provided"

    client = arxiv.Client()
    
    wait_time = initial_wait
    for attempt in range(max_retries):
        try:
            logger.info(f"Attempting to query arXiv (attempt {attempt + 1})")
            time.sleep(wait_time + random.uniform(0, 1))
            
            search = arxiv.Search(query=search_query, max_results=1)
            results = list(client.results(search))
            
            if results:
                paper = results[0]
                print(f"Paper title: {paper.title}")
                
                # Compare titles if a title was provided
                if title and not titles_are_similar(title, paper.title):
                    logger.warning(f"Found paper title '{paper.title}' does not match the provided title '{title}'")
                    return False, "Title mismatch"
                
                # Download PDF
                pdf_filename = f"{paper.get_short_id()}.pdf"
                paper.download_pdf(filename=pdf_filename)
                
                with open(pdf_filename, 'rb') as pdf_file:
                    pdf_content = pdf_file.read()
                
                # Remove the temporary PDF file
                os.remove(pdf_filename)
                
                return True, {"pdf": pdf_content}
            
            logger.info("No matching entry found in arXiv response")
            return False, "Full text not found in arXiv"

        except Exception as e:
            logger.warning(f"Network error when querying arXiv (attempt {attempt + 1}): {str(e)}")
            if attempt == max_retries - 1:
                return False, f"Network error: {str(e)}"
            wait_time = min(max_wait, wait_time * 2)

    return False, "Max retries reached"


def get_doi_from_title(title):
    logging.info(f"Attempting to find DOI for title: '{title}'")
    
    # First, try to find DOI using habanero
    doi, url, found_title = find_doi_with_habanero(title)
    if doi:
        logging.info(f"DOI found with habanero: {doi}")
        return doi
    
    # If not found with habanero or title doesn't match, try OpenAlex
    doi, url, found_title = find_doi_with_openalex(title)
    if doi:
        logging.info(f"DOI found with OpenAlex: {doi}")
        return doi
    
    logging.warning(f"No matching DOI found for title: '{title}'")
    return None

def extract_pdf_url_from_xml(xml_content):
    """Extract PDF URL from XML content if present."""
    pdf_url_match = re.search(r'https?://\S+\.pdf', xml_content, re.IGNORECASE)
    if pdf_url_match:
        return pdf_url_match.group(0)
    return None

def download_pdf_from_url(url, output_path):
    """Download PDF from URL using requests with headers from browser inspection."""
    logger.debug(f"Entering download_pdf_from_url with URL: {repr(url)}, output_path: {repr(output_path)}")
    
    headers = {
        'Upgrade-Insecure-Requests': '1',
        'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/129.0.0.0 Safari/537.36',
        'sec-ch-ua': '"Google Chrome";v="129", "Not=A?Brand";v="8", "Chromium";v="129"',
        'sec-ch-ua-mobile': '?0',
        'sec-ch-ua-platform': '"Linux"'
    }
    
    try:
        # Add a short delay to avoid overwhelming the server
        time.sleep(1)
        
        # Stream the response to handle large files
        with requests.get(url, headers=headers, stream=True, allow_redirects=True) as response:
            response.raise_for_status()
            
            # Check if the content type is PDF
            content_type = response.headers.get('Content-Type', '').lower()
            if 'application/pdf' not in content_type:
                logger.warning(f"The URL {url} does not point to a PDF (Content-Type: {content_type})")
                return False, f"The URL does not point to a PDF (Content-Type: {content_type})"
            
            # Save the PDF content to the output path
            with open(output_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
        
        # Verify the downloaded file
        if os.path.exists(output_path) and is_valid_pdf(output_path):
            logger.info(f"PDF successfully downloaded from {url} to {output_path}")
            return True, f"PDF successfully downloaded from {url}"
        else:
            logger.warning(f"Downloaded file from {url} is not a valid PDF")
            return False, f"Downloaded file from {url} is not a valid PDF"
    
    except requests.RequestException as e:
        logger.error(f"Error downloading PDF from {url}: {str(e)}")
        return False, f"Error downloading PDF: {str(e)}"
    
    finally:
        logger.debug("Exiting download_pdf_from_url")

def extract_pdf_url_from_xml(xml_content):
    """Extract PDF URL from XML content if present."""
    # Check for full URL
    pdf_url_match = re.search(r'<xocs:ucs-locator>(.*?\.pdf)</xocs:ucs-locator>', xml_content)
    if pdf_url_match:
        return pdf_url_match.group(1)
    
    # Check for relative URL
    relative_url_match = re.search(r'/articles/.*?\.pdf', xml_content)
    if relative_url_match:
        return relative_url_match.group(0)
    
    return None

def is_xml_only_pdf_link(xml_content):
    """Check if XML content only contains a PDF link."""
    xml_content = xml_content.strip()
    return xml_content.startswith('/articles/') and xml_content.endswith('.pdf')

def download_pdf(url):
    try:
        response = requests.get(url)
        response.raise_for_status()
        return response.content
    except Exception as e:
        logger.error(f"Error downloading PDF from {url}: {str(e)}")
        return None

def extract_redirect_url(html_content):
    """Extract redirect URL from HTML content."""
    soup = BeautifulSoup(html_content, 'html.parser')
    redirect_input = soup.find('input', {'id': 'redirectURL'})
    if redirect_input and 'value' in redirect_input.attrs:
        return unquote(redirect_input['value'])
    return None

def construct_full_url(url, source):
    """Construct a full URL from a potentially relative URL."""

    print(f"Constructing full URL for {url} from {source}")
    parsed_url = urlparse(url)
    if parsed_url.scheme:
        return url  # Already a full URL
    
    # Define base URLs for known sources
    base_urls = {
        "Elsevier API": "https://api.elsevier.com",
        "Springer API": "https://static-content.springer.com",
        "Wiley API": "https://onlinelibrary.wiley.com",
        "arXiv API": "https://arxiv.org",
        "Unpaywall API": "https://unpaywall.org",
        "DOI resolution": "https://doi.org"
    }
    
    # Special case for Nature articles
    if url.startswith('/article/') or url.startswith('/articles/'):
        return urljoin("https://www.nature.com", url)
    
    base_url = base_urls.get(source, "https://")
    return urljoin(base_url, url)