import os
from bs4 import BeautifulSoup
import re
from Levenshtein import ratio

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)



MAX_TITLE_LENGTH = 100


def clean_title(title):
    # Remove HTML tags
    title = BeautifulSoup(title, "html.parser").get_text()
    # Remove non-letter characters and convert to lower case
    title = re.sub(r'[^a-zA-Z]', '', title).lower()
    return title

# Function to truncate filenames
def truncate_filename(filename, max_length):
    return filename if len(filename) <= max_length else filename[:max_length]
    
def standardize_title(title):
    """Converts the title to lowercase, replaces spaces with hyphens, and truncates if necessary."""
    standardized_title = title.replace(" ", "-").lower()
    truncated_title = truncate_filename(standardized_title, MAX_TITLE_LENGTH)
    return truncated_title

def clean_and_lowercase(text):
    return ' '.join(clean_title(text).lower().split())

def titles_are_similar(title1, title2, threshold=0.8):
    """
    Compare two titles using Levenshtein distance.
    Returns True if the titles are similar enough, False otherwise.
    """
    return ratio(clean_and_lowercase(title1), clean_and_lowercase(title2)) >= threshold


def sanitize_filename(filename):
    """Remove or replace characters that are unsafe for filenames."""
    logger.debug(f"Sanitizing filename: {repr(filename)}")
    # Remove null characters
    filename = filename.replace('\0', '')
    # Replace other unsafe characters with underscore
    sanitized = re.sub(r'[<>:"/\\|?*]', '_', filename)
    logger.debug(f"Sanitized filename: {repr(sanitized)}")
    return sanitized

def generate_output_path(doi, file_type, base_output_dir):
    """Generate a safe output path for a given DOI and file type."""
    logger.debug(f"Generating output path for DOI: {repr(doi)}, file_type: {repr(file_type)}, base_output_dir: {repr(base_output_dir)}")
    safe_doi = sanitize_filename(doi)
    path = os.path.join(base_output_dir, file_type, f"{safe_doi}.{file_type}")
    logger.debug(f"Generated output path: {repr(path)}")
    return path
