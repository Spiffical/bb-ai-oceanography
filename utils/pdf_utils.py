import os
from bs4 import BeautifulSoup
import re

def is_valid_pdf(filepath):
    if os.path.getsize(filepath) == 0:
        return False
    try:
        with open(filepath, 'rb') as f:
            header = f.read(4)
            return header == b'%PDF'
    except Exception as e:
        print(f"Error checking PDF validity: {e}")
        return False
    
def clean_title(title):
    # Remove HTML tags
    title = BeautifulSoup(title, "html.parser").get_text()
    # Remove non-letter characters and convert to lower case
    title = re.sub(r'[^a-zA-Z]', '', title).lower()
    return title
    
def standardize_title(title):
    """Converts title to lowercase and replaces spaces with hyphens."""
    return title.replace(" ", "-").lower()