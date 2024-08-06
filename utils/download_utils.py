import os
from habanero import Crossref
import pandas as pd
import subprocess
import glob
import tempfile
import shutil
import requests
from utils.pdf_utils import is_valid_pdf, clean_title


scihub_mirrors = [
    'http://sci-hub.mobi', 'https://sci-hub.se', 'http://sci-hub.se', 
    'http://sci-hub.ru', 'https://sci-hub.mobi', 'https://sci-hub.ru', 
    'http://sci-hub.st', 'https://sci-hub.st', 'https://sci-hub.ee'
]

class NotFoundError(Exception):
    pass

def find_doi_with_habanero(title):
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
                    return doi, url
    
    print("No DOI found with habanero.")
    return None, None

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
            return doi, url

    print(f"No results or DOI found for the title: '{title}'")
    return None, None



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


def download_pdf_with_pypaperbot(doi, download_path):
    temp_dir = tempfile.mkdtemp()
    try:
        result = subprocess.run([
            "python", "-m", "PyPaperBot",
            "--doi", doi,
            "--dwn-dir", temp_dir
        ], capture_output=True, text=True)
        print(result.stdout)
        return handle_downloaded_file(temp_dir, download_path)
    except Exception as e:
        print(f"Failed to download PDF for DOI {doi} using pypaperbot: {e}")
        return False, str(e)
    finally:
        shutil.rmtree(temp_dir)  # Clean up temporary directory


def download_pdf_with_scidownl(doi, download_path, title=None):
    temp_dir = tempfile.mkdtemp()
    try:
        if doi:
            cmd = ["scidownl", "download", "--doi", doi, "--out", temp_dir, "--proxy", "http=socks5://localhost:9050"]
        else:
            cmd = ["scidownl", "download", "--title", title, "--out", temp_dir, "--proxy", "http=socks5://localhost:9050"]
        
        os.makedirs(temp_dir, exist_ok=True)
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)  # Set timeout to 1 minute (60 seconds)
        except subprocess.TimeoutExpired:
            print(f"scidownl timed out for DOI {doi} or title {title}")
            return False, "scidownl timed out"
        
        print(result.stdout)
        return handle_downloaded_file(temp_dir, download_path)
    except Exception as e:
        print(f"Failed to download PDF for DOI {doi} using scidownl: {e}")
        return False, str(e)
    finally:
        shutil.rmtree(temp_dir)  # Clean up temporary directory

def download_pdf_with_doi2pdf(doi=None, title=None, download_path=None):
    try:
        cmd = ["doi2pdf", "--doi", doi, "--output", download_path] if doi else ["doi2pdf", "--name", title, "--output", download_path]
        result = subprocess.run(cmd, capture_output=True, text=True)
        print(result.stdout)
        
        if is_valid_pdf(download_path):
            print(f"Downloaded paper saved to {download_path}")
            return True, ""
        else:
            print("Downloaded PDF is invalid or corrupted.")
            os.remove(download_path)  # Remove the invalid file
            return False, "Downloaded PDF is invalid or corrupted."
    except Exception as e:
        entity = f"DOI {doi}" if doi else f"title {title}"
        print(f"Failed to download PDF for {entity}: {e}")
        return False, str(e)


def find_and_download_paper(title, download_path):
    try:
        doi, url = find_doi_with_habanero(title)
        if not doi:
            doi, url = find_doi_with_openalex(title)
        if doi:
            print(f"Found DOI URL: {url} for the title: '{title}'")
            success, reason = download_pdf_with_doi2pdf(doi=doi, download_path=download_path)
            if not success:
                print(f"doi2pdf failed: {reason}. Attempting to download using PyPaperBot...")
                success, reason = download_pdf_with_pypaperbot(doi, download_path)
                if not success:
                    print(f"PyPaperBot failed: {reason}. Attempting to download using scidownl...")
                    success, reason = download_pdf_with_scidownl(doi, download_path)
                    if not success:
                        return False, reason
        else:
            print(f"DOI not found for the title: '{title}'. Attempting to download using doi2pdf by title...")
            success, reason = download_pdf_with_doi2pdf(title=title, download_path=download_path)
            if not success:
                print(f"doi2pdf failed: {reason}. Attempting to download using scidownl by title...")
                success, reason = download_pdf_with_scidownl(None, download_path, title=title)
                if not success:
                    return False, reason
    finally:
        pass
    return True, ""
