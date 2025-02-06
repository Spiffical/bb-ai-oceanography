import os
import requests
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv(override=True)

def get_full_text_springer(doi):
    """Get full text from Springer Nature using the Full Text TDM API."""
    api_key = os.getenv('SPRINGER_API_KEY')
    if not api_key:
        return False, "No API key found"
    
    base_url = "https://spdi.public.springernature.app/xmldata/jats"
    params = {
        'api_key': api_key,
        'q': f'doi:{doi}'
    }
    
    try:
        response = requests.get(base_url, params=params)
        print(f"\nRequest URL: {response.url}")  # Print the actual URL being called
        print(f"Status Code: {response.status_code}")
        response.raise_for_status()
        
        if response.status_code == 200:
            content = response.text
            
            # Check if we got meaningful content
            if '<response><total>0</total>' in content or '<recordsDisplayed>0</recordsDisplayed>' in content:
                return False, "No content found for this DOI"
            
            # Print first few XML tags to help debug
            print("\nFirst few XML tags found:")
            import re
            tags = re.findall(r'<(\w+)[>\s]', content[:1000])
            print(f"Tags: {tags}")
            
            # Check for various possible article tags
            article_tags = ['article', 'paper', 'book-part', 'chapter', 'document']
            found_tags = [tag for tag in article_tags if f'<{tag}' in content.lower()]
            
            if not found_tags:
                return False, f"Response does not contain any expected content tags. Looking for: {article_tags}"
            
            print(f"\nFound content tags: {found_tags}")
            return True, content
        else:
            return False, f"Error: {response.status_code}"
            
    except requests.exceptions.RequestException as e:
        return False, str(e)

def test_get_full_text_springer():
    # Let's try a different paper - this one is definitely in Springer's database
    test_doi = "10.1007/s00158-019-02374-9"  # "Topology optimization considering connectivity..."
    
    success, result = get_full_text_springer(test_doi)
    
    print(f"\nSuccess: {success}")
    print(f"Result length: {len(result) if isinstance(result, str) else 'N/A'}")
    
    if not success:
        print(f"Error message: {result}")
        assert False, f"Failed to retrieve content: {result}"
    
    print("\nFirst 1000 characters of result:")
    print(result[:1000])
    
    # Save the XML content to a file
    if success:
        # Create a data directory if it doesn't exist
        os.makedirs('data', exist_ok=True)
        # Save the file using the DOI as the filename (replacing / with _)
        filename = f"data/springer_{test_doi.replace('/', '_')}.xml"
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(result)
        print(f"\nSaved XML content to: {filename}")
    
    assert success is True, "Failed to retrieve content from Springer"
    assert result is not None, "No content returned from Springer"
    assert len(result) > 1000, "Response seems too short to be a full paper"
    
    # Check for any of the possible article tags
    article_tags = ['article', 'paper', 'book-part', 'chapter', 'document']
    found_tags = [tag for tag in article_tags if f'<{tag}' in result.lower()]
    assert found_tags, f"Response does not contain any expected content tags. Looking for: {article_tags}" 