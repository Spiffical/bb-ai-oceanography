import requests


def verify_and_get_metadata(doi):
    base_url = "https://api.crossref.org/works/"
    headers = {
        "User-Agent": "YourApp/1.0 (mailto:your@email.com)"
    }

    def try_doi(doi):
        try:
            response = requests.get(f"{base_url}{doi}", headers=headers)
            response.raise_for_status()
            data = response.json()

            if 'message' in data:
                message = data['message']
                
                # Extract authors and affiliations
                authors = []
                all_affiliations = []
                for author in message.get('author', []):
                    family = author.get('family', '')
                    given = author.get('given', '')
                    if family and given:
                        authors.append(f"{family}, {given}")
                    elif family:
                        authors.append(family)
                    
                    # Extract affiliations for this author
                    for affiliation in author.get('affiliation', []):
                        if 'name' in affiliation:
                            all_affiliations.append(affiliation['name'])
                
                # Remove duplicates from affiliations while preserving order
                affiliations = list(dict.fromkeys(all_affiliations))
                
                # Get primary affiliation (last one in the list)
                primary_affiliation = affiliations[-1] if affiliations else None
                
                # Extract publication date
                published_date = None
                date_fields = ['published-online', 'published-print', 'created']
                for field in date_fields:
                    date_info = message.get(field)
                    if date_info and isinstance(date_info, dict) and 'date-parts' in date_info:
                        date_parts = date_info['date-parts'][0]
                        published_date = '-'.join(map(str, date_parts))
                        break

                # Extract publication information
                publication_title = message.get('container-title', [])
                publication_title = publication_title[0] if publication_title else None

                # Extract domain information
                content_domain = message.get('content-domain', {})
                domain = content_domain.get('domain', [])
                domain = domain[0] if domain else None

                return {
                    'is_valid': True,
                    'authors': '; '.join(authors),
                    'affiliations': '; '.join(affiliations),
                    'primary_affiliation': primary_affiliation,
                    'published_date': published_date,
                    'publication_title': publication_title,
                    'domain': domain,
                }
            else:
                return None
        except requests.exceptions.RequestException:
            return None

    # Try with original DOI
    result = try_doi(doi)
    if result:
        return result

    # If original fails, try replacing last '/' with '_'
    if '/' in doi:
        modified_doi = '_'.join(doi.rsplit('/', 1))
        result = try_doi(modified_doi)
        if result:
            return result

    # If both attempts fail, return invalid result
    return {'is_valid': False}