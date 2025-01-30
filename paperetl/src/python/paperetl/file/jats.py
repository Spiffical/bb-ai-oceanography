"""
JATS (Journal Article Tag Suite) XML processing module
"""

import datetime
import hashlib
import os

from bs4 import BeautifulSoup
from dateutil import parser
from lxml import etree

from ..schema.article import Article
from ..table import Table
from ..text import Text
from .metadata import verify_and_get_metadata


class JATS:
    """
    Methods to transform JATS XML into article objects.
    """

    # Define common namespaces
    NAMESPACES = {
        'dc': "http://purl.org/dc/elements/1.1/",
        'ce': "http://www.elsevier.com/xml/common/dtd",
        'xocs': "http://www.elsevier.com/xml/xocs/dtd",
        'prism': "http://prismstandard.org/namespaces/basic/2.0/",
        'xlink': "http://www.w3.org/1999/xlink",
        'mml': "http://www.w3.org/1998/Math/MathML",
        'sb': "http://www.elsevier.com/xml/common/struct-bib/dtd",
        'ja': "http://www.elsevier.com/xml/ja/dtd"
    }

    @staticmethod
    def get_text_content(element):
        """
        Extracts text content from an XML element, preserving only meaningful spaces.
        
        Args:
            element: XML element to extract text from
            
        Returns:
            Cleaned text content with proper spacing
        """
        # Get all text content, including nested elements
        text_parts = []
        for text in element.xpath('.//text()'):
            # Skip if parent is a script or style element
            parent = text.getparent()
            if parent is not None and parent.tag in ['script', 'style']:
                continue
            # Clean and add non-empty text
            cleaned = text.strip()
            if cleaned:
                text_parts.append(cleaned)
        
        # Join with single spaces and clean up any remaining whitespace issues
        return ' '.join(text_parts).strip()

    @staticmethod
    def parse(stream, source):
        """
        Parses a JATS XML datastream and returns a processed article.

        Args:
            stream: handle to input data stream
            source: text string describing stream source, can be None

        Returns:
            Article
        """
        print("\nJATS Parser:")

        # Parse XML with lxml for better namespace handling
        try:
            # Read the content first since stream might be used multiple times
            content = stream.read()
            tree = etree.fromstring(content.encode('utf-8') if isinstance(content, str) else content)
        except Exception as e:
            print(f"\nError parsing XML: {str(e)}")
            return None

        # Initialize domain
        domain = None

        # Detect XML format by checking root element
        root = tree
        is_elsevier = root.tag.endswith('full-text-retrieval-response')

        # Extract title based on format
        title = None
        if is_elsevier:
            # Try Elsevier coredata first
            title_elements = tree.xpath('//dc:title/text()', namespaces=JATS.NAMESPACES)
            if title_elements:
                title = title_elements[0].strip()
            
            # If not found, try article title
            if not title:
                title_elements = tree.xpath('//ce:title/text()', namespaces=JATS.NAMESPACES)
                if title_elements:
                    title = title_elements[0].strip()
        else:
            # Standard JATS format
            title_elements = tree.xpath('//article-title/text()')
            if title_elements:
                title = title_elements[0].strip()

        if not title:
            print("\nWarning: No title found")

        # Extract metadata based on format
        if is_elsevier:
            metadata = JATS.extract_elsevier_metadata(tree)
        else:
            metadata = JATS.extract_jats_metadata(tree)

        (published, publication, authors, affiliations, primary_affiliation, reference) = metadata

        # Only print metadata if something is missing
        if not all([published, publication, authors, reference]):
            print("\nWarning: Some metadata missing:")
            if not published:
                print("- No publication date")
            if not publication:
                print("- No journal title")
            if not authors:
                print("- No authors")
            if not reference:
                print("- No DOI reference")

        # Extract DOI from filename if available and use it to get metadata
        if source:
            # Replace underscores with forward slashes to reconstruct the original DOI
            doi = os.path.splitext(os.path.basename(source))[0].replace('_', '/')
            
            result = verify_and_get_metadata(doi)
            if not result['is_valid']:
                print(f"\nWarning: Failed to get metadata for DOI: {doi}")
            else:
                # Update metadata with DOI results if available
                published = result['published_date'] or published
                authors = result['authors'] or authors
                affiliations = result['affiliations'] or affiliations
                primary_affiliation = result['primary_affiliation'] or primary_affiliation
                reference = f"https://doi.org/{doi}" if doi else reference
                publication = result['publication_title'] or publication
                domain = result['domain'] or domain

        # If we do not have a unique ID (like a title or reference), bail
        if not title and not reference:
            print("\nError: Failed to parse JATS content - no unique identifier found")
            return None

        # Parse text sections based on format
        if is_elsevier:
            sections = JATS.extract_elsevier_sections(tree, title)
        else:
            sections = JATS.extract_jats_sections(tree, title)

        # Only print warning if no content sections found
        if len(sections) <= 1:  # Only title section
            print("\nWarning: No content sections found")
            print("Article structure:")
            if is_elsevier:
                structure = tree.xpath('//article/*', namespaces=JATS.NAMESPACES)
            else:
                structure = tree.xpath('//article/*')
            for elem in structure[:5]:
                print(f"- <{elem.tag.split('}')[-1]}>")

        # Create a stable UID from the title (or fallback to reference)
        uid_input = title if title else reference
        uid = hashlib.sha1(uid_input.encode("utf-8")).hexdigest()

        # If we do not have an actual 'title' fallback
        if not title:
            title = source or "Unknown"

        # Build the final metadata tuple
        metadata = (
            uid,                    # id
            source,                 # source
            published,              # published
            publication,            # publication
            authors,               # authors
            affiliations,          # affiliations
            primary_affiliation,   # affiliation
            title,                 # title
            "PDF",                 # tags (hardcoded as in TEI example)
            reference,             # reference
            parser.parse(datetime.datetime.now().strftime("%Y-%m-%d")),  # entry
            domain                 # domain
        )

        return Article(metadata, sections)

    @staticmethod
    def extract_elsevier_metadata(tree):
        """
        Extracts metadata from Elsevier XML format.
        """
        print("\nExtracting Elsevier metadata:")
        
        # Published date
        published_date = None
        cover_date = tree.xpath('//prism:coverDate/text()', namespaces=JATS.NAMESPACES)
        if cover_date:
            try:
                published_date = parser.parse(cover_date[0])
                print(f"- Found publication date: {published_date}")
            except:
                print("- Failed to parse publication date")
        
        # Journal title
        journal_title = None
        pub_name = tree.xpath('//prism:publicationName/text()', namespaces=JATS.NAMESPACES)
        if pub_name:
            journal_title = pub_name[0]
            print(f"- Found journal title: {journal_title}")
        
        # Authors
        authors = []
        creator_tags = tree.xpath('//dc:creator/text()', namespaces=JATS.NAMESPACES)
        if creator_tags:
            print(f"- Found {len(creator_tags)} authors")
            authors = [author.strip() for author in creator_tags if author.strip()]
        
        # DOI reference
        reference = None
        doi = tree.xpath('//prism:doi/text()', namespaces=JATS.NAMESPACES)
        if doi:
            reference = f"https://doi.org/{doi[0]}"
            print(f"- Found DOI reference: {reference}")
        
        # For Elsevier XML, we don't have structured affiliations in coredata
        affiliations = []
        primary_affiliation = None
        
        return (
            published_date,
            journal_title,
            "; ".join(authors),
            "; ".join(affiliations),
            primary_affiliation,
            reference
        )

    @staticmethod
    def extract_jats_metadata(tree):
        """
        Extracts metadata from standard JATS XML format.
        """
        print("\nExtracting JATS metadata:")
        
        # Published date
        published_date = None
        pub_date = tree.xpath('//pub-date[@date-type="pub"]|//pub-date[@publication-format="electronic"]')
        if pub_date:
            date_parts = []
            year = pub_date[0].xpath('.//year/text()')
            month = pub_date[0].xpath('.//month/text()')
            day = pub_date[0].xpath('.//day/text()')
            
            if year:
                date_parts.append(year[0])
            if month:
                date_parts.append(month[0])
            if day:
                date_parts.append(day[0])
                
            if date_parts:
                try:
                    published_date = parser.parse('-'.join(date_parts))
                    print(f"- Found publication date: {published_date}")
                except:
                    print("- Failed to parse publication date")

        # Journal title
        journal_title = None
        journal_title_elem = tree.xpath('//journal-title/text()|//abbrev-journal-title/text()')
        if journal_title_elem:
            journal_title = journal_title_elem[0]
            print(f"- Found journal title: {journal_title}")
        
        # Authors and Affiliations
        authors = []
        affiliations = []
        primary_affiliation = None
        
        author_elements = tree.xpath('//contrib[@contrib-type="author"]')
        print(f"- Found {len(author_elements)} authors")
        
        for author in author_elements:
            name_parts = []
            surname = author.xpath('.//surname/text()')
            given_names = author.xpath('.//given-names/text()')
            
            if surname:
                name_parts.append(surname[0])
            if given_names:
                name_parts.append(given_names[0])
                
            if name_parts:
                authors.append(", ".join(name_parts))
            
            # Get affiliations
            aff_refs = author.xpath('.//xref[@ref-type="aff"]/@rid')
            for ref in aff_refs:
                aff = tree.xpath(f'//aff[@id="{ref}"]')
                if aff:
                    # Try structured elements first
                    aff_parts = []
                    institution = aff[0].xpath('.//institution[@content-type="org-name"]/text()')
                    if institution:
                        aff_parts.append(institution[0])
                    
                    division = aff[0].xpath('.//institution[@content-type="org-division"]/text()')
                    if division:
                        aff_parts.append(division[0])
                        
                    city = aff[0].xpath('.//addr-line[@content-type="city"]/text()')
                    if city:
                        aff_parts.append(city[0])
                        
                    country = aff[0].xpath('.//country/text()')
                    if country:
                        aff_parts.append(country[0])
                        
                    if aff_parts:
                        affiliations.append(", ".join(aff_parts))
                    else:
                        # Fallback to full text
                        aff_text = " ".join(aff[0].xpath('.//text()'))
                        if aff_text.strip():
                            affiliations.append(aff_text.strip())
        
        # Remove duplicates while preserving order
        affiliations = list(dict.fromkeys(affiliations))
        if affiliations:
            primary_affiliation = affiliations[0]
        
        # DOI reference
        reference = None
        doi = tree.xpath('//article-id[@pub-id-type="doi"]/text()')
        if doi:
            reference = f"https://doi.org/{doi[0]}"
            print(f"- Found DOI reference: {reference}")
        
        return (
            published_date,
            journal_title,
            "; ".join(authors),
            "; ".join(affiliations),
            primary_affiliation,
            reference
        )

    @staticmethod
    def extract_table_content(table, namespaces=None):
        """
        Helper method to extract content from a table element.
        Handles both Elsevier and JATS table formats.
        """
        table_content = []
        
        # Get table caption/label if available
        caption = []
        if namespaces:  # Elsevier
            label = table.xpath('.//ce:label/text()', namespaces=namespaces)
            caption_text = table.xpath('normalize-space(.//ce:caption)', namespaces=namespaces)
        else:  # JATS
            label = table.xpath('.//label/text()')
            caption_text = table.xpath('normalize-space(.//caption)')
            
        if label:
            caption.append(label[0].strip())
        if caption_text:
            caption.append(caption_text.strip())
            
        if caption:
            table_content.append(" ".join(caption))

        # Get table rows
        if namespaces:  # Elsevier
            rows = table.xpath('.//ce:row', namespaces=namespaces)
            if rows:
                # Get headers from first row
                headers = []
                first_row = rows[0].xpath('.//ce:entry', namespaces=namespaces)
                if first_row:
                    headers = [entry.xpath('normalize-space(.)') for entry in first_row]
                
                # Process remaining rows
                for row in rows[1:]:
                    entries = row.xpath('.//ce:entry', namespaces=namespaces)
                    if entries:
                        row_text = []
                        for i, entry in enumerate(entries):
                            # Get all text content including nested elements
                            value = entry.xpath('normalize-space(.)')
                            header = headers[i] if i < len(headers) else ""
                            if header and value:
                                row_text.append(f"{header}: {value}")
                            elif value:
                                row_text.append(value)
                        if row_text:
                            table_content.append(" | ".join(row_text))
        else:  # JATS
            rows = table.xpath('.//tr')
            if rows:
                # Get headers from first row
                headers = []
                first_row = rows[0].xpath('.//th|.//td')
                if first_row:
                    headers = [cell.xpath('normalize-space(.)') for cell in first_row]
                
                # Process remaining rows
                for row in rows[1:]:
                    cells = row.xpath('.//td')
                    if cells:
                        row_text = []
                        for i, cell in enumerate(cells):
                            # Get all text content including nested elements
                            value = cell.xpath('normalize-space(.)')
                            header = headers[i] if i < len(headers) else ""
                            if header and value:
                                row_text.append(f"{header}: {value}")
                            elif value:
                                row_text.append(value)
                        if row_text:
                            table_content.append(" | ".join(row_text))
                            
        return table_content

    @staticmethod
    def extract_elsevier_sections(tree, title):
        """
        Extract sections from Elsevier XML format.
        """
        sections = []
        print("\nExtracting sections:")

        # Add title section
        sections.append(("TITLE", title))
        print("- Added title section")
        
        # Abstract
        abstract = tree.xpath('//ce:abstract', namespaces=JATS.NAMESPACES)
        if abstract:
            print("- Found abstract")
            abstract_paras = abstract[0].xpath('.//ce:para|.//ce:simple-para', namespaces=JATS.NAMESPACES)
            if abstract_paras:
                print(f"  - Found {len(abstract_paras)} abstract paragraphs")
                for para in abstract_paras:
                    text = JATS.get_text_content(para)
                    if text:
                        sections.append(("ABSTRACT", Text.transform(text)))
            else:
                # Fallback to full abstract text
                text = JATS.get_text_content(abstract[0])
                if text:
                    sections.append(("ABSTRACT", Text.transform(text)))

        # Main sections - only look within the article body
        main_sections = tree.xpath('//ce:sections/ce:section|//ce:section[@role="section"]', namespaces=JATS.NAMESPACES)
        if main_sections:
            print(f"- Found {len(main_sections)} main sections")
            for section in main_sections:
                # Get section title
                title_elem = section.xpath('./ce:section-title/text()', namespaces=JATS.NAMESPACES)
                section_title = title_elem[0] if title_elem else "SECTION"
                print(f"  - Processing section: {section_title}")
                
                # Get paragraphs - only direct paragraph children of the section or its subsections
                paragraphs = section.xpath('.//ce:para[not(ancestor::ce:table)]|.//ce:simple-para[not(ancestor::ce:table)]', namespaces=JATS.NAMESPACES)
                if paragraphs:
                    print(f"    - Found {len(paragraphs)} paragraphs")
                    for para in paragraphs:
                        text = JATS.get_text_content(para)
                        if text:
                            sections.append((section_title.upper(), Text.transform(text)))
                
                # Get tables within this section
                section_tables = section.xpath('.//ce:table[not(ancestor::ce:bibliography)]', namespaces=JATS.NAMESPACES)
                if section_tables:
                    print(f"    - Found {len(section_tables)} tables")
                    for i, table in enumerate(section_tables):
                        table_content = JATS.extract_table_content(table, JATS.NAMESPACES)
                        for content in table_content:
                            sections.append((f"{section_title.upper()}_TABLE_{i+1}", Text.transform(content)))
        
        # Tables outside sections
        tables = tree.xpath('//ce:table[not(ancestor::ce:section) and not(ancestor::ce:bibliography)]', namespaces=JATS.NAMESPACES)
        if tables:
            print(f"- Found {len(tables)} tables outside sections")
            for i, table in enumerate(tables):
                table_content = JATS.extract_table_content(table, JATS.NAMESPACES)
                for content in table_content:
                    sections.append((f"TABLE_{i+1}", Text.transform(content)))
        
        print(f"- Total sections extracted: {len(sections)}")
        return sections

    @staticmethod
    def extract_jats_sections(tree, title):
        """
        Extract sections from standard JATS XML format.
        """
        sections = []
        print("\nExtracting sections:")

        # Add title section
        sections.append(("TITLE", title))
        print("- Added title section")
        
        # Abstract
        abstract = tree.xpath('//abstract')
        if abstract:
            print("- Found abstract")
            abstract_paras = abstract[0].xpath('.//p')
            if abstract_paras:
                print(f"  - Found {len(abstract_paras)} abstract paragraphs")
                for para in abstract_paras:
                    text = JATS.get_text_content(para)
                    if text:
                        sections.append(("ABSTRACT", Text.transform(text)))
            else:
                # Fallback to full abstract text
                text = JATS.get_text_content(abstract[0])
                if text:
                    sections.append(("ABSTRACT", Text.transform(text)))

        # Main sections
        main_sections = tree.xpath('//sec')
        if main_sections:
            print(f"- Found {len(main_sections)} main sections")
            for section in main_sections:
                # Get section title
                title_elem = section.xpath('./title/text()')
                section_title = title_elem[0] if title_elem else "SECTION"
                print(f"  - Processing section: {section_title}")
                
                # Get paragraphs - only direct paragraph children of the section
                paragraphs = section.xpath('.//p[not(ancestor::table)]')
                if paragraphs:
                    print(f"    - Found {len(paragraphs)} paragraphs")
                    for para in paragraphs:
                        text = JATS.get_text_content(para)
                        if text:
                            sections.append((section_title.upper(), Text.transform(text)))

        return sections 