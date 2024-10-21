"""
TEI (Text Encoding Initiative) XML processing module
"""

import datetime
import hashlib

from bs4 import BeautifulSoup
from dateutil import parser
from nltk.tokenize import sent_tokenize

from ..schema.article import Article
from ..table import Table
from ..text import Text


class TEI:
    """
    Methods to transform Elsevier XML into article objects.
    """

    @staticmethod
    def parse(stream, source):
        """
        Parses a TEI XML datastream and returns a processed article.

        Args:
            stream: handle to input data stream
            source: text string describing stream source, can be None

        Returns:
            Article
        """

        soup = BeautifulSoup(stream, "lxml")

        # Save soup to file and overwrite existing file
        with open("soup.xml", "w") as f:
            f.write(soup.prettify())

        # Print the structure of the XML
        # print("XML Structure:")
        # print_xml_structure(soup)

        # Extract title
        title_element = soup.find('dc:title')
        title = title_element.text.strip() if title_element else "No title found"

        # Extract article metadata
        (published, publication, authors, affiliations, affiliation, reference) = TEI.metadata(soup)

        # Extract abstract
        # abstract = TEI.abstract(soup)

        # Validate parsed data
        if not title and not reference:
            print("Failed to parse content - no unique identifier found")
            return None
        
        # print("Title:", title)
        # print("Abstract:", abstract)
        # print("Published:", published)
        # print("Publication:", publication)
        # print("Authors:", authors)
        # print("Affiliations:", affiliations)
        # print("Affiliation:", affiliation)
        # print("Reference:", reference)

        # Parse text sections
        sections = TEI.text(soup, title)

        # Derive uid
        uid = hashlib.sha1(
            title.encode("utf-8") if title else reference.encode("utf-8")
        ).hexdigest()

        # Default title to source if empty
        title = title if title else source

        # Article metadata - id, source, published, publication, authors, affiliations, affiliation, title,
        #                    tags, reference, entry date
        metadata = (
            uid,
            source,
            published,
            publication,
            authors,
            affiliations,
            affiliation,
            title,
            "PDF",
            reference,
            parser.parse(datetime.datetime.now().strftime("%Y-%m-%d")),
        )

        return Article(metadata, sections)

    @staticmethod
    def date(published):
        """
        Attempts to parse a publication date, if available. Otherwise, None is returned.

        Args:
            published: published object

        Returns:
            publication date if available/found, None otherwise
        """

        if published:
            try:
                return parser.parse(published)
            except:
                return None
        return None

    @staticmethod
    def authors(soup):
        """
        Parses authors and associated affiliations from the article.

        Args:
            elements: authors elements

        Returns:
            (semicolon separated list of authors, semicolon separated list of affiliations, primary affiliation)
        """

        authors = []
        affiliations = []
        affiliation = None

        author_group = soup.find('ce:author-group')
        if author_group:
            for author in author_group.find_all('ce:author'):
                given_name = author.find('ce:given-name')
                surname = author.find('ce:surname')
                if given_name and surname:
                    authors.append(f"{surname.text.strip()}, {given_name.text.strip()}")

            for aff in author_group.find_all('ce:affiliation'):
                aff_text = aff.find('ce:textfn')
                if aff_text:
                    affiliations.append(aff_text.text.strip())

            # Set the first affiliation as the primary one
            if affiliations:
                affiliation = affiliations[0]

        return ("; ".join(authors), "; ".join(affiliations), affiliation)

    @staticmethod
    def metadata(soup):
        """
        Extracts article metadata.

        Args:
            soup: bs4 handle

        Returns:
            (published, publication, authors, affiliations, affiliation, reference)
        """
        # Extract publication date
        date_element = soup.find('prism:coverdate')
        # print("Date element:", date_element)
        # if date_element:
            # print("Date text:", date_element.text)
        published = TEI.date(date_element.text if date_element else None)
        # print("Parsed published date:", published)

        # Extract publication name
        publication_element = soup.find('prism:publicationname')
        # print("Publication element:", publication_element)
        # if publication_element:
        #     print("Publication text:", publication_element.text)
        publication = publication_element.text.strip() if publication_element else None

        # Extract authors and affiliations
        authors, affiliations, affiliation = TEI.authors(soup)
        # print("Authors:", authors)
        # print("Affiliations:", affiliations)
        # print("Primary affiliation:", affiliation)

        # Extract DOI (reference)
        doi_element = soup.find('prism:doi')
        # print("DOI element:", doi_element)
        # if doi_element:
            # print("DOI text:", doi_element.text)
        reference = doi_element.text.strip() if doi_element else None

        return (published, publication, authors, affiliations, affiliation, reference)

    @staticmethod
    def abstract(soup):
        """
        Builds a list of title and abstract sections.

        Args:
            soup: bs4 handle
            title: article title

        Returns:
            list of sections
        """

        abstract_element = soup.find('dc:description')
        if abstract_element:
            abstract = abstract_element.text.strip()
            # Transform and clean text
            abstract = Text.transform(abstract)
            abstract = abstract.replace("\n", " ")
            return [("ABSTRACT", x) for x in sent_tokenize(abstract)]
        return []

    @staticmethod
    def text(soup, title):
        """
        Builds a list of text sections.

        Args:
            soup: bs4 handle
            title: article title

        Returns:
            list of sections
        """

        # Initialize with title and abstract text
        sections = [("TITLE", title)] + TEI.abstract(soup)

        article = soup.find('article')
        if article:
            # Create a new list to store the extracted sections and sentences
            extracted_sections = []

            for sections_element in article.find_all('ce:sections'):
                for section in sections_element.find_all('ce:section'):
                    # Find the section title element
                    section_title_element = section.find('ce:section-title')
                    if section_title_element:
                        # Extract the section title text
                        section_title = section_title_element.get_text().strip()
                        section_title = section_title.upper()
                    else:
                        section_title = None

                    # Find all 'para' elements within the section
                    paragraphs = section.find_all('ce:para')
                    text = " ".join([p.get_text() for p in paragraphs])
                    text = text.replace("\n", " ")

                    # Transform and clean text
                    text = Text.transform(text)

                    # Split text into sentences, transform text and add to extracted_sections
                    extracted_sections.extend([(section_title, x) for x in sent_tokenize(text)])

            # Extend the original sections list with the extracted sections and sentences
            sections.extend(extracted_sections)

        if article:         
            # New code for handling tables within figures
            for para in article.find_all('ce:para'):
                for figure in para.find_all('figure'):
                    table = figure.find('ce:table')
                    if table:
                        caption = figure.find('ce:caption')
                        caption_text = caption.get_text().strip() if caption else "No caption"
                        table_name = f"TABLE: {caption_text}"
                        try:
                            extracted_data = Table.extract(table)
                            if extracted_data:
                                sections.extend([(table_name.upper(), x) for x in extracted_data])
                            else:
                                print(f"No data extracted from table: {table_name}")
                        except Exception as e:
                            print(f"Error extracting table {table_name}: {str(e)}")

        if article:
            # Handle tables that are not with in figures
            for table in article.find_all('ce:table'):
                table_name = table.find('ce:label').text.strip() if table.find('ce:label') else "Unnamed Table"
                try:
                    extracted_data = Table.extract(table)
                    if extracted_data:
                        sections.extend([(table_name.upper(), x) for x in extracted_data])
                    else:
                        print(f"No data extracted from table: {table_name}")
                except Exception as e:
                    print(f"Error extracting table {table_name}: {str(e)}")

        return sections

def print_xml_structure(element, level=0):
    """
    Recursively prints the structure of the XML.

    Args:
        element: BeautifulSoup element
        level: indentation level (default: 0)
    """
    indent = "  " * level
    if element.name:
        print(f"{indent}{element.name}")
    else:
        print(f"{indent}Text: {element.strip()}")

    for child in element.children:
        if child.name:
            print_xml_structure(child, level + 1)
