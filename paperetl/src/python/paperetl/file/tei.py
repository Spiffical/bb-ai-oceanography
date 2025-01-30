"""
TEI (Text Encoding Initiative) XML processing module
"""

import datetime
import hashlib
import os
import logging

from bs4 import BeautifulSoup
from dateutil import parser

from ..schema.article import Article
from ..table import Table
from ..text import Text
from .metadata import verify_and_get_metadata

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TEI:
    """
    Methods to transform TEI (Text Encoding Initiative) XML into article objects.
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
        title = soup.title.text if soup.title else None
        domain = None 
        reference = None
        published, publication, authors, affiliations, affiliation = None, None, None, None, None

        # First get DOI from filename if available - this is our primary source of truth
        if source:
            # Replace underscores with forward slashes to reconstruct the original DOI
            filename_doi = os.path.splitext(os.path.basename(source))[0].replace('_', '/')
            logger.info(f"Using DOI from filename: {filename_doi}")
            reference = f"https://doi.org/{filename_doi}"
            
            # Try to get additional metadata from CrossRef, but don't fail if we can't
            try:
                result = verify_and_get_metadata(filename_doi)
                if result['is_valid']:
                    logger.info(f"Successfully retrieved additional metadata for DOI: {filename_doi}")
                    published = result['published_date']
                    authors = result['authors']
                    affiliations = result['affiliations']
                    affiliation = result['primary_affiliation']
                    publication = result['publication_title']
                    domain = result['domain']
                else:
                    logger.info(f"Could not retrieve additional metadata for DOI: {filename_doi}")
            except Exception as e:
                logger.info(f"Error retrieving additional metadata for DOI {filename_doi}: {str(e)}")

        # If we don't have metadata yet, try extracting from TEI XML
        if not authors or not affiliations:
            logger.info("Attempting to extract metadata from TEI XML")
            try:
                xml_metadata = TEI.metadata(soup)
                # Only use metadata we haven't already gotten from CrossRef
                published = published or xml_metadata[0]
                publication = publication or xml_metadata[1]
                authors = authors or xml_metadata[2]
                affiliations = affiliations or xml_metadata[3]
                affiliation = affiliation or xml_metadata[4]
            except Exception as e:
                logger.error(f"Error extracting metadata from TEI XML: {str(e)}")

        # Validate parsed data
        if not title and not reference:
            logger.error("Failed to parse content - no unique identifier found")
            return None

        # Parse text sections
        sections = TEI.text(soup, title)

        # Derive uid - use DOI if available, otherwise hash of title
        if reference:
            uid = reference.split('/')[-1]  # Use last part of DOI as uid
        else:
            uid = hashlib.sha1(title.encode("utf-8")).hexdigest()

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
            domain,
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

        # Parse publication date
        # pylint: disable=W0702
        try:
            published = (
                parser.parse(published["when"])
                if published and "when" in published.attrs
                else None
            )
        except:
            published = None

        return published

    @staticmethod
    def authors(source):
        """
        Parses authors and associated affiliations from the article.

        Args:
            elements: authors elements

        Returns:
            (semicolon separated list of authors, semicolon separated list of affiliations, primary affiliation)
        """

        authors = []
        affiliations = []

        for name in source.find_all("persname"):
            surname = name.find("surname")
            forename = name.find("forename")

            if surname and forename:
                authors.append(f"{surname.text}, {forename.text}")

        for affiliation in source.find_all("affiliation"):
            names = [name.text for name in affiliation.find_all("orgname")]
            affiliations.append((", ".join(names)))

        return (
            "; ".join(authors),
            "; ".join(dict.fromkeys(affiliations)),
            affiliations[-1] if affiliations else None,
        )

    @staticmethod
    def metadata(soup):
        """
        Extracts article metadata.

        Args:
            soup: bs4 handle

        Returns:
            (published, publication, authors, reference)
        """

        # Build reference link
        source = soup.find("sourcedesc")
        if source:
            published = source.find("monogr").find("date")
            publication = source.find("monogr").find("title")

            # Parse publication information
            published = TEI.date(published)
            publication = publication.text if publication else None
            authors, affiliations, affiliation = TEI.authors(source)

            struct = soup.find("biblstruct")
            reference = (
                "https://doi.org/" + struct.find("idno").text
                if struct and struct.find("idno")
                else None
            )
        else:
            published, publication, authors, affiliations, affiliation, reference = (
                None,
                None,
                None,
                None,
                None,
                None,
            )

        return (published, publication, authors, affiliations, affiliation, reference)

    @staticmethod
    def abstract(soup, title):
        """
        Builds a list of title and abstract sections.

        Args:
            soup: bs4 handle
            title: article title

        Returns:
            list of sections
        """

        sections = [("TITLE", title)]

        abstract = soup.find("abstract").text
        if abstract:
            # Transform and clean text
            abstract = Text.transform(abstract)
            abstract = abstract.replace("\n", " ")

            # sections.extend([("ABSTRACT", x) for x in sent_tokenize(abstract)])
            # sections.extend([("ABSTRACT", x) for x in Text.paragraph_tokenize(abstract)])
            sections.extend([("ABSTRACT", abstract)])
        return sections

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
        sections = TEI.abstract(soup, title)

        for section in soup.find("text").find_all("div", recursive=False):
            # Section name and text
            children = list(section.children)

            # Attempt to parse section header
            if children and not children[0].name:
                name = str(children[0]).upper()
                children = children[1:]
            else:
                name = None

            # Extract paragraphs
            paragraphs = section.find_all('p')
            for para in paragraphs:
                text = para.get_text(strip=True)
                
                # Transform and clean text
                text = Text.transform(text)

                # Add paragraph to sections
                # sections.extend([(name, x) for x in sent_tokenize(text)])
                sections.extend([(name, text)])

        # Extract text from tables
        for i, figure in enumerate(soup.find("text").find_all("figure")):
            name = figure.get("xml:id")
            name = name.upper() if name else f"FIGURE_{i}"

            table = figure.find("table")
            if table:
                table_text = Table.extract(table)
                sections.extend([(name, x) for x in table_text])

        return sections
