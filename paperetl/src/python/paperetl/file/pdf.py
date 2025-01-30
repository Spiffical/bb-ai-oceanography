"""
PDF processing module
"""

from io import StringIO
import logging

import requests

from .tei import TEI

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PDF:
    """
    Methods to transform medical/scientific PDFs into article objects.
    """

    @staticmethod
    def parse(stream, source):
        """
        Parses a medical/scientific PDF datastream and returns a processed article.

        Args:
            stream: handle to input data stream
            source: text string describing stream source, can be None

        Returns:
            Article
        """
        logger.info(f"Starting PDF processing for source: {source}")

        try:
            # Attempt to convert PDF to TEI XML
            xml = PDF.convert(stream)
            if not xml:
                logger.error("Failed to convert PDF to TEI XML")
                return None

            # Parse and return object
            article = TEI.parse(xml, source)
            if article:
                logger.info("Successfully processed PDF and extracted metadata")
                # Access reference through metadata tuple - it's the 9th element (index 8)
                if article.metadata[9]:
                    logger.info(f"Found DOI: {article.metadata[9]}")
                else:
                    logger.warning("No DOI found in processed PDF")
            else:
                logger.error("Failed to parse TEI XML")
            
            return article

        except Exception as e:
            logger.error(f"Error processing PDF: {str(e)}")
            return None

    @staticmethod
    def convert(stream):
        """
        Converts a medical/scientific article PDF into TEI XML via a GROBID Web Service API call.

        Args:
            stream: handle to input data stream

        Returns:
            TEI XML stream
        """
        try:
            logger.info("Attempting to convert PDF using GROBID")
            
            # Call GROBID API
            response = requests.post(
                "http://localhost:8070/api/processFulltextDocument", 
                files={"input": stream},
                timeout=300  # 5 minute timeout
            )

            # Validate request was successful
            if not response.ok:
                logger.error(f"GROBID processing failed - Status code: {response.status_code}")
                logger.error(f"GROBID error message: {response.text}")
                return None

            logger.info("Successfully converted PDF using GROBID")
            # Wrap as StringIO
            return StringIO(response.text)

        except requests.exceptions.Timeout:
            logger.error("GROBID processing timed out after 5 minutes")
            return None
        except requests.exceptions.ConnectionError:
            logger.error("Failed to connect to GROBID service - ensure it is running")
            return None
        except Exception as e:
            logger.error(f"Unexpected error during GROBID conversion: {str(e)}")
            return None
