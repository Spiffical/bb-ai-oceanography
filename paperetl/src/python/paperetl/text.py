"""
Text module
"""

import re

# Compiled pattern for cleaning text
# pylint: disable=W0603
PATTERN = None


def getPattern():
    """
    Gets or builds a pre-compiled regex for cleaning text.
    Only removes problematic patterns while preserving actual content.

    Returns:
        compiled regex
    """

    global PATTERN

    if not PATTERN:
        # List of patterns
        patterns = []

        # Remove emails
        patterns.append(r"\w+@\w+(\.[a-z]{2,})+")

        # Remove urls
        patterns.append(r"http(s)?\:\/\/\S+")

        # Remove citation references (ex. [3] [4] [5])
        patterns.append(r"(\[\d+\]\,?\s?){3,}(\.|\,)?")

        # Remove citation references (ex. [3, 4, 5])
        patterns.append(r"\[[\d\,\s]+\]")

        # Remove citation references (ex. (1) (2) (3))
        patterns.append(r"(\(\d+\)\s){3,}")

        # Remove excessive whitespace
        patterns.append(r"\s{2,}")

        # Remove excessive periods
        patterns.append(r"\.{2,}")

        PATTERN = re.compile("|".join([f"({p})" for p in patterns]))

    return PATTERN


class Text:
    """
    Methods for formatting and cleaning text.
    """

    @staticmethod
    def transform(text):
        """
        Transforms and cleans text to help improve text indexing accuracy.
        Preserves actual content while removing only problematic patterns.

        Args:
            text: input text line

        Returns:
            transformed text
        """

        if not text:
            return text

        # Clean/transform text
        text = getPattern().sub(" ", text)

        # Remove leading/trailing whitespace
        text = text.strip()

        return text
    
    @staticmethod
    def paragraph_tokenize(text):
        """
        Splits text into paragraphs.

        Args:
            text: input text

        Returns:
            list of paragraphs
        """
        if not text:
            return []
            
        return [p.strip() for p in text.split("\n\n") if p.strip()]
