"""
Table module
"""

import re

from lxml import etree
from bs4 import BeautifulSoup


class Table:
    """
    Parses text content from HTML tables.
    """

    @staticmethod
    def parse(text):
        """
        Parses content from a HTML table string. Builds a list of header-value pairs for each row.

        Args:
            text: HTML table string

        Returns:
            list of header-value pairs for each row
        """

        # Parse HTML content using lxml
        # pylint: disable=c-extension-no-member
        table = etree.HTML(text).find("body/table")

        return Table.extract(table)

    @staticmethod
    def extract(table):
        """
        Parses content from a HTML table element. Builds a list of header-value pairs for each row.

        Args:
            table: HTML table element or BeautifulSoup element

        Returns:
            list of header-value pairs for each row
        """
        # Table rows
        output = []

        # Handle BeautifulSoup element (from XML parsing)
        if isinstance(table, BeautifulSoup) or str(type(table)).find("bs4") != -1:
            # Check if this is an Elsevier table
            if table.name == "ce:table":
                # Get headers from thead/tr/entry elements
                thead = table.find("thead")
                if thead:
                    headers = [entry.get_text(strip=True) for entry in thead.find_all("entry")]
                else:
                    # If no thead, use first row as headers
                    first_row = table.find("row")
                    headers = [entry.get_text(strip=True) for entry in first_row.find_all("entry")] if first_row else []

                # Process remaining rows
                for row in table.find_all("row"):
                    values = []
                    for i, entry in enumerate(row.find_all("entry")):
                        header = headers[i] if i < len(headers) else ""
                        value = entry.get_text(strip=True)
                        values.append(f"{header} {value}")
                    
                    # Create single row string
                    row_text = " ".join(values)
                    row_text = re.sub(r"[\n\xa0\t]|\s{2,}", " ", row_text).strip()
                    if row_text:
                        output.append(row_text)

            # Standard JATS table
            else:
                rows = table.find_all("tr")
                if rows:
                    # Get headers from first row
                    headers = [th.get_text(strip=True) for th in rows[0].find_all(["th", "td"])]
                    
                    # Process remaining rows
                    for row in rows[1:]:
                        values = []
                        for i, cell in enumerate(row.find_all("td")):
                            header = headers[i] if i < len(headers) else ""
                            value = cell.get_text(strip=True)
                            values.append(f"{header} {value}")
                        
                        # Create single row string
                        row_text = " ".join(values)
                        row_text = re.sub(r"[\n\xa0\t]|\s{2,}", " ", row_text).strip()
                        if row_text:
                            output.append(row_text)

        # Handle lxml element (from HTML parsing)
        elif len(table) > 0:
            rows = iter(table)
            headers = [col.text for col in next(rows)]

            for row in rows:
                # Build concatenated header value string
                values = [
                    f"{headers[x] if x < len(headers) else ''} {column.text}"
                    for x, column in enumerate(row)
                ]

                # Create single row string
                value = " ".join(values)

                # Remove whitespace
                value = re.sub(r"[\n\xa0\t]|\s{2,}", " ", value).strip()
                if value:
                    output.append(value)

        return output
