
"""
Table module
"""

from bs4 import BeautifulSoup

class Table:
    """
    Parses text content from XML tables.
    """

    @staticmethod
    def extract(table):
        """
        Parses content from an XML table element. Builds a list of header-value pairs for each row.

        Args:
            table: XML table element

        Returns:
            list of header-value pairs for each row
        """
        output = []

        # Parse the table using BeautifulSoup
        soup = BeautifulSoup(str(table), 'xml')

        # Find the tgroup element
        tgroup = soup.find('tgroup')
        if not tgroup:
            return output

        # Extract headers
        thead = tgroup.find('thead')
        if thead:
            headers = [th.text.strip() for th in thead.find_all('entry')]
        else:
            # If there's no thead, use column names from colspec
            headers = [col.get('colname', f'Column {i+1}') for i, col in enumerate(tgroup.find_all('colspec'))]

        # Extract rows
        tbody = tgroup.find('tbody')
        if not tbody:
            return output

        rows = tbody.find_all('row')

        for row in rows:
            values = [td.text.strip() for td in row.find_all('entry')]
            
            # Combine headers and values, handling cases where they might not match in length
            row_data = []
            for i, value in enumerate(values):
                header = headers[i] if i < len(headers) else f"Column {i+1}"
                row_data.append(f"{header}: {value}")
            
            # Join the row data into a single string
            row_string = " | ".join(row_data).strip()
            
            if row_string:
                output.append(row_string)

        return output
