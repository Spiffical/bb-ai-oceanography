"""
Report module
"""

import regex as re
from datetime import datetime
from dateutil import parser
from tqdm import tqdm

from txtai.pipeline import Extractor, Labels, Similarity, Tokenizer

from ..index import Index
from ..query import Query

from .column import Column

from .summarizer import Summarizer

class Report:
    """
    Methods to build reports from a series of queries
    """

    def __init__(self, embeddings, db, options):
        """
        Creates a new report.

        Args:
            embeddings: embeddings index
            db: database connection
            options: report options
        """

        # Store references to embeddings index and open database cursor
        self.embeddings = embeddings
        self.cur = db.cursor()

        # Report options
        self.options = options

        # Column names
        self.names = []

        self.similarity = (
            Similarity(options["similarity"]) if "similarity" in options else None
        )
        self.labels = Labels(model=self.similarity) if self.similarity else None

        # Question-answering model
        # Determine if embeddings or a custom similarity model should be used to build question context
        self.extractor = Extractor(
            self.similarity if self.similarity else self.embeddings,
            options["qa"] if options["qa"] else "NeuML/bert-small-cord19qa",
            minscore=options.get("minscore"),
            mintokens=options.get("mintokens"),
            context=options.get("context", 512),
        )

        # Initialize the Summarizer
        llm_mode = options.get("llm_mode", "local")
        if llm_mode == "api":
            model_name = options.get("api", {}).get("model", "gpt-4o-mini")
            api_provider = options.get("api", {}).get("provider", "openai")
        else:
            local_opts = options.get("local", {})
            model_name = local_opts.get("model", "gemma2:9b")
            provider = local_opts.get("provider", "ollama")
            gpu_strategy = local_opts.get("gpu_strategy", "auto")
        self.summarizer = Summarizer(
            llm_name=model_name,
            mode=llm_mode,
            gpu_strategy=gpu_strategy if llm_mode == "local" else None,
            provider=api_provider if llm_mode == "api" else provider if llm_mode == "local" else None
        )

    def generate_summary(self, results, topn, query):
        """
        Generates a summary using the specified model based on the top results and the query.
        """
        top_results = results[:topn]
        context, citation_details = self._prepare_context(top_results)
        
        summary = self.summarizer.generate_summary(context, query)
        
        return summary, citation_details

    def _prepare_context(self, top_results):
        """
        Prepares the context for summarization and citation details from top results.

        Args:
            top_results (list): A list of tuples containing the top results from the search.

        Returns:
            tuple: A tuple containing two elements:
                - str: The prepared context as a string, with each result prefixed by its citation number.
                - dict: A dictionary of citation details, where keys are citation numbers and values are
                        dictionaries containing 'authors', 'published', and 'source' information.
        """
        context = []
        citation_details = {}
        for i, (_, _, uid, text) in enumerate(top_results, start=1):
            self.cur.execute("SELECT Authors, Published, Source FROM articles WHERE id = ?", [uid])
            authors, published, source = self.cur.fetchone()
            citation_details[i] = {"authors": authors, "published": published, "source": source}
            context.append(f"[{i}]: {text}")
        return "\n\n".join(context), citation_details

    def format_citations(self, summary, citation_details):
        """
        Replaces simplified citations with formatted citations including DOI links.
        Removes duplicate citation numbers.
        """
        def format_single_citation(citation_number):
            details = citation_details[citation_number]
            first_author = details['authors'].split(',')[0].split()[-1] if details['authors'] else "Unknown"
            year = self._get_year(details['published'])
            doi = details['source'].split('.pdf')[0].replace('_', '/') if details['source'] else "Unknown"
            return f'[{first_author} et al., {year}](https://doi.org/{doi})'

        def replace_citation(match):
            citation_numbers = re.split(r'[,;]\s*', match.group(1))
            # Remove duplicates while preserving order
            unique_numbers = []
            seen = set()
            for num in citation_numbers:
                num = int(num.strip())
                if num not in seen:
                    unique_numbers.append(num)
                    seen.add(num)
            
            formatted_citations = [format_single_citation(num) for num in unique_numbers]
            return f'({", ".join(formatted_citations)})'

        citation_pattern = r'\[(\d+(?:\s*[,;]\s*\d+)*)\]'
        return re.sub(citation_pattern, replace_citation, summary)

    def _get_year(self, published):
        """
        Extracts the year from a given date string or integer.

        Args:
            published: input date (string or integer)

        Returns:
            year as a string, or "n.d." if the year couldn't be determined
        """
        if published:
            try:
                # If published is already an integer, assume it's a year
                if isinstance(published, int):
                    return str(published)

                # If it's a string, try parsing it
                if isinstance(published, str):
                    # First, try the original format
                    try:
                        return str(datetime.strptime(published, "%Y-%m-%d %H:%M:%S").year)
                    except ValueError:
                        # If that fails, use the more flexible dateutil parser
                        parsed_date = parser.parse(published, default=datetime(1, 1, 1))
                        
                        # Only return the year if it was actually present in the input
                        if parsed_date.year != 1:
                            return str(parsed_date.year)

            except (ValueError, TypeError, parser.ParserError):
                # If all parsing attempts fail, fall through to return "n.d."
                pass

        return "n.d."

    def build(self, queries, options, output):
        """
        Builds a report using a list of input queries.

        Processes each query to generate a report section containing:
        - Query details
        - Highlights from top results
        - Generated summary with citations
        - Detailed articles table

        Args:
            queries: List of query configurations to process
            options: Dictionary of report generation options
            output: Path where the report will be saved

        Prints:
            - Start message when report generation begins
            - Progress bars for query processing
            - Completion message with output file location
        """
        print("\n=== Starting Report Generation ===")
        
        # Default to 50 documents if not specified
        topn = options.get("topn", 50)

        for name, config in tqdm(queries, desc="Processing queries"):
            query = config["query"]
            columns = config["columns"]

            # Write query string
            self.query(output, name, query)

            # Write separator
            self.separator(output)

            # Query for best matches
            results = Query.search(
                self.embeddings, self.cur, query, topn, options.get("threshold")
            )

            # Generate highlights section
            self.section(output, "Highlights")

            # Generate highlights
            self.highlights(output, results, int(topn / 10))

            # Separator between highlights and articles
            self.separator(output)

            # Generate summary
            summary, citation_details = self.generate_summary(results, 20, query)
            formatted_summary = self.format_citations(summary, citation_details)

            # Write summary section
            self.section(output, "Summary")
            self.write(output, formatted_summary)

            # Separator between summary and articles
            self.separator(output)

            # Generate articles section
            self.section(output, "Articles")

            # Generate table headers
            self.headers([column["name"] for column in columns], output)

            # Generate table rows
            self.articles(output, topn, (name, query, columns), results)

            # Write section separator
            self.separator(output)

        # Add completion message with clean file path
        output_path = output.name if hasattr(output, 'name') else str(output)
        # Convert /work/reports to ./reports for user clarity
        output_path = output_path.replace('/work/reports', './reports')
        print(f"\n✓ Report saved to: {output_path}")

    def highlights(self, output, results, topn):
        """
        Builds a highlights section.

        Args:
            output: output file
            results: search results
            topn: number of results to return
        """

        # Extract top sections as highlights
        for highlight in Query.highlights(results, topn):
            # Get matching article
            uid = [article for _, _, article, text in results if text == highlight][0]
            self.cur.execute(
                "SELECT Authors, Reference FROM articles WHERE id = ?", [uid]
            )
            article = self.cur.fetchone()

            # Write out highlight row
            self.highlight(output, article, highlight)

    def articles(self, output, topn, metadata, results):
        """
        Builds an articles section.

        Args:
            output: output file
            topn: number of documents to return
            metadata: query metadata
            results: search results
        """

        # Unpack metadata
        _, query, _ = metadata

        # Retrieve list of documents
        documents = (
            Query.all(self.cur) if query == "*" else Query.documents(results, topn)
        )

        # Collect matching rows
        rows = []

        for uid in tqdm(documents, desc="Processing documents", unit="doc"):
            # Get article metadata
            self.cur.execute(
                "SELECT Published, Title, Reference, Publication, Source, Entry, Id FROM articles WHERE id = ?",
                [uid],
            )
            article = self.cur.fetchone()

            # Calculate derived fields
            calculated = self.calculate(uid, metadata)

            # Builds a row for article
            rows.append(self.buildRow(article, documents[uid], calculated))

        # Print report by published desc
        for row in tqdm(sorted(rows, key=lambda x: x["Date"], reverse=True), desc="Writing rows"):
            row = [row[column] for column in self.names]

            # Write out row
            self.writeRow(output, row)

    def calculate(self, uid, metadata):
        """
        Builds a dict of calculated fields for a given document.
        """
        # Get article metadata and abstract
        self.cur.execute("""
            SELECT a.Title, s.Text as Abstract 
            FROM articles a
            LEFT JOIN sections s ON a.id = s.article 
            WHERE a.id = ? AND s.Name = 'ABSTRACT'
            LIMIT 1
        """, [uid])
        
        result = self.cur.fetchone()
        title = result[0] if result else ""
        abstract = result[1] if result else ""

        # Get introduction for additional context
        self.cur.execute("""
            SELECT Text 
            FROM sections 
            WHERE article = ? AND Name = 'INTRODUCTION'
            LIMIT 1
        """, [uid])
        
        intro = self.cur.fetchone()
        introduction = intro[0] if intro else ""

        # Parse column parameters
        fields, params = self.params(metadata)

        # Different type of calculations
        #  1. Similarity query
        #  2. Extractor query (similarity + question)
        #  3. Question-answering on other field
        queries, extractions, questions = [], [], []

        # Retrieve indexed document text for article
        sections = self.sections(uid)
        texts = [text for _, text in sections]

        for name, query, question, snippet, _, _, matches, _ in params:
            if query.startswith("$"):
                questions.append((name, query.replace("$", ""), question, snippet))
            elif matches:
                queries.append((name, query, matches))
            else:
                extractions.append((name, query, question, snippet))

        # Run all extractor queries against document text
        results = self.extractor.query([query for _, query, _ in queries], texts)

        # Process queries with matches
        for x, (name, query, matches) in enumerate(queries):
            if results[x]:
                topn = [text for _, text, _ in results[x]][:matches]
                value = [self.resolve(params, sections, uid, name, value) for value in topn]
                fields[name] = "\n\n".join(value) if value else ""
            else:
                fields[name] = ""

        # Add extraction fields
        if extractions:
            for name, value in self.extractor(extractions, texts):
                fields[name] = self.resolve(params, sections, uid, name, value) if value else ""

        # Add question fields with enhanced context
        names, qa, contexts, snippets = [], [], [], []
        for name, query, question, snippet in questions:
            names.append(name)
            qa.append(question)
            
            # Build enhanced context by combining:
            # 1. Title and abstract for high-level context
            # 2. Introduction for background
            # 3. Field-specific context
            field_context = fields[query] if query in fields else ""
            
            # Get relevant sections using similarity search
            self.cur.execute("""
                SELECT Text 
                FROM sections 
                WHERE article = ? AND Name NOT IN ('ABSTRACT', 'INTRODUCTION')
            """, [uid])
            
            other_sections = [row[0] for row in self.cur.fetchall()]
            relevant_sections = self.extractor.query([question], other_sections)[0] if other_sections else []
            relevant_text = "\n\n".join([text for _, text, _ in relevant_sections[:2]]) if relevant_sections else ""
            
            enhanced_context = (
                f"Title: {title}\n\n"
                f"Abstract: {abstract}\n\n"
                f"Introduction: {introduction}\n\n"
                f"Relevant Context: {field_context}\n\n"
                f"Additional Context: {relevant_text}"
            ).strip()
            
            contexts.append(enhanced_context)
            snippets.append(snippet)

        answers = self.extractor.answers(qa, contexts)
        for (name, answer), snippet in zip(answers, snippets):
            value = answer if isinstance(answer, str) else answer[0]
            fields[name] = self.resolve(params, sections, uid, name, value) if value else ""

        return fields

    def params(self, metadata):
        """
        Process and prepare parameters using input metadata.

        Args:
            metadata: query metadata

        Returns:
            fields, params - constant field values, query parameters for query columns
        """

        # Derived field values
        fields = {}

        # Query column parameters
        params = []

        # Unpack metadata
        _, _, columns = metadata

        for column in columns:
            # Constant column
            if "constant" in column:
                fields[column["name"]] = column["constant"]
            # Question-answer column
            elif "query" in column:
                # Query variable substitutions
                query = self.variables(column["query"], metadata)
                question = (
                    self.variables(column["question"], metadata)
                    if "question" in column
                    else query
                )

                # Additional context parameters
                section = column.get("section", False)
                surround = column.get("surround", 0)
                matches = column.get("matches", 0)
                dtype = column.get("dtype")
                snippet = column.get("snippet", False)
                snippet = True if section or surround else snippet

                params.append(
                    (
                        column["name"],
                        query,
                        question,
                        snippet,
                        section,
                        surround,
                        matches,
                        dtype,
                    )
                )

        return fields, params

    def variables(self, value, metadata):
        """
        Runs variable substitution for value.

        Args:
            value: input value
            metadata: query metadata

        Returns:
            value with variable substitution
        """

        name, query, _ = metadata

        # Cleanup name for queries
        name = name.replace("_", "").lower()
        query = query.lower()

        if value:
            value = value.replace("$NAME", name).replace("$QUERY", query)

        return value

    def sections(self, uid):
        """
        Retrieves all sections as list for article with given uid.

        Args:
            uid: article id

        Returns:
            list of section text elements
        """

        # Retrieve indexed document text for article
        self.cur.execute(Index.SECTION_QUERY + " WHERE article = ? ORDER BY id", [uid])

        # Get list of document text sections
        sections = []
        for sid, name, text in self.cur.fetchall():
            if (
                not self.embeddings.isweighted()
                or not name
                or not re.search(Index.SECTION_FILTER, name.lower())
                or self.options.get("allsections")
            ):
                # Check that section has at least 1 token
                if Tokenizer.tokenize(text):
                    sections.append((sid, text))

        return sections

    def resolve(self, params, sections, uid, name, value):
        """
        Fully resolves a value from an extractor call.

         - If section=True, this method pull the full section text
         - If surround is specified, this method will pull the surrounding text
         - Otherwise, the original value is returned

        Args:
            params: query parameters
            sections: section text
            uid: article id
            name: column name
            value: initial query value after running through extractor process

        Returns:
            resolved value
        """

        # Get all column parameters
        index = [params.index(x) for x in params if x[0] == name][0]
        _, _, _, _, section, surround, _, dtype = params[index]

        if value:
            # Find matching section
            sid = [sid for sid, text in sections if value in text]

            if sid:
                sid = sid[0]

                if section:
                    # Get full text for matching subsection
                    value = self.subsection(uid, sid)
                elif surround:
                    value = self.surround(uid, sid, surround)

            # Column dtype formatting
            if dtype == "int":
                value = Column.integer(value)
            elif isinstance(dtype, list):
                value = Column.categorical(self.labels, value, dtype)
            elif dtype in ["days", "weeks", "months", "years"]:
                value = Column.duration(value, dtype)

        return value

    def subsection(self, uid, sid):
        """
        Extracts all subsection text for columns with section=True.

        Args:
            uid: article id
            sid: section id

        Returns:
            full text for matching section
        """

        self.cur.execute(
            "SELECT Text FROM sections WHERE article = ? AND name = (SELECT name FROM sections WHERE id = ?)",
            [uid, sid],
        )
        return " ".join([x[0] for x in self.cur.fetchall()])

    def surround(self, uid, sid, size):
        """
        Extracts surrounding text for section with specified id.

        Args:
            uid: article id
            sid: section id
            size: number of surrounding lines to extract from each side

        Returns:
            matching text with surrounding context
        """

        self.cur.execute(
            "SELECT Text FROM sections WHERE article = ? AND id in (SELECT id FROM sections WHERE id >= ? AND id <= ?) AND "
            + "name = (SELECT name FROM sections WHERE id = ?)",
            [uid, sid - size, sid + size, sid],
        )

        return " ".join([x[0] for x in self.cur.fetchall()])

    def cleanup(self, outfile):
        """
        Allow freeing or cleaning up resources.

        Args:
            outfile: output file path
        """

    def query(self, output, task, query):
        """
        Writes query.

        Args:
            output: output file
            task: task name
            query: query string
        """

    def section(self, output, name):
        """
        Writes a section name

        Args:
            output: output file
            name: section name
        """

    def highlight(self, output, article, highlight):
        """
        Writes a highlight row

        Args:
            output: output file
            article: article reference
            highlight: highlight text
        """

    def headers(self, columns, output):
        """
        Writes table headers.

        Args:
            columns: column names
            output: output file
        """

    def buildRow(self, article, sections, calculated):
        """
        Converts a document to a table row.

        Args:
            article: article
            sections: text sections for article
            calculated: calculated fields
        """

    def writeRow(self, output, row):
        """
        Writes a table row.

        Args:
            output: output file
            row: output row
        """

    def separator(self, output):
        """
        Writes a separator between sections
        """

