"""
Report module
"""

import regex as re
import json
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

        # Initialize the Summarizer first
        llm_mode = options.get("llm_mode", "local")
        if llm_mode == "api":
            api_provider = options.get("api", {}).get("provider", "openai")
        else:
            local_opts = options.get("local", {})
            provider = local_opts.get("provider", "ollama")
            gpu_strategy = local_opts.get("gpu_strategy", "auto")
        
        self.summarizer = Summarizer(
            llm_options=options,
            mode=llm_mode,
            provider=api_provider if llm_mode == "api" else provider if llm_mode == "local" else None,
            gpu_strategy=gpu_strategy if llm_mode == "local" else None
        )

        # Question-answering model setup after summarizer
        qa_provider = options.get("api", {}).get("provider") if options.get("llm_mode") == "api" else None
        
        if qa_provider == "gemini":
            # For Gemini, we'll use the summarizer for QA
            self.extractor = self.summarizer
        else:
            # For other providers or default behavior, use the standard Extractor
            self.extractor = Extractor(
                self.similarity if self.similarity else self.embeddings,
                options["qa"] if options["qa"] else "NeuML/bert-small-cord19qa",
                minscore=options.get("minscore"),
                mintokens=options.get("mintokens"),
                context=options.get("context", 512),
            )

    def save_source_details(self, results, output_path):
        """
        Saves detailed information about the sources used in the summary to a JSON file.
        
        Args:
            results: List of tuples containing search results
            output_path: Path to save the source details file
        """
        sources = []
        for _, _, uid, text in results:
            # Get article metadata
            self.cur.execute("""
                SELECT Title, Authors, Published, Source, Reference, Publication
                FROM articles 
                WHERE id = ?
            """, [uid])
            title, authors, published, source, reference, publication = self.cur.fetchone()
            
            # Get section name and full context
            self.cur.execute("""
                SELECT Name, Text
                FROM sections
                WHERE article = ? AND Text LIKE ?
            """, [uid, f"%{text}%"])
            section_name, full_text = self.cur.fetchone() or (None, None)
            
            # Format the date
            if published:
                try:
                    if isinstance(published, str):
                        published = parser.parse(published).strftime("%Y-%m-%d")
                    elif isinstance(published, int):
                        published = str(published)
                except (ValueError, TypeError):
                    published = str(published)

            # Build source entry
            source_entry = {
                "title": title,
                "authors": authors.split(", ") if authors else [],
                "published_date": published,
                "doi": source.split(".pdf")[0].replace("_", "/") if source else None,
                "url": reference,
                "publication": publication,
                "section": {
                    "name": section_name,
                    "text": text,
                    "full_context": full_text
                }
            }
            sources.append(source_entry)

        # Save to file
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump({"sources": sources}, f, indent=2, ensure_ascii=False)

    def generate_summary(self, results, topn, query, columns=None):
        """
        Generates a summary using the specified model based on the top results and the query.
        
        Args:
            results: Search results
            topn: Number of top results to use
            query: Main query string
            columns: List of column configurations to guide the summary
        """
        print(f"\nGenerating summary for query: {query}... using: \n"
              f"- Model: {self.summarizer.model_name} \n"
              f"- Provider: {self.summarizer.provider} \n"
              f"- Mode: {self.summarizer.mode}")
        
        top_results = results[:topn]
        context, citation_details = self._prepare_context(top_results)
        
        # Build enhanced query incorporating column configurations
        enhanced_query = query
        if columns:
            aspects = []
            for col in columns:
                if isinstance(col, dict) and "query" in col:
                    aspect = f"Information about '{col['query']}'"
                    if "question" in col:
                        aspect += f", specifically '{col['question']}'"
                    aspects.append(aspect)
            
            if aspects:
                enhanced_query += "\n\nPlease ensure the summary addresses these specific aspects:\n- "
                enhanced_query += "\n- ".join(aspects)
        
        # Generate the summary with a single call
        summary = self.summarizer._generate_text(
            self._get_summary_prompt(context, enhanced_query),
            self._get_summary_system_prompt()
        )
        formatted_summary = self.format_citations(summary, citation_details)
        
        # Save source details to a JSON file
        output_base = getattr(self.options.get("output", ""), "name", "report")
        if isinstance(output_base, str):
            output_base = output_base.replace(".md", "").replace(".csv", "")
        sources_file = f"{output_base}_sources.json"
        self.save_source_details(top_results, sources_file)
        
        return formatted_summary, citation_details

    def _get_summary_prompt(self, context, query):
        """
        Get the prompt for summary generation.
        """
        example_paragraph = (
            "Recent advances in machine learning have revolutionized oceanographic research. \"Deep learning models have "
            "enabled unprecedented accuracy in predicting ocean temperature patterns\" [1.2], while \"satellite data combined "
            "with neural networks has improved our understanding of global ocean circulation\" [1.3]. These technological "
            "breakthroughs have led to \"more precise forecasting of extreme weather events and their impacts on marine ecosystems\" [2.1]."
        )

        return (
            f"Write a comprehensive, flowing summary addressing this query: '{query}'\n\n"
            f"Important Instructions:\n"
            f"- Use ONLY the information from the provided context\n"
            f"- Quote directly from the sources using double quotation marks\n"
            f"- Include citation numbers [article.paragraph] immediately after each quote\n"
            f"- Write in clear paragraphs WITHOUT any headings or sections\n"
            f"- Make the text flow naturally from one topic to the next\n"
            f"- Use no more than 3 citations per sentence\n"
            f"- Try not to cite the same article several times in the same paragraph, i.e. synthesize information from multiple articles and try to cite different articles in each paragraph\n"
            f"- Ensure citations are relevant to the discussion\n"
            f"- Do NOT use bullet points or numbered lists in your response\n"
            f"- Do NOT refer to the sources in the context as texts\n"
            f"- Write ONLY in connected paragraphs\n"
            f"- Ensure smooth transitions between ideas\n"
            f"- Be thorough but concise\n"
            f"- Maintain academic writing style\n\n"
            f"Context:\n{context}\n\n"
            f"Example paragraph:\n{example_paragraph}\n\n"
            f"Remember: Your response should be ONLY in paragraph form, with no headings, sections, or bullet points.\n"
            f"Each citation should be in the format [article.paragraph] to reference specific paragraphs.\n"
            f"Summary:\n"
        )

    def _get_summary_system_prompt(self):
        """
        Get the system prompt for summary generation.
        """
        return """You are an AI assistant tasked with writing comprehensive, flowing summaries of scientific papers in paragraph form. Your output must be in clear paragraphs with no headings, sections, or bullet points.

        Your summaries should:
        1. Be written in clear, connected paragraphs without any headings or section breaks
        2. Directly quote relevant passages from the source texts, enclosing them in double quotation marks
        3. Include citation numbers (e.g., [1.2]) immediately after quotation marks
        4. Incorporate quotes and citations naturally into the paragraph flow
        5. Focus only on information relevant to the given query
        6. Be thorough yet concise
        7. Use only the provided citation numbers, not author names
        8. Flow naturally from one topic to the next without artificial breaks
        9. Maintain an academic writing style, i.e. write it as if you are writing an academic paper.
        10. Provide comprehensive coverage while avoiding redundancy
        11. Try not to cite the same article several times in the same paragraph, i.e. synthesize information from multiple articles and try to cite different articles in each paragraph

        Remember: The output should be a flowing narrative with NO headings, sections, or bullet points. Just clean, connected paragraphs."""

    def _prepare_context(self, top_results):
        """
        Prepares the context for summarization and citation details from top results.
        Includes the full text of each article, with paragraphs numbered for citation.

        Args:
            top_results (list): A list of tuples containing the top results from the search.

        Returns:
            tuple: A tuple containing two elements:
                - str: The prepared context as a string, with each paragraph prefixed by its citation number.
                - dict: A dictionary of citation details, where keys are citation IDs (article_num.para_num)
                       and values are dictionaries containing article metadata and paragraph text.
        """
        context = []
        citation_details = {}
        
        # Track processed articles to avoid duplicates
        processed_uids = set()
        article_num = 1
        
        for _, _, uid, matched_text in top_results:
            if uid in processed_uids:
                continue
                
            processed_uids.add(uid)
            
            # Get article metadata
            self.cur.execute("""
                SELECT Title, Authors, Published, Source, Reference, Publication
                FROM articles 
                WHERE id = ?
            """, [uid])
            title, authors, published, source, reference, publication = self.cur.fetchone()
            
            # Get all sections of the article in order
            self.cur.execute("""
                SELECT Name, Text
                FROM sections
                WHERE article = ?
                ORDER BY id
            """, [uid])
            sections = self.cur.fetchall()
            
            # Process article text with numbered paragraphs
            article_text = []
            article_text.append(f"Article {article_num}:")
            article_text.append(f"Title: {title}")
            if authors:
                article_text.append(f"Authors: {authors}")
            
            para_num = 1
            for section_name, section_text in sections:
                if section_name:
                    article_text.append(f"\n{section_name}:")
                
                # Split section into paragraphs and number each one
                paragraphs = [p.strip() for p in section_text.split('\n\n') if p.strip()]
                for paragraph in paragraphs:
                    citation_id = f"{article_num}.{para_num}"
                    
                    # Store citation details
                    citation_details[citation_id] = {
                        "authors": authors,
                        "published": published,
                        "source": source,
                        "title": title,
                        "reference": reference,
                        "publication": publication,
                        "section": section_name,
                        "text": paragraph,
                        "is_matched": matched_text in paragraph
                    }
                    
                    # Add paragraph with citation number
                    article_text.append(f"\n[{citation_id}] {paragraph}")
                    
                    # Add emphasis if this is a matched paragraph
                    if matched_text in paragraph:
                        article_text.append("[This paragraph was specifically matched for relevance to the query]")
                    
                    para_num += 1
            
            # Add the full article text to context
            context.append('\n'.join(article_text))
            article_num += 1
        
        return "\n\n---\n\n".join(context), citation_details

    def format_citations(self, summary, citation_details):
        """
        Replaces citations with formatted citations including DOI links and hover text.
        Handles paragraph-level citations in the format [article_num.para_num].
        """
        def format_single_citation(citation_id):
            details = citation_details[citation_id]
            first_author = details['authors'].split(',')[0].split()[-1] if details['authors'] else "Unknown"
            year = self._get_year(details['published'])
            doi = details['source'].split('.pdf')[0].replace('_', '/') if details['source'] else "Unknown"
            
            # Create hover text with section, title and the exact paragraph being cited
            hover_text = f"Title: {details['title']}"
            if details['section']:
                hover_text += f"\nSection: {details['section']}"
            hover_text += f"\n\nCited text:\n{details['text']}"
            
            # Escape quotes and newlines for title attribute
            hover_text = hover_text.replace('"', '&quot;').replace('\n', '&#10;')
            
            return f'[{first_author} et al., {year}](https://doi.org/{doi} "{hover_text}")'

        def replace_citation(match):
            citation_ids = re.split(r'[,;]\s*', match.group(1))
            # Remove duplicates while preserving order
            unique_ids = []
            seen = set()
            for cid in citation_ids:
                if cid not in seen and cid in citation_details:
                    unique_ids.append(cid)
                    seen.add(cid)
            
            formatted_citations = [format_single_citation(cid) for cid in unique_ids]
            return f'({", ".join(formatted_citations)})'

        # Match citations in the format [1.2] or [1.2, 1.3, 2.1]
        citation_pattern = r'\[(\d+\.\d+(?:\s*[,;]\s*\d+\.\d+)*)\]'
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
            summary, citation_details = self.generate_summary(results, min(20, int(topn/2)), query, columns)
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
            SELECT a.Title, s.Text as Abstract, a.Authors, a.Published, a.Source, a.Reference, a.Publication
            FROM articles a
            LEFT JOIN sections s ON a.id = s.article 
            WHERE a.id = ? AND s.Name = 'ABSTRACT'
            LIMIT 1
        """, [uid])
        
        result = self.cur.fetchone()
        title = result[0] if result else ""
        abstract = result[1] if result else ""
        authors = result[2] if result else ""
        published = result[3] if result else ""
        source = result[4] if result else ""
        reference = result[5] if result else ""
        publication = result[6] if result else ""

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
                fields[name] = self._format_table_cell(" ".join(value)) if value else ""
            else:
                fields[name] = ""

        # Add extraction fields
        if extractions:
            for name, value in self.extractor(extractions, texts):
                fields[name] = self._format_table_cell(self.resolve(params, sections, uid, name, value)) if value else ""

        # Add question fields with enhanced context
        if questions:
            # Build article metadata context
            metadata_context = (
                f"Title: {title}\n"
                f"Authors: {authors}\n"
                f"Published: {published}\n"
                f"Publication: {publication}\n"
                f"DOI: {source.split('.pdf')[0].replace('_', '/')} if source else 'Unknown'\n"
                f"Reference: {reference}\n\n"
            )

            # Process each question
            for name, query, question, snippet in questions:
                # Get field-specific context if query references another field
                field_context = fields[query] if query in fields else ""
                
                # Build comprehensive context
                context = (
                    f"{metadata_context}\n"
                    f"Abstract:\n{abstract}\n\n"
                    f"Introduction:\n{introduction}\n\n"
                )
                
                # Add field-specific context if available
                if field_context:
                    context += f"Relevant Context:\n{field_context}\n\n"
                
                # Add other relevant sections
                self.cur.execute("""
                    SELECT Name, Text 
                    FROM sections 
                    WHERE article = ? AND Name NOT IN ('ABSTRACT', 'INTRODUCTION')
                    ORDER BY id
                """, [uid])
                
                other_sections = self.cur.fetchall()
                if other_sections:
                    # Use similarity search to find relevant sections
                    section_texts = [text for _, text in other_sections]
                    relevant_sections = self.extractor.query([question], section_texts)[0] if section_texts else []
                    
                    if relevant_sections:
                        context += "Additional Relevant Sections:\n"
                        for _, text, _ in relevant_sections[:2]:  # Include top 2 most relevant sections
                            context += f"{text}\n\n"
                
                # Generate answer and format for table cell
                if isinstance(self.extractor, Summarizer):
                    # For Gemini, use _generate_gemini directly with table formatting instructions
                    prompt = (
                        "Based on the following context, provide a MAXIMUM of 2 CONCISE SENTENCES answering this question:\n"
                        f"{question}\n\n"
                        "Important:\n"
                        "- Response must be MAXIMUM of 2 CONCISE SENTENCES\n"
                        "- No bullet points or lists\n"
                        "- No section headers\n"
                        "- Keep it concise and focused\n"
                        "- Avoid line breaks\n\n"
                        "Context:\n"
                        f"{context}"
                    )
                    answer = self.extractor._generate_gemini(prompt, "", is_summary=False)
                else:
                    # For other extractors, use standard answer method
                    answer = self.extractor.answers([question], [context])[0][1]
                    answer = answer if isinstance(answer, str) else answer[0]
                
                fields[name] = self._format_table_cell(self.resolve(params, sections, uid, name, answer)) if answer else ""

        return fields

    def _format_table_cell(self, text):
        """
        Formats text for table cell display, ensuring it's a clean, single paragraph.
        
        Args:
            text: The text to format
            
        Returns:
            str: Formatted text as a clean, single paragraph
        """
        if not text:
            return ""
            
        # Remove any markdown-style headers
        text = re.sub(r'^#+\s+.*$', '', text, flags=re.MULTILINE)
        
        # Remove bullet points and numbered lists
        text = re.sub(r'^\s*[-*•]\s+', '', text, flags=re.MULTILINE)
        text = re.sub(r'^\s*\d+\.\s+', '', text, flags=re.MULTILINE)
        
        # Collapse multiple spaces and newlines
        text = re.sub(r'\s+', ' ', text)
        
        # Remove any remaining markdown formatting
        text = re.sub(r'[_*~`]', '', text)
        
        # Clean up the text
        text = text.strip()
        
        return text

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
        Improves text extraction by:
        1. Maintaining section order
        2. Handling subsections properly
        3. Cleaning and normalizing text
        4. Filtering out non-content sections

        Args:
            uid: article id

        Returns:
            list of tuples containing (section_id, text)
        """
        # Retrieve indexed document text for article in order
        self.cur.execute("""
            SELECT s.id, s.Name, s.Text,
                   CASE 
                       WHEN s.Name = 'ABSTRACT' THEN 1
                       WHEN s.Name = 'INTRODUCTION' THEN 2
                       WHEN s.Name = 'METHODS' OR s.Name = 'METHODOLOGY' THEN 3
                       WHEN s.Name = 'RESULTS' THEN 4
                       WHEN s.Name = 'DISCUSSION' THEN 5
                       WHEN s.Name = 'CONCLUSION' OR s.Name = 'CONCLUSIONS' THEN 6
                       ELSE 7
                   END as section_order
            FROM sections s
            WHERE s.article = ?
            ORDER BY section_order, s.id
        """, [uid])

        # Get list of document text sections
        sections = []
        for sid, name, text, _ in self.cur.fetchall():  # Added _ to unpack the section_order
            # Skip non-content sections
            if name and re.search(r'^(REFERENCES|ACKNOWLEDGMENTS?|APPENDIX|SUPPLEMENTARY)', name.upper()):
                continue

            # Check if section should be included based on embeddings weight and section filter
            if (not self.embeddings.isweighted() or 
                not name or 
                not re.search(Index.SECTION_FILTER, name.lower()) or 
                self.options.get("allsections")):
                
                # Clean and normalize text
                if text:
                    # Remove excessive whitespace
                    text = re.sub(r'\s+', ' ', text.strip())
                    
                    # Remove figure/table references
                    text = re.sub(r'(Fig\.|Figure|Table|Tab\.)\s*\d+[a-zA-Z]?', '', text)
                    
                    # Remove citations in parentheses
                    text = re.sub(r'\([^)]*\d{4}[^)]*\)', '', text)
                    
                    # Check that section has at least 1 token after cleaning
                    if Tokenizer.tokenize(text):
                        # Add section name as prefix for context if available
                        if name and name.upper() not in ['ABSTRACT', 'INTRODUCTION']:
                            text = f"{name}:\n{text}"
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

    def _get_first_stage_prompt(self, context, query):
        example_paragraph = (
            "Recent advances in machine learning have revolutionized oceanographic research. \"Deep learning models have "
            "enabled unprecedented accuracy in predicting ocean temperature patterns\" [1.2], while \"satellite data combined "
            "with neural networks has improved our understanding of global ocean circulation\" [1.3]. These technological "
            "breakthroughs have led to \"more precise forecasting of extreme weather events and their impacts on marine ecosystems\" [2.1]."
        )

        return (
            f"Write a flowing, paragraph-based summary addressing this query: '{query}'\n\n"
            f"Important Instructions:\n"
            f"- Use ONLY the information from the provided context\n"
            f"- Quote directly from the sources using double quotation marks\n"
            f"- Include citation numbers [article.paragraph] immediately after each quote\n"
            f"- Write in clear paragraphs WITHOUT any headings or sections\n"
            f"- Make the text flow naturally from one topic to the next\n"
            f"- Use no more than 3 citations per sentence\n"
            f"- Ensure citations are relevant to the discussion\n"
            f"- Do NOT use bullet points or numbered lists in your response\n"
            f"- Do NOT refer to the sources in the context as texts, just write the summary without mentioning the context\n"
            f"- Write ONLY in connected paragraphs\n\n"
            f"Context:\n{context}\n\n"
            f"Example paragraph:\n{example_paragraph}\n\n"
            f"Remember: Your response should be ONLY in paragraph form, with no headings, sections, or bullet points.\n"
            f"Each citation should be in the format [article.paragraph] to reference specific paragraphs.\n"
            f"Summary:\n"
        )

