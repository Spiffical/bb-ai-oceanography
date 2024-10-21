"""
Report module
"""

import regex as re
from datetime import datetime
from dateutil import parser

from txtai.pipeline import Extractor, Labels, Similarity, Tokenizer

from ..index import Index
from ..query import Query

from .column import Column

from llama_cpp import Llama
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import torch

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
            context=options.get("context"),
        )

        # Load the specified model for summarization
        model_name = options.get("model", "google/gemma-2-9b-it")
        if "google/gemma" in model_name:
            print(f"Loading Gemma model: {model_name}")
            quantization_config = BitsAndBytesConfig(load_in_4bit=True)
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                quantization_config=quantization_config,
                device_map="auto",
            )
            self.is_gemma = True
            self.is_gguf = False
        elif "MaziyarPanahi/Mistral-7B-Instruct-v0.3-GGUF" in model_name:
            # Specify the exact file you want to download
            filename = "Mistral-7B-Instruct-v0.3.Q8_0.gguf" 
            print(f"Downloading model {filename} from Hugging Face...")
            self.model = Llama.from_pretrained(
                repo_id="MaziyarPanahi/Mistral-7B-Instruct-v0.3-GGUF",
                filename=filename,
                n_ctx=4096, 
                n_batch=512,  # You can adjust this based on your GPU memory
                n_gpu_layers=-1,  # This will offload all layers to GPU
                verbose=False
            )
            self.is_gguf = True
            self.is_gemma = False
        elif "bartowski/Ministral-8B-Instruct-2410-HF-GGUF-TEST" in model_name:
            print(f"Loading Ministral-8B-Instruct model...")
            self.model = Llama.from_pretrained(
                repo_id="bartowski/Ministral-8B-Instruct-2410-HF-GGUF-TEST",
                # filename="Ministral-8B-Instruct-2410-HF-Q8_0.gguf",
                filename="Ministral-8B-Instruct-2410-HF-f16.gguf",
                n_ctx=4096,
                n_batch=512,
                n_gpu_layers=-1,
                verbose=False
            )
            self.is_gguf = True
            self.is_gemma = False
        else:
            # Use Hugging Face Transformers for non-GGUF models
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.bfloat16,  # Use float16 instead of bfloat16
                device_map="auto",
            )
            self.is_gguf = False
            self.is_gemma = False
        
        self.max_length = 4096  # Adjust this to match the n_ctx value

    def generate_summary(self, results, topn, query):
        """
        Generates a summary using the specified model based on the top results and the query.
        """
        top_results = results[:topn]
        context, citation_details = self._prepare_context(top_results)
        
        # First input text to the model
        input_prompt = (
            f"Task: Generate a 1000 word summary based on the provided context, addressing the query: '{query}'\n\n"
            f"Instructions:\n"
            f"1. Use ONLY the information from the provided context.\n"
            f"2. Quote directly from the context, using double quotation marks.\n"
            f"3. Include the citation number (e.g., [1]) immediately after each quotation.\n"
            f"4. Incorporate quotes and citations seamlessly into your summary.\n"
            f"5. Focus on information relevant to the query.\n"
            f"6. Be concise and informative.\n"
            f"7. Do not mention that you are quoting or summarizing.\n"
            f"8. Do not include author names in citations, use only the provided numbers.\n"
            f"9. Try to tell a story with the information provided, using the context and query to provide details and information.\n"
            f"10: Use AT MOST 3 citations per quote, and ensure they are directly relevant to the quote and the query.\n"
            f"11: DO NOT write the references at the end of the summary.\n"
            f"12: You DO NOT need to cite the sources in order, but you should use them if they are relevant to the query.\n"
            f"Context (Sources to quote from):\n{context}\n\n"
            f"Summary:"
        )

        first_summary = self._generate_text(input_prompt, self._get_first_stage_prompt())

        # Second input text to the model
        input_prompt = (
            f"Task: Refine the following summary, ensuring the information from each source is directly relevant to where it is used.\n\n"
            f"Original Summary: {first_summary}\n\n"
            f"Context (Sources to quote from):\n{context}\n\n"
            f"Instructions:\n"
            f"1. Maintain the overall structure and content of the summary.\n"
            f"2. Ensure all quotes are properly enclosed in double quotation marks.\n"
            f"3. Verify that all citations are in the format [n] or [n, m, ...], where n and m are numbers.\n"
            f"4. Remove any author name citations that may have been included.\n"
            f"5. Do not add any new information or change the meaning of the text.\n"
            f"6. Ensure that the summary ends with a conclusion.\n"
            f"7. Ensure that the summary contains NO notes, NO information irrelevant to the query or summary, and NO lists of the context provided.\n\n"
            f"Revised Summary:"
        )
        final_summary = self._generate_text(input_prompt, self._get_second_stage_prompt())
        
        return final_summary, citation_details

    def _prepare_context(self, top_results):
        context = []
        citation_details = {}
        for i, (_, _, uid, text) in enumerate(top_results, start=1):
            self.cur.execute("SELECT Authors, Published, Source FROM articles WHERE id = ?", [uid])
            authors, published, source = self.cur.fetchone()
            citation_details[i] = {"authors": authors, "published": published, "source": source}
            context.append(f"[{i}]: {text}")
        return "\n\n".join(context), citation_details

    def _generate_text(self, user_prompt, system_prompt):
        if not self.is_gemma:
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ]
        else:
            messages = [{"role": "user", "content": user_prompt}]
        
        if self.is_gguf:
            output = self.model.create_chat_completion(
                messages=messages,
                max_tokens=1000,
                temperature=0.7,
                top_p=0.9,
                # stop=["<|end_of_turn|>"]  # Add a stop token
            )
            generated_text = output['choices'][0]['message']['content']
        elif self.is_gemma:
            input_text = f"<start_of_turn>user\n{user_prompt}<end_of_turn>\n<start_of_turn>model\n"
            input_ids = self.tokenizer(input_text, return_tensors="pt").to(self.model.device)
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **input_ids,
                    max_new_tokens=1000,
                    min_new_tokens=100,
                    temperature=0.7,
                    do_sample=True,
                    top_p=0.9,
                    pad_token_id=self.tokenizer.eos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                )
            
            generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        else:
            inputs = self.tokenizer(self.tokenizer.apply_chat_template(messages, tokenize=False), 
                                    return_tensors="pt", truncation=True, max_length=self.max_length)
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=1000,
                    min_new_tokens=100,
                    temperature=0.7,
                    do_sample=True,
                    top_p=0.9,
                    pad_token_id=self.tokenizer.eos_token_id,  # Use EOS token for padding
                    eos_token_id=self.tokenizer.eos_token_id,  # Specify EOS token
                )
            
            generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Remove "assistant" prefix and any leading whitespace
        summary = generated_text.split("Summary:")[-1].strip()
        if summary.lower().startswith("assistant"):
            summary = summary[len("assistant"):].lstrip()
        elif summary.lower().startswith("model"):
            summary = summary[len("model"):].lstrip()
        
        return summary

    def _get_first_stage_prompt(self):
        return """You are an AI assistant tasked with summarizing scientific papers in around 1000 words. Your summaries should:
        1. Directly quote relevant passages from the source (context) texts, enclosing them in double quotation marks.
        2. MAKE SURE to include the citation number (e.g., [1]) immediately after the quotation marks.
        3. Incorporate these quotes and citations seamlessly into your summary.
        4. Do not mention that you are quoting or summarizing.
        5. Focus only on information relevant to the given query.
        6. Be concise and informative.
        7. Do not include author names in citations, use only the provided numbers."""

    def _get_second_stage_prompt(self):
        return """You are an AI assistant tasked with refining summaries of scientific papers. Your task is to:
        1. Ensure all citations are in the format [n] or [n, m, ...], where n and m are numbers.
        2. Remove any author name citations that may have been included (e.g., (John et al., 2024)).
        3. Maintain the overall structure and content of the summary.
        4. Ensure quotes are properly enclosed in double quotation marks.
        5. Do not add any new information or change the meaning of the text."""

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
        Builds a report using a list of input queries

        Args:
            queries: queries to execute
            options: report options
            output: output I/O object
        """

        # Default to 50 documents if not specified
        topn = options.get("topn", 50)

        for name, config in queries:
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
            summary, citation_details = self.generate_summary(results, int(topn / 5), query)
            formatted_summary = self.format_citations(summary, citation_details)

            # Write summary section
            self.section(output, "Summary")
            self.write(output, formatted_summary)
            
            print(f"\nOriginal summary: {summary}")
            print(f"\nFormatted summary: {formatted_summary}\n")

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

        for x, uid in enumerate(documents):
            # Get article metadata
            self.cur.execute(
                "SELECT Published, Title, Reference, Publication, Source, Entry, Id FROM articles WHERE id = ?",
                [uid],
            )
            article = self.cur.fetchone()

            if x and x % 100 == 0:
                print(f"Processed {x} documents", end="\r")

            # Calculate derived fields
            calculated = self.calculate(uid, metadata)

            # Builds a row for article
            rows.append(self.buildRow(article, documents[uid], calculated))

        # Print report by published desc
        for row in sorted(rows, key=lambda x: x["Date"], reverse=True):
            # print(f"Available columns: {list(row.keys())}")  # Debug print
            # Convert row dict to list
            row = [row[column] for column in self.names]

            # Write out row
            self.writeRow(output, row)

    def calculate(self, uid, metadata):
        """
        Builds a dict of calculated fields for a given document. This method calculates
        constant field columns and derived query columns. Derived query columns run through
        an embedding search and either run an additional QA query to extract a value or
        use the top n embedding search matches.

        Args:
            uid: article id
            metadata: query metadata

        Returns:
            {name: value} containing derived column values
        """

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

        # Only execute embeddings queries for columns with matches set
        for x, (name, query, matches) in enumerate(queries):
            if results[x]:
                # Get topn text matches
                topn = [text for _, text, _ in results[x]][:matches]

                # Join results into String and return
                value = [
                    self.resolve(params, sections, uid, name, value) for value in topn
                ]
                fields[name] = "\n\n".join(value) if value else ""
            else:
                fields[name] = ""

        # Add extraction fields
        if extractions:
            for name, value in self.extractor(extractions, texts):
                # Resolves the full value based on column parameters
                fields[name] = (
                    self.resolve(params, sections, uid, name, value) if value else ""
                )

        # Add question fields
        names, qa, contexts, snippets = [], [], [], []
        for name, query, question, snippet in questions:
            names.append(name)
            qa.append(question)
            contexts.append(fields[query])
            snippets.append(snippet)

        answers = self.extractor.answers(qa, contexts)
        for (name, answer), snippet in zip(answers, snippets):
            # Resolves the full value based on column parameters
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
