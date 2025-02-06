"""
Indexing module
"""

import os.path
import sqlite3
import sys
import os

import regex as re
import yaml
from dotenv import load_dotenv

from txtai.embeddings import Embeddings
from txtai.pipeline import Tokenizer
from txtai.vectors import WordVectors

# Load environment variables from .env file
load_dotenv(override=True)

class Index:
    """
    Methods to build a new sentence embeddings index.
    """

    # Section query and filtering logic constants
    SECTION_FILTER = r"background|(?<!.*?results.*?)discussion|introduction|reference"
    SECTION_QUERY = "SELECT Id, Name, Text FROM sections"

    @staticmethod
    def stream(dbfile, maxsize, scoring):
        """
        Streams documents from an articles.sqlite file. This method is a generator and will yield a row at time.

        Args:
            dbfile: input SQLite file
            maxsize: maximum number of documents to process
            scoring: True if index uses a scoring model, False otherwise
        """

        # Connection to database file
        db = sqlite3.connect(dbfile)
        cur = db.cursor()

        # Get total number of sections
        cur.execute("SELECT COUNT(*) FROM sections")
        total_sections = cur.fetchone()[0]
        print(f"\nTotal sections in database: {total_sections}")

        # Get number of articles with tags
        cur.execute("SELECT COUNT(*) FROM articles WHERE tags IS NOT NULL")
        tagged_articles = cur.fetchone()[0]
        print(f"Articles with tags: {tagged_articles}")

        # Get sections count for tagged articles
        cur.execute("""
            SELECT COUNT(*) 
            FROM sections s 
            WHERE article IN (SELECT id FROM articles WHERE tags IS NOT NULL)
        """)
        tagged_sections = cur.fetchone()[0]
        print(f"Sections from tagged articles: {tagged_sections}")

        # Select sentences from tagged articles
        query = (
            Index.SECTION_QUERY
            + " WHERE article in (SELECT article FROM articles a WHERE a.id = article AND a.tags IS NOT NULL)"
        )

        if maxsize > 0:
            print(f"\nLimiting to {maxsize} most recent articles")
            query += f" AND article in (SELECT id FROM articles ORDER BY entry DESC LIMIT {maxsize})"

        # Run the query
        cur.execute(query)

        count = 0
        skipped_empty = 0
        skipped_filtered = 0
        processed = 0
        
        for row in cur:
            # Unpack row
            uid, name, text = row
            count += 1

            if not text:
                skipped_empty += 1
                continue

            if not scoring or not name or not re.search(Index.SECTION_FILTER, name.lower()):
                # Tokenize text
                text = Tokenizer.tokenize(text) if scoring else text

                document = (uid, text, None)

                processed += 1
                if processed % 1000 == 0:
                    print(f"Processed {processed} documents", end="\r")

                # Skip documents with no tokens parsed
                if text:
                    yield document
                else:
                    skipped_empty += 1
            else:
                skipped_filtered += 1

        print(f"\nProcessing summary:")
        print(f"Total sections examined: {count}")
        print(f"Skipped due to empty text: {skipped_empty}")
        print(f"Skipped due to section filters: {skipped_filtered}")
        print(f"Successfully processed: {processed}")

        # Free database resources
        db.close()

    @staticmethod
    def config(vectors):
        """
        Builds embeddings configuration.

        Args:
            vectors: vector model path or configuration

        Returns:
            configuration
        """
        
        # Get HF token from environment
        hf_token = os.getenv('HUGGING_FACE_HUB_TOKEN') or os.getenv('HF_TOKEN')

        # Configuration as a dictionary
        if isinstance(vectors, dict):
            if hf_token:
                vectors['token'] = hf_token
            return vectors

        # Configuration as a YAML file
        if isinstance(vectors, str) and vectors.endswith(".yml"):
            with open(vectors, "r", encoding="utf-8") as f:
                config = yaml.safe_load(f)
                if hf_token:
                    config['token'] = hf_token
                return config

        # Configuration for word vectors model
        if WordVectors.isdatabase(vectors):
            config = {"path": vectors, "scoring": "bm25", "pca": 3, "quantize": True}
            if hf_token:
                config['token'] = hf_token
            return config

        # Use vector path if provided, else use default txtai configuration
        config = {"path": vectors} if vectors else None
        if config and hf_token:
            config['token'] = hf_token
        return config

    @staticmethod
    def embeddings(dbfile, vectors, maxsize):
        """
        Builds a sentence embeddings index.

        Args:
            dbfile: input SQLite file
            vectors: path to vectors file or configuration
            maxsize: maximum number of documents to process

        Returns:
            embeddings index
        """

        # Load .env file and get HF token
        load_dotenv()  # Reload to ensure we have the latest values
        hf_token = os.getenv('HUGGING_FACE_HUB_TOKEN') or os.getenv('HF_TOKEN')
        print(f"Debug: Found HF token: {'Yes' if hf_token else 'No'}")

        # Read config and create Embeddings instance
        config = Index.config(vectors)
        print(f"Debug: Initial config: {config}")
        
        # If no config exists, create a default one
        if not config:
            config = {
                'path': 'sentence-transformers/all-MiniLM-L6-v2'  # This is a commonly used model that should work
            }
            print("Debug: Created new config with default model path")
        
        # Ensure token is in config
        if hf_token:
            config['token'] = hf_token
            # Also set it as an environment variable for transformers library
            os.environ['HUGGING_FACE_HUB_TOKEN'] = hf_token
            os.environ['HF_TOKEN'] = hf_token
            print("Debug: Added token to config and environment: ", hf_token)
        
        print(f"Debug: Final config being passed to Embeddings: {config}")

        embeddings = Embeddings(config)
        scoring = embeddings.isweighted()

        # Build scoring index if scoring method provided
        if scoring:
            embeddings.score(Index.stream(dbfile, maxsize, scoring))

        # Build embeddings index
        embeddings.index(Index.stream(dbfile, maxsize, scoring))

        return embeddings

    @staticmethod
    def run(path, vectors, maxsize=0):
        """
        Executes an index run.

        Args:
            path: model path
            vectors: path to vectors file or configuration, if None uses default path
            maxsize: maximum number of documents to process
        """

        dbfile = os.path.join(path, "articles.sqlite")

        print("Building new model")
        embeddings = Index.embeddings(dbfile, vectors, maxsize)
        embeddings.save(path)


if __name__ == "__main__":
    Index.run(
        sys.argv[1] if len(sys.argv) > 1 else None,
        sys.argv[2] if len(sys.argv) > 2 else None,
        int(sys.argv[3]) if len(sys.argv) > 3 else 0,
    )
