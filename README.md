# bb-ai-oceanography

# bb-ai-oceanography: Full-Text Analysis of Machine Learning in Oceanography

This repository is an extended version of Taylor Denouden's [bb-ai-oceanography](https://github.com/taylordenouden/bb-ai-oceanography) project. While the original repository focused on analyzing paper abstracts using Large Language Models (LLMs), this fork expands the scope to include:

1. Downloading full-text scientific papers related to machine learning applications in oceanography
2. Processing and extracting information from these papers
3. Analyzing the complete content of the papers, not just abstracts

By working with full-text papers, this project aims to provide a more comprehensive and in-depth analysis of the intersection between machine learning and oceanography.

## Project Overview

The project aims to provide a comprehensive analysis of the current state of machine learning in oceanography by:

1. Downloading full-text papers from various sources
2. Processing and extracting relevant information from the papers
3. Analyzing the content using natural language processing techniques
4. Visualizing the results to identify trends and patterns in the field

## Key Features

- Multi-source paper retrieval (Elsevier, Springer, Wiley, arXiv, Unpaywall)
- Full-text extraction from PDFs and XMLs
- Robust error handling and logging
- Rate-limited API calls to respect usage policies
- Machine learning-based clustering and labeling of papers
- Interactive visualization of the paper landscape

## Getting Started

### Prerequisites

- Python 3.7+
- Required Python packages (listed in `requirements.txt`)

### Installation

1. Clone the repository:
   ```
   git clone https://github.com/Spiffical/bb-ai-oceanography.git
   cd bb-ai-oceanography
   ```

2. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

3. Set up environment variables:
   Create a `.env` file in the project root and add your API keys and email address for Unpaywall:
   ```
   ELSEVIER_API_KEY=your_elsevier_api_key
   SPRINGER_API_KEY=your_springer_api_key
   WILEY_API_KEY=your_wiley_api_key
   UNPAYWALL_EMAIL=your_email_address
   ```

### Usage

To download papers:

```
python scripts/download_papers.py --csv path/to/your/doi_list.csv output_path
```

Or for a single DOI:

```
python scripts/download_papers.py --doi 10.1016/j.example.2023.123456 output_path
```

## Project Structure

- `scripts/`: Contains the main scripts for downloading and processing papers
- `utils/`: Utility functions for API calls, file handling, and PDF processing
- `notebooks/`: Jupyter notebooks for data analysis and visualization
- `data/`: Directory to store downloaded papers and processed data

## Acknowledgments

- This work builds upon the initial analysis by Leland McInnes at the Tutte Institute, and Taylor Denouden's [bb-ai-oceanography](https://github.com/taylordenouden/bb-ai-oceanography) project
- Thanks to all the publishers and platforms providing access to scientific literature