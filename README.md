# bb-ai-oceanography: Full-Text Analysis of Machine Learning in Oceanography

This repository is an extended version of Taylor Denouden's [bb-ai-oceanography](https://github.com/taylordenouden/bb-ai-oceanography) project. While the original repository focused on analyzing paper *abstracts* using Large Language Models (LLMs), this fork expands the scope to include *full-text* analysis and fully-cited report generation of scientific papers related to machine learning applications in oceanography. Report generation could absolutely benefit from the work previously completed to identify topic areas, e.g. the report queries could focus on specific topic areas.

## Table of Contents

1. [Project Overview](#project-overview)
2. [Key Features](#key-features)
3. [Getting Started](#getting-started)
   - [Prerequisites](#prerequisites)
   - [Installation](#installation)
4. [Usage](#usage)
   - [Downloading Papers](#downloading-papers)
   - [Extracting Content with paperetl](#extracting-content-with-paperetl)
   - [Generating Reports with paperai](#generating-reports-with-paperai)
5. [Project Structure](#project-structure)
6. [Acknowledgments](#acknowledgments)

## Project Overview

The project aims to provide a comprehensive analysis of the current state of machine learning in oceanography by:

1. Downloading full-text scientific papers from various sources
2. Processing and extracting relevant information from the papers using paperetl
3. Analyzing the content using natural language processing techniques with paperai
4. Generating reports to identify trends and patterns in the field

## Key Features

- Multi-source paper retrieval (Elsevier, Springer, Wiley, arXiv, Unpaywall)
- Full-text extraction from PDFs and XMLs using GROBID and paperetl
- Robust error handling and logging
- Rate-limited API calls to respect usage policies
- Machine learning-based analysis and report generation using paperai

Changes made to `paperetl`:
- Extracting paragraphs instead of sentences for improved summary quality
- Improved error handling

Changes made to `paperai`  :
- Added a properly cited summary section, using an LLM, to the report

## Getting Started

### Prerequisites

- Python 3.7+
- Docker (for running [GROBID](https://github.com/kermitt2/grobid))
- NVIDIA GPU (optional, for faster processing)

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

3. Install paperetl and paperai:
   ```
   cd paperetl
   pip install .
   cd ../paperai
   pip install .
   cd ..
   ```

4. Set up environment variables if you are downloading papers from the Elsevier, Springer, and Wiley APIs:
   Create a `.env` file in the project root and add your API keys and email address for Unpaywall:
   ```
   ELSEVIER_API_KEY=your_elsevier_api_key
   SPRINGER_API_KEY=your_springer_api_key
   WILEY_API_KEY=your_wiley_api_key
   UNPAYWALL_EMAIL=your_email_address
   ```

## Usage

### Downloading Papers

To download papers:

```
python scripts/download_papers.py --csv path/to/your/doi_list.csv output_path
```

Or for a single DOI:

```
python scripts/download_papers.py --doi 10.1016/j.example.2023.123456 output_path
```

### Extracting Content with paperetl

1. Start GROBID with the custom configuration:

   ```
   sudo docker run --rm --gpus all --init --ulimit core=0 -p 8070:8070 -v /path/to/bb-ai-oceanography/config/grobid.yaml:/opt/grobid/grobid-home/config/grobid.yaml:ro grobid/grobid:0.8.0
   ```

   Replace `/path/to/bb-ai-oceanography` with the actual path to your project directory.

2. In a separate terminal, run paperetl to extract content and create an SQLite database:

   ```
   python -m paperetl.file /path/to/pdfs /path/to/output
   ```

   Replace `/path/to/pdfs` with the directory containing your downloaded PDFs, and `/path/to/output` with the desired output directory for the SQLite database.

### Generating Reports with paperai

1. Index the extracted content:

   ```
   python -m paperai.index /path/to/output
   ```
   
   Use the same `/path/to/output` as in the paperetl step.



2. Generate a report using a YAML configuration file:

   ```
   python -m paperai.report /path/to/report_config.yml 100 md /path/to/output
   ```

   Replace:
   -  `/path/to/report_config.yml` with the path to your report configuration file (examples can be found in the `paperai/reports` directory).
   - `/path/to/output` with the same `/path/to/output` as in the paperetl step.

One of the major improvements to `paperai`'s report generation made in this repository is the addition of a properly cited summary section. This is done by using an open-source LLM (best results so far have been with [Gemma](https://github.com/gemma-ai/gemma)) to generate the summary given a query. Here is an example of what the summary looks like given the query "emerging trends or future directions in machine learning for ocean sciences":

> Machine learning (ML) is rapidly becoming a valuable tool in ocean sciences, offering solutions to complex problems that traditional methods struggle with. ML's ability to handle vast datasets, identify patterns, and make predictions is particularly suited for the oceanographic realm, where data is increasingly abundant and complex.  
> 
> "Machine learning uses dynamic models to make data-driven choices, and ML approaches may be used to high-dimensional, complicated, non-linear, and big-data problems." ([Sunkara et al., 2023](https://doi.org/10.3389/fmars.2023.1075822))  It is proving effective in addressing issues like ocean acidification, sea-level rise, and the impacts of climate change on marine ecosystems.  
> 
> The Southern Ocean, known for its intricate circulation patterns, is a prime example of where ML is making a significant impact. "Recently, machine learning methods are being used to fuel progress within prediction, numerical modelling and beyond, speeding up and refining existing tasks (see review in refs.18,19)." ([Sonnewald et al., 2023](https://doi.org/10.1038/s43247-023-00793-7))  
> 
> ML's strength lies in its ability to classify complex features in oceanic systems. "Machine learning is a tool that can be leveraged to classify dynamic and complex features in oceanic systems(Jones et al., 2014;Gangopadhyay et al., 2015)." ([Phillips et al., 2020](https://doi.org/10.3389/fmars.2020.00365)) This is particularly valuable for tasks like identifying ocean currents, predicting weather patterns, and understanding the distribution of marine species.
> 
> While ML shows immense promise, researchers acknowledge the need for further development and application. "The move from multi-spectral to hyperspectral remote sensing has resulted in the availability of a larger amount of information and more degrees of freedom for improving quantification of PIC. This increase in information can make it challenging for a human to find patterns and analyze the data, but opens the door to machine learning approaches." ([Balch et al., 2023](https://doi.org/10.1016/j.earscirev.2023.104363)) 
> 
> "In the future, with the explosive growth of marine big data, making efficient use of oceanic big data is still an important research direction. We will continue to explore faster, more accurate and more complex prediction models for ocean hydrological data with a combination of in-depth learning and integrated learning." ([Yang et al., 2019](https://doi.org/10.3390/s19071562)) As technology advances and datasets grow, ML will undoubtedly play an increasingly crucial role in unraveling the mysteries of the ocean and ensuring its sustainable future.

## Project Structure

- `scripts/`: Contains the main scripts for downloading papers
- `utils/`: Utility functions for API calls, file handling, and PDF processing
- `notebooks/`: Jupyter notebooks for data analysis and visualization
- `config/`: Configuration files, including the custom GROBID configuration
- `paperetl/`: The modified paperetl package for extracting content from PDFs
- `paperai/`: The modified paperai package for analyzing and generating reports

## Acknowledgments

- This work builds upon the initial analysis by Leland McInnes at the Tutte Institute, and Taylor Denouden's [bb-ai-oceanography](https://github.com/taylordenouden/bb-ai-oceanography) project
- Thanks to all the publishers and platforms providing access to scientific literature
- [GROBID](https://github.com/kermitt2/grobid) for PDF content extraction
- [paperetl](https://github.com/neuml/paperetl) and [paperai](https://github.com/neuml/paperai) for content processing and analysis
