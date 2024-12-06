# 🌊 bb-ai-oceanography: Full-Text Analysis of Machine Learning in Oceanography

This repository is an extended version of Taylor Denouden's [bb-ai-oceanography](https://github.com/tayden/bb-ai-oceanography/) project. While the original repository focused on analyzing paper *abstracts* using Large Language Models (LLMs), this fork expands the scope to include *full-text* analysis and fully-cited report generation of scientific papers related to machine learning applications in oceanography. Report generation could absolutely benefit from the work previously done to identify topic areas, e.g. the report queries could focus on specific topic areas.

## 📋 Table of Contents

1. [Project Overview](#project-overview)
2. [Key Features](#key-features)
3. [Getting Started](#getting-started)
   - [Prerequisites](#prerequisites)
   - [Installation and Setup Options](#installation-and-setup-options)
4. [Report Configuration](#report-configuration)
5. [Usage](#usage)
   - [Quick Start (Using Pre-processed Model)](#quick-start-using-pre-processed-model)
   - [Building from Scratch (Optional)](#building-from-scratch-optional)
6. [Project Structure](#project-structure)
7. [Acknowledgments](#acknowledgments)

## 🎯 Project Overview

The project aims to provide a comprehensive analysis of the current state of machine learning in oceanography by:

1. Downloading full-text scientific papers from various sources
2. Processing and extracting relevant information from the papers using paperetl
3. Analyzing the content using natural language processing techniques with paperai
4. Generating reports to identify trends and patterns in the field

## ⭐ Key Features

- Multi-source paper retrieval (Elsevier, Springer, Wiley, arXiv, Unpaywall)
- Full-text extraction from PDFs and XMLs using GROBID and paperetl
- Robust error handling and logging
- Rate-limited API calls to respect usage policies
- Machine learning-based analysis and report generation using paperai

Changes made to `paperetl`:
- Extracting paragraphs instead of sentences for improved summary quality
- Improved error handling

Changes made to `paperai`  :
- One of the major improvements to `paperai`'s report generation made in this repository is the addition of a properly cited summary section. This is done by using an LLM (either through the API or locally) to generate the summary given a query. See below for an example of what the summary looks like given the query "emerging trends or future directions in machine learning for ocean sciences":
<details><summary><b>Example Summary</b></summary>

Machine learning (ML) is rapidly becoming a valuable tool in ocean sciences, offering solutions to complex problems that traditional methods struggle with. ML's ability to handle vast datasets, identify patterns, and make predictions is particularly suited for the oceanographic realm, where data is increasingly abundant and complex.  

"Machine learning uses dynamic models to make data-driven choices, and ML approaches may be used to high-dimensional, complicated, non-linear, and big-data problems." ([Sunkara et al., 2023](https://doi.org/10.3389/fmars.2023.1075822))  It is proving effective in addressing issues like ocean acidification, sea-level rise, and the impacts of climate change on marine ecosystems.  

The Southern Ocean, known for its intricate circulation patterns, is a prime example of where ML is making a significant impact. "Recently, machine learning methods are being used to fuel progress within prediction, numerical modelling and beyond, speeding up and refining existing tasks (see review in refs.18,19)." ([Sonnewald et al., 2023](https://doi.org/10.1038/s43247-023-00793-7))  

ML's strength lies in its ability to classify complex features in oceanic systems. "Machine learning is a tool that can be leveraged to classify dynamic and complex features in oceanic systems(Jones et al., 2014;Gangopadhyay et al., 2015)." ([Phillips et al., 2020](https://doi.org/10.3389/fmars.2020.00365)) This is particularly valuable for tasks like identifying ocean currents, predicting weather patterns, and understanding the distribution of marine species.

While ML shows immense promise, researchers acknowledge the need for further development and application. "The move from multi-spectral to hyperspectral remote sensing has resulted in the availability of a larger amount of information and more degrees of freedom for improving quantification of PIC. This increase in information can make it challenging for a human to find patterns and analyze the data, but opens the door to machine learning approaches." ([Balch et al., 2023](https://doi.org/10.1016/j.earscirev.2023.104363)) 

"In the future, with the explosive growth of marine big data, making efficient use of oceanic big data is still an important research direction. We will continue to explore faster, more accurate and more complex prediction models for ocean hydrological data with a combination of in-depth learning and integrated learning." ([Yang et al., 2019](https://doi.org/10.3390/s19071562)) As technology advances and datasets grow, ML will undoubtedly play an increasingly crucial role in unraveling the mysteries of the ocean and ensuring its sustainable future.
</details>

## 🚀 Getting Started

### Prerequisites

- Python 3.7+
- Docker (recommended)
- NVIDIA GPU (optional, for local models)
- One of the following:
  - OpenAI API key (recommended)
  - Hugging Face account (for API or local models)
  - Local GPU for running models (optional)

### Installation and Setup Options

There are three main ways to use this project, listed in order of recommendation:

<details>
<summary><b>Option 1: Docker with OpenAI API (Recommended)</b></summary>

1. Clone the repository:
   ```bash
   git clone https://github.com/Spiffical/bb-ai-oceanography.git
   cd bb-ai-oceanography
   ```

2. Create a `.env` file in the project root with your OpenAI API key:
   ```plaintext
   OPENAI_API_KEY=your_openai_api_key
   ```

3. Build the Docker image:
   
   Ensure Docker is installed and running, then run:
   ```bash
   docker build -f docker/Dockerfile.api -t paperai-api .
   ```
</details>

<details>
<summary><b>Option 2: Docker with Local Models</b></summary>

Choose this option if you want to run models locally without API costs.

1. Clone and enter the repository as shown above

2. Choose your preferred local model provider:

   **A. Using Ollama (Easier)**
   1. Build the Docker image:
      ```bash
      docker build -f docker/Dockerfile.gpu -t paperai-gpu .
      ```
   
   **B. Using Hugging Face (More flexible)**
   1. Create a Hugging Face account and get your access token
   2. Add to your `.env` file:
      ```plaintext
      HUGGING_FACE_HUB_TOKEN=your_token
      ```
   3. Build the Docker image:
      ```bash
      docker build -f docker/Dockerfile.gpu -t paperai-gpu .
      ```
</details>

<details>
<summary><b>Option 3: Local Installation</b></summary>

1. Clone the repository:
   ```bash
   git clone https://github.com/Spiffical/bb-ai-oceanography.git
   cd bb-ai-oceanography
   ```

2. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

3. Install paperetl and paperai:
   ```bash
   cd paperetl
   pip install .
   cd ../paperai
   pip install .
   cd ..
   ```

4. Set up environment variables if you are downloading papers from the Elsevier, Springer, and Wiley APIs:
   Create a `.env` file in the project root and add your API keys and email address for Unpaywall:
   ```plaintext
   ELSEVIER_API_KEY=your_elsevier_api_key
   SPRINGER_API_KEY=your_springer_api_key
   WILEY_API_KEY=your_wiley_api_key
   UNPAYWALL_EMAIL=your_email_address
   ```
</details>

### ⚙️ Report Configuration

Reports are configured through YAML files in the `reports/` directory. Each configuration file defines how the report should be generated and what content to analyze.

Here's an example of a report configuration file:

```yaml
name: Your_Report_Name

options:
  # General settings
  topn: 100
  render: md
  qa: "deepset/roberta-base-squad2"
  generate_summary: true
  
  # Choose your mode
  llm_mode: "api"  # "api" or "local"
  
  # API Settings (if llm_mode is "api")
  api:
    provider: "openai"  # "openai" or "huggingface"
    model: "gpt-4o-mini"  # or other API models
  
  # Local Settings (if llm_mode is "local")
  local:
    provider: "ollama"  # "ollama" or "huggingface"
    model: "mistral:instruct"  # See supported models below
    gpu_strategy: "auto"  # For HuggingFace models

sections:
  Your_Section_Name:
    query: your search query here
    columns:
      - name: Date
      - name: Study
      - {name: Custom_Column, query: specific search terms, question: what specific information to extract}
```

Key components:
- `name`: The name of the report
- `options`: General settings for the report
  - `topn`: Number of results (paragraphs from the database) to use
  - `render`: Output format (md for markdown)
  - `qa`: Model to use for question answering
  - `generate_summary`: Whether to include an LLM-generated summary
  - `llm_mode`: Choose between API or local models
  - Mode-specific settings for API or local model usage
- `sections`: Define what content to analyze
  - `query`: Main search query for this section
  - `columns`: What information to extract and how to organize it
    - Simple columns just need a name
    - Complex columns can include specific queries and questions

Supported Models:
- OpenAI API: gpt-4o-mini, gpt-4-turbo, gpt-3.5-turbo, etc. (see [OpenAI API Models](https://platform.openai.com/docs/models))
- Ollama: mistral:instruct, gemma:7b, llama2:7b, etc. (see [Ollama Models](https://ollama.com/library?sort=newest))
- Hugging Face: google/gemma-2-9b-it, mistralai/Mistral-7B-Instruct-v0.2, etc. (see [Hugging Face Models](https://huggingface.co/models))

Example sections from the default report include:
- ML Applications in ocean sciences
- Research gaps and challenges
- Emerging trends

See the `reports/` directory for complete examples.

## 📖 Usage

### Quick Start (Using Pre-processed Model)

To generate a report using a pre-processed paperetl embeddings model:

<details>
<summary><b>Using OpenAI API</b></summary>

```bash
# On Linux/Mac
docker run --rm --env-file ".env" -v "$(pwd):/work" paperai-api -m paperai.report /work/reports/report_file.yml /work/path/to/your/model

# On Windows PowerShell
docker run --rm --env-file ".env" -v "${PWD}:/work" paperai-api -m paperai.report /work/reports/report_file.yml /work/path/to/your/model
```
</details>

<details>
<summary><b>Using Local Models with GPU</b></summary>

```bash
# On Linux/Mac
docker run --rm --env-file ".env" --gpus all -v "$(pwd):/work" paperai-gpu -m paperai.report /work/reports/report_file.yml /work/path/to/your/model

# On Windows PowerShell
docker run --rm --env-file ".env" --gpus all -v "${PWD}:/work" paperai-gpu -m paperai.report /work/reports/report_file.yml /work/path/to/your/model
```
</details>

<details>
<summary><b>Using Local Installation</b></summary>

```bash
python -m paperai.report reports/report_file.yml path/to/your/model
```
</details>

Replace:
- `path/to/your/model` with the path to your embeddings model directory
- `reports/report_file.yml` with the path to your report configuration file

For example, if you want to use the OpenAI API, your embeddings model is in `paperetl/models/pdf-oceanai`, your report configuration file is in `reports/report_oceans_gaps.yml`, and you are currently in the `bb-ai-oceanography` directory:
```bash
docker run --rm -v "$(pwd):/work" paperai-api python -m paperai.report /work/reports/report_oceans_gaps.yml /work/paperetl/models/pdf-oceanai
```

### Building from Scratch (Optional)

If you need to process new papers or build the database from scratch, follow these additional steps:

1. Download papers:

```bash
python scripts/download_papers.py --csv path/to/your/doi_list.csv output_path
```

Or for a single DOI:

```bash
python scripts/download_papers.py --doi 10.1016/j.example.2023.123456 output_path
```

2. Start GROBID with the custom configuration:

```bash
sudo docker run --rm --gpus all --init --ulimit core=0 -p 8070:8070 -v /path/to/bb-ai-oceanography/config/grobid.yaml:/opt/grobid/grobid-home/config/grobid.yaml:ro grobid/grobid:0.8.0
```

Replace `/path/to/bb-ai-oceanography` with the actual path to your project directory.

3. In a separate terminal, run paperetl to extract content and create an SQLite database:

```bash
python -m paperetl.file /path/to/pdfs /path/to/output
```

Replace `/path/to/pdfs` with the directory containing your downloaded PDFs, and `/path/to/output` with the desired output directory for the SQLite database.

4. Index the extracted content:

```bash
python -m paperai.index /path/to/output
```

Use the same `/path/to/output` as in the paperetl step.

5. Generate a report using a YAML configuration file (see [Report Configuration](#report-configuration) for details):

```bash
python -m paperai.report /path/to/report_config.yml 100 md /path/to/output
```

Replace:
- `/path/to/report_config.yml` with the path to your report configuration file (examples can be found in the `reports` directory).
- `/path/to/output` with the same `/path/to/output` as in the paperetl step.


## 📁 Project Structure

- `scripts/`: Contains the main scripts for downloading papers
- `utils/`: Utility functions for API calls, file handling, and PDF processing
- `notebooks/`: Jupyter notebooks for data analysis and visualization
- `config/`: Configuration files, including the custom GROBID configuration
- `paperetl/`: The modified paperetl package for extracting content from PDFs
- `paperai/`: The modified paperai package for analyzing and generating reports
- `docker/`: Docker configuration files for building and running the project
- `reports/`: Report configuration files for generating reports

## 🙏 Acknowledgments

- This work builds upon the initial analysis by Leland McInnes at the Tutte Institute, and Taylor Denouden's [bb-ai-oceanography](https://github.com/tayden/bb-ai-oceanography) project
- Thanks to all the publishers and platforms providing access to scientific literature
- [GROBID](https://github.com/kermitt2/grobid) for PDF content extraction
- [paperetl](https://github.com/neuml/paperetl) and [paperai](https://github.com/neuml/paperai) for content processing and analysis
