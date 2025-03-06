# 🌊 bb-ai-oceanography: Full-Text Analysis of Machine Learning in Oceanography

This repository is an extended version of Taylor Denouden's [bb-ai-oceanography](https://github.com/tayden/bb-ai-oceanography/) project. While the original repository focused on analyzing paper *abstracts* using Large Language Models (LLMs), this fork expands the scope to include *full-text* analysis and fully-cited report generation of scientific papers related to machine learning applications in oceanography. Report generation could absolutely benefit from the work previously done by Taylor to identify topic areas, e.g. the report queries could focus on specific topic areas identified in the abstract analysis.

## 📋 Table of Contents

1. [Project Overview](#-project-overview)
2. [Key Features](#-key-features)
3. [Getting Started](#-getting-started)
   - [Prerequisites](#prerequisites)
   - [Installation and Setup Options](#installation-and-setup-options)
4. [Report Configuration](#-report-configuration)
5. [Usage](#-usage)
   - [Quick Start (Using Pre-processed Model)](#quick-start-using-pre-processed-model)
   - [Building from Scratch (Optional)](#building-from-scratch-optional)
6. [Project Structure](#-project-structure)
7. [Acknowledgments](#-acknowledgments)

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
- Added support for more XML formats

Changes made to `paperai`  :
- One of the major improvements to `paperai`'s report generation made in this repository is the addition of a properly cited summary section. This is done by using an LLM (either through the API or locally) to generate the summary given a query. See below for an example of what the summary looks like given the query "emerging trends or future directions in machine learning for ocean sciences":
<details><summary><b>Example Summary</b></summary>

Machine learning is increasingly becoming an essential tool in ocean sciences, offering unprecedented solutions for managing complex and high-dimensional data challenges. Its ability to efficiently tackle "high-dimensional, complicated, non-linear, and big-data problems" makes it particularly promising for oceanic applications ([Sunkara et al., 2023](https://doi.org/10.3389/fmars.2023.1075822)). This capability allows machine learning to address issues that traditional methods find difficult or unfeasible. For example, machine learning has been employed to enhance prediction models of Southern Ocean circulation patterns beyond the capabilities of conventional approaches ([Sonnewald et al., 2023](https://doi.org/10.1038/s43247-023-00793-7)). In marine ecology, these advanced tools are used for classifying dynamic oceanic features through various datasets like images and optical spectra ([Phillips et al., 2020](https://doi.org/10.3389/fmars.2020.00365))([Rubbens et al., 2023](https://doi.org/10.1093/icesjms/fsad100)), leading to significant advancements in understanding ecological systems by integrating diverse marine data sources.

The adaptation of machine learning also shows great promise in addressing critical issues such as climate change impacts on oceans. Despite its extensive global use in areas like climate analysis and ecological environments ([Sunkara et al., 2023](https://doi.org/10.3389/fmars.2023.1075822)), its application within specific regions like the Gulf of Mexico (GOM) remains limited even though there are abundant data resources available([Sunkara et al., 2023](https://doi.org/10.3389/fmars.2023.1075822))([Sunkara et al., 2023](https://doi.org/10.3389/fmars.2023.1075822)). Emerging technologies such as image-based machine learning methods offer transformative capabilities by accelerating crucial image processing tasks for marine research([Belcher et al., 2022](https://doi.org/10.1101/2022.12.24.521836)), but technical complexities still present substantial barriers to their adoption.

Furthermore, integrating physical models into machine learning algorithms can significantly improve predictions. This approach has already demonstrated potential by refining surface ocean pCO2 estimates when combined with outputs from global biogeochemical models([Gloege et al., 2021](https://doi.org/10.1002/essoar.10507164.1)). As computational power continues to increase alongside advances in sensor technology that enhance data collection across the world's oceans([Sunkara et al., 2023](https://doi.org/10.3389/fmars.2023.1075822)), future directions suggest more sophisticated simulations involving multiscale phenomena within coastal environments([Tang et al., 2021](https://doi.org/10.3390/jmse9080847))([Tang et al., 2021](https://doi.org/10.3390/jmse9080847)).

Overall, these trends underscore an exciting trajectory where interdisciplinary collaborations could lead toward intelligent autonomous systems capable of comprehensive ocean monitoring. Such advancements would not only benefit scientific exploration but also practical applications relevant to societal needs such as energy security or environmental sustainability initiatives surrounding our planet's vast aquatic ecosystems([Lermusiaux et al., 2017](https://doi.org/10.1357/002224017823524035))([Yang et al., 2019](https://doi.org/10.3390/s19071562)).
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

## ⚙️ Report Configuration

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
docker run --rm \
  -v "$(pwd):/work" \
  --env-file .env \
  paperai-api -m paperai.report /work/reports/report_oceans_gaps.yml /work/paperetl/models/pdf-oceanai
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

Replace `/path/to/bb-ai-oceanography` with the actual path to your project directory, e.g. if you're currently in the project directory, you can use $PWD:

```bash
sudo docker run --rm --gpus all --init --ulimit core=0 -p 8070:8070 -v $PWD/config/grobid.yaml:/opt/grobid/grobid-home/config/grobid.yaml:ro grobid/grobid:0.8.1
```

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
