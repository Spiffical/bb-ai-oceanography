from utils.download_utils import find_and_download_paper
from utils.pdf_utils import standardize_title
import pandas as pd
import os
import argparse
import multiprocessing
from multiprocessing import Value, Lock
from concurrent.futures import ProcessPoolExecutor, as_completed

# Function to update the progress
def update_progress(progress, total_papers):
    print(f"Downloaded {progress} of {total_papers} papers.")

def download_paper_task(title, download_path):
    standardized_title = standardize_title(title)
    filename = standardized_title + '.pdf'
    full_path = os.path.join(download_path, filename)
    if os.path.exists(full_path):
        return title, True, "Already downloaded"
    success, reason = find_and_download_paper(title, full_path)
    return title, success, reason

def worker(task_queue, result_queue, download_path):
    while True:
        title = task_queue.get()
        if title is None:
            break
        result = download_paper_task(title, download_path)
        result_queue.put(result)

def terminate_processes(futures):
    for future in futures:
        if not future.done():
            process = future._process
            process.terminate()

def main():
    parser = argparse.ArgumentParser(description='Download PDFs for papers using DOI or pypaperbot.')
    parser.add_argument('--input_csv', type=str, default='./data/Search String H Results - Literature Reviews.csv', help='Path to the input CSV file')
    parser.add_argument('--download_path', type=str, default='data/pdfs/', help='Path to save the downloaded PDFs')
    args = parser.parse_args()

    input_csv = args.input_csv
    download_path = args.download_path

    if not os.path.exists(download_path):
        os.makedirs(download_path)

    # Read titles from CSV
    df = pd.read_csv(input_csv)

    # Filter for included papers
    included_papers = df[df['Include / Exclude'].str.lower() == 'included']

    paper_titles = included_papers['Title'].tolist()[:1000]

    # Filter out already downloaded papers
    already_downloaded = 0
    paper_titles_to_download = []
    for title in paper_titles:
        standardized_title = standardize_title(title)
        filename = standardized_title + '.pdf'
        full_path = os.path.join(download_path, filename)
        if os.path.exists(full_path):
            already_downloaded += 1
        else:
            paper_titles_to_download.append(title)

    total_papers_to_download = len(paper_titles_to_download)

    print(f"Total papers: {len(paper_titles)}")
    print(f"Papers already downloaded: {already_downloaded}")
    print(f"Papers to be downloaded: {total_papers_to_download}")

    # Create queues for communication between processes
    task_queue = multiprocessing.Queue()
    result_queue = multiprocessing.Queue()

    # Start worker processes
    num_workers = 4#multiprocessing.cpu_count()
    workers = []
    for _ in range(num_workers):
        p = multiprocessing.Process(target=worker, args=(task_queue, result_queue, download_path))
        p.start()
        workers.append(p)

    # Add tasks to the queue
    for title in paper_titles_to_download:
        task_queue.put(title)

    # Add stop signals for workers
    for _ in range(num_workers):
        task_queue.put(None)

    progress = 0
    with open('failed_downloads.txt', 'w') as failed_log:
        for _ in range(total_papers_to_download):
            title, success, reason = result_queue.get()
            if success:
                progress += 1
            else:
                failed_log.write(f"Failed to download '{title}': {reason}\n")
            update_progress(progress, total_papers_to_download)

    # Wait for all workers to finish
    for p in workers:
        p.join()

if __name__ == "__main__":
    main()

