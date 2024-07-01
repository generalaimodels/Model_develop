import os
import argparse
from typing import List
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin
from tqdm import tqdm

def download_pdfs(base_url: str, num_pages: int, output_dir: str) -> None:
    """
    Download PDF files from the specified URL and save them in the output directory.

    Args:
        base_url (str): The base URL of the website.
        num_pages (int): The total number of pages to scrape.
        output_dir (str): The directory to save the downloaded PDF files.
    """
    os.makedirs(output_dir, exist_ok=True)

    for page in tqdm(range(1, num_pages + 1), desc="Downloading PDFs"):
        url = f"{base_url}?category=natural-language-processing&page={page}&"
        response = requests.get(url)
        soup = BeautifulSoup(response.text, "html.parser")

        pdf_links = extract_pdf_links(soup, base_url)
        download_and_save_pdfs(pdf_links, output_dir)

def extract_pdf_links(soup: BeautifulSoup, base_url: str) -> List[str]:
    """
    Extract PDF links from the parsed HTML.

    Args:
        soup (BeautifulSoup): The parsed HTML.
        base_url (str): The base URL of the website.

    Returns:
        List[str]: A list of PDF links.
    """
    pdf_links = []
    for link in soup.find_all("a"):
        href = link.get("href")
        if href and href.endswith(".pdf"):
            pdf_links.append(urljoin(base_url, href))
    return pdf_links

def download_and_save_pdfs(pdf_links: List[str], output_dir: str) -> None:
    """
    Download and save PDF files from the given links.

    Args:
        pdf_links (List[str]): A list of PDF links.
        output_dir (str): The directory to save the downloaded PDF files.
    """
    for link in pdf_links:
        response = requests.get(link)
        filename = os.path.join(output_dir, link.split("/")[-1])
        with open(filename, "wb") as file:
            file.write(response.content)

    
download_pdfs(
    base_url="https://research.google/pubs/",
    num_pages=72,
    output_dir="E:/LLMS/hemanth/Hemanth/google_pdf/"
)



