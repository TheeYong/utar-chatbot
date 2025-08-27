import os
import requests
import certifi
import urllib3
from urllib.parse import urljoin
from playwright.sync_api import sync_playwright
from bs4 import BeautifulSoup

# Disable only insecure request warnings for UTAR's SSL issue
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)


def scrape_pdfs(urls, department, base_folder="./data"):
    """
    Scrapes a webpage for all linked PDFs and downloads them into a department folder.
    Handles UTAR's broken SSL for PDFs specifically.
    """
    download_folder = os.path.join(base_folder, department)
    os.makedirs(download_folder, exist_ok=True)

    pdf_files = []

    # Use Playwright to fetch rendered HTML
    with sync_playwright() as p:
        browser = p.chromium.launch()

        for url in urls:
            page = browser.new_page()
            page.goto(url)
            page.wait_for_timeout(3000)  # wait for JS to load
            html_content = page.content()
            page.close()


            soup = BeautifulSoup(html_content, 'html.parser')

            # Download PDFs only
            for link in soup.find_all('a', href=True):
                full_url = urljoin(url, link['href'])
                if full_url.lower().endswith(".pdf"):
                    file_name = os.path.basename(full_url)
                    pdf_path = os.path.join(download_folder, file_name)

                    if os.path.exists(pdf_path):
                        print(f"Skipping (already exists): {file_name}")
                        continue

                    try:
                        if "utar.edu.my" in full_url.lower():
                            # Bypass SSL verification for UTAR
                            r = requests.get(full_url, stream=True, verify=False, timeout=10)
                        else:
                            r = requests.get(full_url, stream=True, verify=certifi.where(), timeout=10)
                        r.raise_for_status()
                        with open(pdf_path, "wb") as f:
                            for chunk in r.iter_content(8192):
                                f.write(chunk)
                        pdf_files.append(pdf_path)
                        print(f"Downloaded PDF: {pdf_path}")
                    except Exception as e:
                        print(f"Failed to download {full_url}: {e}")
                        
        browser.close()

    return pdf_files


if __name__ == "__main__":
    department = "Department of Student Affairs"
    urls = ["https://dsa.kpr.utar.edu.my/Vehicle-Stickers.php", "https://dsa.kpr.utar.edu.my/Bus-Services.php"]

    pdf_files = scrape_pdfs(urls, department)

    print("\nPDF files saved:")
    for pf in pdf_files:
        print(f"  - {pf}")
