from playwright.sync_api import sync_playwright
from bs4 import BeautifulSoup
from urllib.parse import urljoin

def scrape_utar_page(urls):
    seen_links = set()
    results = []

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)

        for url in urls:
            page = browser.new_page()
            page.goto(url)  
            page.wait_for_timeout(3000)  # wait for JS to load
            html_content = page.content()
            page.close()

            soup = BeautifulSoup(html_content, 'html.parser')



            # ---- Collect Text ----
            page_text_parts = []
            for div in soup.find_all('div', class_='mg'):
                section_text = div.get_text(separator='\n', strip=True)
                if section_text:
                    page_text_parts.append(section_text)

            # ---- Collect Unique Links ----
            seen_links = set()
            link_texts = []
            link_metadata_list = []
            for link in soup.find_all('a', href=True):
                full_url = urljoin(url, link['href'])
                if full_url not in seen_links:  # ensure uniqueness
                    seen_links.add(full_url)
                    text = link.get_text(strip=True)
                    if text:
                        link_texts.append(text)
                        link_metadata_list.append({"text": text, "url": full_url})

            # ---- Decide How to Combine ----
            if not page_text_parts:  # if no main text, combine link texts
                combined_text = " | ".join(link_texts)
            else:
                combined_text = "\n".join(page_text_parts)
                if link_texts:
                    combined_text += "\nLinks: " + " | ".join(link_texts)

            # results.append({
            #     "text": combined_text,
            #     "metadata": {
            #         "source": url,
            #         # "links": link_metadata_list
            #     }
            # })
            results.append(
                Document(
                    page_content=combined_text,
                    metadata={'source': url}
                )
            )
            


        browser.close()

    return results

# Example usage
url_to_scrape = [
    "https://dsa.kpr.utar.edu.my/Vehicle-Stickers.php",
    "https://collaboration.utar.edu.my/"
]
result = scrape_utar_page(url_to_scrape)

for item in result:
    print(item["text"])
    print(item["metadata"])
