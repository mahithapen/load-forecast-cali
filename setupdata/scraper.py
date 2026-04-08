import os
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin

# Configuration
BASE_URL = "https://www.caiso.com/library/historical-ems-hourly-load"
DOWNLOAD_DIR = "caiso_load_data"


def scrape_caiso_load_data():
    if not os.path.exists(DOWNLOAD_DIR):
        os.makedirs(DOWNLOAD_DIR)
        print(f"Created directory: {DOWNLOAD_DIR}")

    print(f"Fetching page: {BASE_URL}...")
    response = requests.get(BASE_URL)
    if response.status_code != 200:
        print("Failed to retrieve the webpage.")
        return

    soup = BeautifulSoup(response.text, 'html.parser')

    links = soup.find_all('a', href=True)
    download_count = 0

    for link in links:
        href = link['href']
        if href.endswith('.xlsx') and 'historical' in href.lower() and 'load' in href.lower():
            file_url = urljoin(BASE_URL, href)
            file_name = os.path.join(DOWNLOAD_DIR, href.split('/')[-1])

            print(f"Downloading: {file_name}...")
            file_data = requests.get(file_url)

            with open(file_name, 'wb') as f:
                f.write(file_data.content)
            download_count += 1

    print(f"\nSuccess! Downloaded {download_count} files to '{DOWNLOAD_DIR}'.")


if __name__ == "__main__":
    scrape_caiso_load_data()
