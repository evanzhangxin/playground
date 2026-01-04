import requests
from bs4 import BeautifulSoup
import os
from urllib.parse import urlparse

class WebCrawler:
    def __init__(self, storage_dir="data"):
        self.storage_dir = storage_dir
        if not os.path.exists(storage_dir):
            os.makedirs(storage_dir)

    def fetch_page(self, url):
        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            }
            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()
            return response.text
        except Exception as e:
            print(f"Error fetching {url}: {e}")
            return None

    def parse_content(self, html_content):
        if not html_content:
            return ""
        
        soup = BeautifulSoup(html_content, 'html.parser')
        
        # Remove script and style elements
        for script in soup(["script", "style", "nav", "footer", "header"]):
            script.decompose()
            
        # Get text
        text = soup.get_text()
        
        # Break into lines and remove leading/trailing space on each
        lines = (line.strip() for line in text.splitlines())
        # Break multi-headlines into a line each
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        # Drop blank lines
        text = '\n'.join(chunk for chunk in chunks if chunk)
        
        return text

    def save_content(self, url, text):
        if not text:
            return None
            
        parsed_url = urlparse(url)
        filename = f"{parsed_url.netloc}{parsed_url.path}".replace("/", "_").replace(".", "_")
        if not filename.endswith(".txt"):
            filename += ".txt"
        
        # Limit filename length
        if len(filename) > 200:
            filename = filename[:200]
            
        filepath = os.path.join(self.storage_dir, filename)
        
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(f"Source: {url}\n\n")
            f.write(text)
            
        return filepath

    def crawl(self, url):
        print(f"Crawling {url}...")
        html = self.fetch_page(url)
        text = self.parse_content(html)
        if text:
            saved_path = self.save_content(url, text)
            print(f"Saved content to {saved_path}")
            return text, saved_path
        return None, None

if __name__ == "__main__":
    # Test
    crawler = WebCrawler()
    crawler.crawl("https://example.com")
