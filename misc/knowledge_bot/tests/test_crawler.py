#!/usr/bin/env python3
"""
Test cases for Web Crawler functionality
"""
import os
import sys
# Add parent directory to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import logging
import tempfile
from crawler import WebCrawler

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_crawler_initialization():
    """Test WebCrawler initialization"""
    logger.info("=== Testing Crawler Initialization ===")
    
    # Test with default storage directory
    crawler = WebCrawler()
    assert crawler.storage_dir == "data", f"Expected storage_dir to be 'data', got '{crawler.storage_dir}'"
    assert os.path.exists(crawler.storage_dir), f"Storage directory '{crawler.storage_dir}' does not exist"
    
    # Test with custom storage directory
    with tempfile.TemporaryDirectory() as temp_dir:
        crawler = WebCrawler(storage_dir=temp_dir)
        assert crawler.storage_dir == temp_dir, f"Expected storage_dir to be '{temp_dir}', got '{crawler.storage_dir}'"
        assert os.path.exists(crawler.storage_dir), f"Storage directory '{crawler.storage_dir}' does not exist"
    
    logger.info("✅ Crawler initialization test passed")

def test_fetch_page():
    """Test fetch_page method"""
    logger.info("=== Testing fetch_page Method ===")
    
    crawler = WebCrawler()
    
    # Test with a valid URL
    html = crawler.fetch_page("https://example.com")
    assert html is not None, "Failed to fetch page from https://example.com"
    assert len(html) > 0, "Fetched page is empty"
    assert "<html" in html.lower(), "Fetched content is not HTML"
    
    # Test with an invalid URL
    html = crawler.fetch_page("https://invalid-url-that-does-not-exist-12345.com")
    assert html is None, "Expected None for invalid URL, got HTML content"
    
    logger.info("✅ fetch_page method test passed")

def test_parse_content():
    """Test parse_content method"""
    logger.info("=== Testing parse_content Method ===")
    
    crawler = WebCrawler()
    
    # Test with sample HTML content
    sample_html = """
    <html>
        <head><title>Test Page</title></head>
        <body>
            <h1>Hello World</h1>
            <p>This is a test paragraph.</p>
            <script>alert('This is a script');</script>
            <style>body { background: red; }</style>
        </body>
    </html>
    """
    
    text = crawler.parse_content(sample_html)
    assert text is not None, "Failed to parse HTML content"
    assert len(text) > 0, "Parsed text is empty"
    assert "Hello World" in text, "Expected 'Hello World' in parsed text"
    assert "This is a test paragraph" in text, "Expected paragraph text in parsed content"
    assert "alert('This is a script')" not in text, "Expected script content to be removed"
    assert "body { background: red; }" not in text, "Expected style content to be removed"
    
    # Test with empty HTML
    text = crawler.parse_content("")
    assert text == "", "Expected empty string for empty HTML"
    
    logger.info("✅ parse_content method test passed")

def test_save_content():
    """Test save_content method"""
    logger.info("=== Testing save_content Method ===")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        crawler = WebCrawler(storage_dir=temp_dir)
        
        # Test saving content
        url = "https://example.com"
        text = "Test content for example.com"
        saved_path = crawler.save_content(url, text)
        
        assert saved_path is not None, "Failed to save content"
        assert os.path.exists(saved_path), f"Saved file '{saved_path}' does not exist"
        
        # Read the saved file and verify content
        with open(saved_path, "r", encoding="utf-8") as f:
            content = f.read()
        
        assert f"Source: {url}" in content, f"Expected 'Source: {url}' in saved file"
        assert text in content, f"Expected '{text}' in saved file"
        
        # Test saving empty content
        saved_path = crawler.save_content(url, "")
        assert saved_path is None, "Expected None for empty content"
    
    logger.info("✅ save_content method test passed")

def test_crawl():
    """Test crawl method"""
    logger.info("=== Testing crawl Method ===")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        crawler = WebCrawler(storage_dir=temp_dir)
        
        # Test crawling a valid URL
        text, filepath = crawler.crawl("https://example.com")
        
        assert text is not None, "Failed to crawl https://example.com"
        assert filepath is not None, "Failed to get filepath after crawling"
        assert os.path.exists(filepath), f"Crawled file '{filepath}' does not exist"
        assert len(text) > 0, "Crawled text is empty"
        
        # Test crawling an invalid URL
        text, filepath = crawler.crawl("https://invalid-url-that-does-not-exist-12345.com")
        assert text is None, "Expected None for text when crawling invalid URL"
        assert filepath is None, "Expected None for filepath when crawling invalid URL"
    
    logger.info("✅ crawl method test passed")

if __name__ == "__main__":
    # Run all tests
    test_crawler_initialization()
    test_fetch_page()
    test_parse_content()
    test_save_content()
    test_crawl()
    
    logger.info("\n=== All Crawler Tests Passed ===")