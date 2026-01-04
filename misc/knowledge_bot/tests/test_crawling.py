#!/usr/bin/env python3
"""
Test script to demonstrate the web crawling and RAG pipeline
"""
import os
import sys
# Add parent directory to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import logging
from crawler import WebCrawler
from rag import KnowledgeBase
import config
# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_crawling():
    """Test the web crawling process"""
    logger.info("=== Starting Crawling Test ===")
    
    # 1. Initialize crawler
    crawler = WebCrawler()
    
    # 2. Test URL to crawl
    test_url = "https://example.com"
    
    # 3. Crawl the URL
    logger.info("Crawling URL: %s", test_url)
    text, filepath = crawler.crawl(test_url)
    
    if text:
        logger.info("✅ Crawling successful!")
        logger.info("📄 Saved to: %s", filepath)
        logger.info("📊 Text length: %d characters", len(text))
        
        # Show sample content
        logger.info("\n📝 Sample content:")
        sample = text[:500] + "..." if len(text) > 500 else text
        logger.info(sample)
        
        return filepath
    else:
        logger.error("❌ Crawling failed!")
        return None

def test_rag_pipeline(filepath):
    """Test the RAG pipeline"""
    logger.info("\n=== Starting RAG Pipeline Test ===")
    
    # 1. Initialize knowledge base
    kb = KnowledgeBase()
    
    # 2. Add document to knowledge base
    logger.info("Adding document to knowledge base: %s", filepath)
    num_chunks = kb.add_document(filepath)
    logger.info("✅ Added %d chunks to knowledge base", num_chunks)
    
    # 3. Test query
    test_query = "What is the website about?"
    logger.info("\n🔍 Testing query: %s", test_query)
    response = kb.query(test_query)
    logger.info("💬 Response: %s", response)
    
    logger.info("\n=== Test Completed Successfully ===")

if __name__ == "__main__":
    # Run the test
    filepath = test_crawling()
    if filepath:
        test_rag_pipeline(filepath)