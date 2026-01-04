#!/usr/bin/env python3
"""
Test cases for RAG (Retrieval-Augmented Generation) query functionality
"""
import os
import sys
# Add parent directory to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import logging
import tempfile
from rag import KnowledgeBase

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_knowledge_base_initialization():
    """Test KnowledgeBase initialization"""
    logger.info("=== Testing KnowledgeBase Initialization ===")
    
    # Test with default persist directory
    with tempfile.TemporaryDirectory() as temp_dir:
        # Set custom CHROMA_DB_DIR for this test
        import config
        original_chroma_dir = config.CHROMA_DB_DIR
        config.CHROMA_DB_DIR = temp_dir
        
        try:
            kb = KnowledgeBase()
            assert kb.vector_db is not None, "Vector DB not initialized"
            assert kb.embedding_function is not None, "Embedding function not initialized"
            
            logger.info("✅ KnowledgeBase initialization test passed")
            return True
        finally:
            # Restore original config
            config.CHROMA_DB_DIR = original_chroma_dir

def test_add_document():
    """Test adding documents to knowledge base"""
    logger.info("=== Testing add_document Method ===")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create a sample document
        sample_text = """
        This is a test document about AI and machine learning.
        
        AI stands for Artificial Intelligence, which is the simulation of human intelligence in machines.
        Machine learning is a subset of AI that enables systems to learn from data.
        
        There are different types of machine learning:
        - Supervised learning
        - Unsupervised learning
        - Reinforcement learning
        
        Neural networks are a key component of modern AI systems.
        They are inspired by the human brain's structure and function.
        """
        
        # Create temporary text file
        with open(os.path.join(temp_dir, "test_doc.txt"), "w", encoding="utf-8") as f:
            f.write(sample_text)
        
        # Set custom CHROMA_DB_DIR for this test
        import config
        original_chroma_dir = config.CHROMA_DB_DIR
        config.CHROMA_DB_DIR = os.path.join(temp_dir, "chroma_test")
        
        try:
            kb = KnowledgeBase()
            
            # Add document to knowledge base
            num_chunks = kb.add_document(os.path.join(temp_dir, "test_doc.txt"))
            
            logger.info(f"Added document with {num_chunks} chunks")
            
            assert num_chunks > 0, "No chunks added to knowledge base"
            
            logger.info("✅ add_document method test passed")
            return True
        finally:
            # Restore original config
            config.CHROMA_DB_DIR = original_chroma_dir

def test_query_functionality():
    """Test query functionality"""
    logger.info("=== Testing query Method ===")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create a sample document
        sample_text = """
        This is a test document about AI and machine learning.
        
        AI stands for Artificial Intelligence, which is the simulation of human intelligence in machines.
        Machine learning is a subset of AI that enables systems to learn from data.
        
        There are different types of machine learning:
        - Supervised learning
        - Unsupervised learning
        - Reinforcement learning
        
        Neural networks are a key component of modern AI systems.
        They are inspired by the human brain's structure and function.
        """
        
        # Create temporary text file
        with open(os.path.join(temp_dir, "test_doc.txt"), "w", encoding="utf-8") as f:
            f.write(sample_text)
        
        # Set custom CHROMA_DB_DIR for this test
        import config
        original_chroma_dir = config.CHROMA_DB_DIR
        config.CHROMA_DB_DIR = os.path.join(temp_dir, "chroma_test")
        
        try:
            kb = KnowledgeBase()
            kb.add_document(os.path.join(temp_dir, "test_doc.txt"))
            
            # Test with relevant query
            query = "What is AI?"
            response = kb.query(query)
            logger.info(f"Query: {query}")
            logger.info(f"Response: {response}")
            
            assert isinstance(response, str), "Response is not a string"
            assert len(response) > 0, "Empty response"
            
            # Test with machine learning query
            query = "What are the types of machine learning?"
            response = kb.query(query)
            logger.info(f"Query: {query}")
            logger.info(f"Response: {response}")
            
            assert isinstance(response, str), "Response is not a string"
            assert len(response) > 0, "Empty response"
            
            # Test with neural networks query
            query = "What are neural networks?"
            response = kb.query(query)
            logger.info(f"Query: {query}")
            logger.info(f"Response: {response}")
            
            assert isinstance(response, str), "Response is not a string"
            assert len(response) > 0, "Empty response"
            
            logger.info("✅ query method test passed")
            return True
        finally:
            # Restore original config
            config.CHROMA_DB_DIR = original_chroma_dir

def test_clear_knowledge_base():
    """Test clearing the knowledge base"""
    logger.info("=== Testing clear Method ===")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create a sample document
        sample_text = "This is a test document to be cleared."
        
        # Create temporary text file
        with open(os.path.join(temp_dir, "test_doc.txt"), "w", encoding="utf-8") as f:
            f.write(sample_text)
        
        # Set custom CHROMA_DB_DIR for this test
        import config
        original_chroma_dir = config.CHROMA_DB_DIR
        config.CHROMA_DB_DIR = os.path.join(temp_dir, "chroma_test")
        
        try:
            kb = KnowledgeBase()
            
            # Add document
            kb.add_document(os.path.join(temp_dir, "test_doc.txt"))
            
            # Clear knowledge base
            kb.clear()
            
            logger.info("✅ clear method test passed")
            return True
        finally:
            # Restore original config
            config.CHROMA_DB_DIR = original_chroma_dir

def test_qa_chain_creation():
    """Test QA chain creation"""
    logger.info("=== Testing QA Chain Creation ===")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        # Set custom CHROMA_DB_DIR for this test
        import config
        original_chroma_dir = config.CHROMA_DB_DIR
        config.CHROMA_DB_DIR = os.path.join(temp_dir, "chroma_test")
        
        try:
            kb = KnowledgeBase()
            
            # QA chain should be created if API keys are available
            has_api_key = config.DASHSCOPE_API_KEY or config.OPENAI_API_KEY
            
            if has_api_key:
                assert kb.qa_chain is not None, "QA chain not created despite having API key"
                logger.info("QA chain successfully created with API key")
            else:
                logger.info("No API keys available, skipping QA chain creation check")
            
            logger.info("✅ QA chain creation test passed")
            return True
        finally:
            # Restore original config
            config.CHROMA_DB_DIR = original_chroma_dir

if __name__ == "__main__":
    """Run all RAG query tests"""
    tests = [
        test_knowledge_base_initialization,
        test_add_document,
        test_query_functionality,
        test_clear_knowledge_base,
        test_qa_chain_creation
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            if test():
                passed += 1
            else:
                failed += 1
        except Exception as e:
            logger.error(f"❌ Test {test.__name__} failed with exception: {str(e)}", exc_info=True)
            failed += 1
    
    logger.info(f"\n=== Test Summary ===")
    logger.info(f"Total tests: {len(tests)}")
    logger.info(f"Passed: {passed}")
    logger.info(f"Failed: {failed}")
    
    if failed == 0:
        logger.info("✅ All RAG query tests passed!")
    else:
        logger.error(f"❌ {failed} test(s) failed!")
