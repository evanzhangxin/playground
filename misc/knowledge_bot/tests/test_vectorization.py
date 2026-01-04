#!/usr/bin/env python3
"""
Test cases for Vectorization functionality
"""
import os
import sys
# Add parent directory to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import logging
import tempfile
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_huggingface import HuggingFaceEmbeddings
import config

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_text_splitting():
    """Test text splitting functionality"""
    logger.info("=== Testing Text Splitting ===")
    
    # Create a sample text file
    sample_text = """
    This is a test document for text splitting. It contains multiple sentences and paragraphs.
    
    The document is split into chunks of 1000 characters with 200 characters overlap.
    This ensures that context is preserved between chunks.
    
    Testing is important to ensure that the splitting works correctly.
    The RecursiveCharacterTextSplitter should handle different types of text content.
    
    This test case will verify that the splitter creates the expected number of chunks.
    """
    
    # Create temporary text file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
        f.write(sample_text)
        temp_file = f.name
    
    try:
        # Load document
        loader = TextLoader(temp_file, encoding='utf-8')
        documents = loader.load()
        
        # Test with RecursiveCharacterTextSplitter
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=100,
            chunk_overlap=20
        )
        
        chunks = text_splitter.split_documents(documents)
        
        logger.info(f"Original document length: {len(documents[0].page_content)} characters")
        logger.info(f"Number of chunks created: {len(chunks)}")
        
        # Verify chunks
        assert len(chunks) > 0, "No chunks created"
        assert all(len(chunk.page_content) <= 100 for chunk in chunks), "Chunk size exceeds limit"
        
        # Check overlap between chunks
        if len(chunks) > 1:
            chunk1 = chunks[0].page_content
            chunk2 = chunks[1].page_content
            overlap = ""
            for i in range(min(20, len(chunk1))):
                if chunk2.startswith(chunk1[-i:]):
                    overlap = chunk1[-i:]
                    break
            logger.info(f"Overlap between first two chunks: '{overlap}'")
            assert len(overlap) > 0, "No overlap between chunks"
        
        logger.info("✅ Text splitting test passed")
        return True
    finally:
        # Clean up
        os.unlink(temp_file)

def test_embedding_creation():
    """Test embedding creation"""
    logger.info("=== Testing Embedding Creation ===")
    
    # Test embedding creation
    try:
        # Create embedding model instance
        embedding_model = HuggingFaceEmbeddings(
            model_name=config.EMBEDDING_MODEL_PATH,
            model_kwargs={"device": "cpu"}
        )
        
        # Test with sample text
        sample_texts = [
            "Hello world",
            "This is a test sentence",
            "Embedding generation test"
        ]
        
        embeddings = embedding_model.embed_documents(sample_texts)
        
        logger.info(f"Generated {len(embeddings)} embeddings")
        logger.info(f"Embedding dimension: {len(embeddings[0]) if embeddings else 0}")
        
        # Verify embeddings
        assert len(embeddings) == len(sample_texts), "Number of embeddings doesn't match number of texts"
        assert all(isinstance(emb, list) and len(emb) > 0 for emb in embeddings), "Invalid embedding format"
        
        # Test single embedding
        single_embedding = embedding_model.embed_query("Test query embedding")
        assert isinstance(single_embedding, list) and len(single_embedding) > 0, "Failed to generate single embedding"
        
        logger.info("✅ Embedding creation test passed")
        return True
    except Exception as e:
        logger.error(f"❌ Embedding test failed: {str(e)}", exc_info=True)
        return False

def test_vector_store_operations():
    """Test vector store operations"""
    logger.info("=== Testing Vector Store Operations ===")
    
    try:
        from langchain_community.vectorstores import Chroma
        from langchain_huggingface import HuggingFaceEmbeddings
        
        # Create temporary vector store
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create embedding model
            embedding_model = HuggingFaceEmbeddings(
                model_name=config.EMBEDDING_MODEL_PATH,
                model_kwargs={"device": "cpu"}
            )
            
            # Sample documents
            sample_docs = [
                "This is document 1 about AI",
                "This is document 2 about machine learning",
                "This is document 3 about langchain"
            ]
            
            # Create vector store
            vector_store = Chroma.from_texts(
                sample_docs,
                embedding_model,
                persist_directory=temp_dir
            )
            
            logger.info(f"Created vector store with {vector_store._collection.count()} documents")
            
            # Test similarity search
            query = "AI and machine learning"
            results = vector_store.similarity_search(query, k=2)
            
            logger.info(f"Similarity search for '{query}' returned {len(results)} results")
            for i, result in enumerate(results):
                logger.info(f"  Result {i+1}: {result.page_content}")
            
            assert len(results) == 2, f"Expected 2 results, got {len(results)}"
            
            # Test with scores
            results_with_scores = vector_store.similarity_search_with_score(query, k=2)
            for doc, score in results_with_scores:
                logger.info(f"  Result with score: '{doc.page_content}' (score: {score:.4f})")
            
            logger.info("✅ Vector store operations test passed")
            return True
    except Exception as e:
        logger.error(f"❌ Vector store test failed: {str(e)}", exc_info=True)
        return False

if __name__ == "__main__":
    """Run all vectorization tests"""
    tests = [
        test_text_splitting,
        test_embedding_creation,
        test_vector_store_operations
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
        logger.info("✅ All vectorization tests passed!")
    else:
        logger.error(f"❌ {failed} test(s) failed!")
