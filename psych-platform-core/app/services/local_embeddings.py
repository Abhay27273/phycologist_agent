"""
Local Embedding Service using sentence-transformers.
Production-ready, free, and secure for CodeQL/SBOM compliance.
"""
import logging
from typing import List
from sentence_transformers import SentenceTransformer
from langchain_core.embeddings import Embeddings

logger = logging.getLogger(__name__)

class LocalEmbeddings(Embeddings):
    """
    LangChain-compatible wrapper for sentence-transformers.
    Uses BAAI/bge-small-en-v1.5 model for high-quality embeddings.
    
    Benefits:
    - No API costs
    - No rate limits
    - SBOM/CodeQL compliant
    - Fast local inference
    """
    
    def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5"):
        """
        Initialize the local embedding model.
        
        Args:
            model_name: HuggingFace model identifier
        """
        logger.info(f"Loading local embedding model: {model_name}")
        self.model = model_name
        self.client = SentenceTransformer(model_name)
        logger.info("Local embedding model loaded successfully")
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        Embed a list of documents.
        
        Args:
            texts: List of text strings to embed
            
        Returns:
            List of embedding vectors
        """
        embeddings = self.client.encode(
            texts,
            normalize_embeddings=True,
            show_progress_bar=False
        )
        return embeddings.tolist()
    
    def embed_query(self, text: str) -> List[float]:
        """
        Embed a single query text.
        
        Args:
            text: Query string to embed
            
        Returns:
            Embedding vector
        """
        embedding = self.client.encode(
            [text],
            normalize_embeddings=True,
            show_progress_bar=False
        )
        return embedding[0].tolist()
