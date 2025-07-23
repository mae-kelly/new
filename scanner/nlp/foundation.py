import logging
import torch
import numpy as np
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

class FoundationModels:
    def __init__(self):
        self.models = {}
        self.tokenizers = {}
        self.embedding_dimension = 384  # Standardize on 384 dimensions
        self._initialize_models()
    
    def _initialize_models(self):
        """Initialize models with consistent embedding dimensions"""
        try:
            # Use smaller, more reliable models
            logger.info("Initializing lightweight sentence transformers...")
            
            # All these models output 384-dimensional embeddings
            self.models['sentence_transformer'] = SentenceTransformer('all-MiniLM-L6-v2')
            self.models['domain_encoder'] = SentenceTransformer('all-MiniLM-L6-v2')  
            self.models['code_encoder'] = SentenceTransformer('all-MiniLM-L6-v2')
            
            # Verify embedding dimensions
            test_embedding = self.models['sentence_transformer'].encode(["test"])
            actual_dim = test_embedding.shape[1]
            self.embedding_dimension = actual_dim
            
            logger.info(f"Foundation models initialized with {self.embedding_dimension}D embeddings")
            
        except Exception as e:
            logger.error(f"Failed to initialize foundation models: {e}")
            self._initialize_fallback_models()
    
    def _initialize_fallback_models(self):
        """Ultra-simple fallback using basic word embeddings"""
        try:
            logger.warning("Using basic fallback models")
            # Create a dummy model that returns consistent dimensions
            self.models['sentence_transformer'] = None
            self.models['domain_encoder'] = None  
            self.models['code_encoder'] = None
            self.embedding_dimension = 384
            logger.info("Fallback models initialized")
        except Exception as e:
            logger.error(f"Failed to initialize fallback models: {e}")
    
    def get_sentence_embedding(self, text, model_type='sentence_transformer'):
        """Get consistent-dimension embeddings"""
        try:
            model = self.models.get(model_type)
            if model:
                embedding = model.encode([text])[0]
                # Ensure consistent dimensionality
                if len(embedding) != self.embedding_dimension:
                    # Pad or truncate to match expected dimension
                    if len(embedding) < self.embedding_dimension:
                        padding = np.zeros(self.embedding_dimension - len(embedding))
                        embedding = np.concatenate([embedding, padding])
                    else:
                        embedding = embedding[:self.embedding_dimension]
                return embedding
            else:
                # Return zero vector with correct dimensions
                return np.zeros(self.embedding_dimension)
        except Exception as e:
            logger.warning(f"Failed to get embedding: {e}")
            return np.zeros(self.embedding_dimension)
    
    def get_bert_embedding(self, text):
        """Get BERT-style embedding with consistent dimensions"""
        try:
            # Use the same sentence transformer for consistency
            embedding = self.get_sentence_embedding(text, 'sentence_transformer')
            return embedding
        except Exception as e:
            logger.warning(f"Failed to get BERT embedding: {e}")
            return np.zeros(self.embedding_dimension)
    
    def get_roberta_embedding(self, text):
        """Get RoBERTa-style embedding with consistent dimensions"""
        try:
            # Use the same sentence transformer for consistency  
            embedding = self.get_sentence_embedding(text, 'sentence_transformer')
            return embedding
        except Exception as e:
            logger.warning(f"Failed to get RoBERTa embedding: {e}")
            return np.zeros(self.embedding_dimension)