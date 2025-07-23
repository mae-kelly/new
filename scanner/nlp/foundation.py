#!/usr/bin/env python3

import os
import sys
import logging
import requests
import numpy as np
import warnings
from pathlib import Path

# Suppress warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

class FoundationModels:
    def __init__(self):
        self.models_loaded = False
        self.embedding_dimension = 384
        self.config = self._load_environment_config()
        
        # Session configuration like your working script
        self.session = self._create_session()
        
        try:
            self._initialize_models()
            logger.info("Foundation models initialized successfully")
        except Exception as e:
            logger.warning(f"Failed to initialize full models: {e}")
            self._initialize_fallback()
    
    def _load_environment_config(self):
        """Load configuration from environment variables like your working script"""
        return {
            'authority': os.getenv('AUTHORITY'),
            'client_id': os.getenv('CLIENT_ID'),
            'client_secret': os.getenv('CLIENT_SECRET'),
            'chronicle_api_key': os.getenv('CHRONICLE_API_KEY'),
            'chronicle_secret_key': os.getenv('CHRONICLE_SECRET_KEY'),
            'chronicle_endpoint': os.getenv('CHRONICLE_ENDPOINT'),
            'http_proxy': os.getenv('HTTP_PROXY'),
            'https_proxy': os.getenv('HTTPS_PROXY'),
            'authentication_configured': bool(os.getenv('CLIENT_ID')),
            'chronicle_configured': bool(os.getenv('CHRONICLE_API_KEY')),
            'proxy_configured': bool(os.getenv('HTTP_PROXY') or os.getenv('HTTPS_PROXY'))
        }
    
    def _create_session(self):
        """Create requests session with proxy configuration like your working script"""
        session = requests.Session()
        
        # Configure proxy like your working script
        if self.config['http_proxy'] or self.config['https_proxy']:
            session.proxies = {
                'http': self.config['http_proxy'],
                'https': self.config['https_proxy']
            }
            logger.info(f"Using proxy: {self.config['https_proxy'] or self.config['http_proxy']}")
        
        # Set timeouts
        session.timeout = 30
        
        # Configure SSL verification
        ssl_cert_file = os.getenv('SSL_CERT_FILE') or os.getenv('REQUESTS_CA_BUNDLE')
        if ssl_cert_file and os.path.exists(ssl_cert_file):
            session.verify = ssl_cert_file
            logger.info(f"Using SSL cert file: {ssl_cert_file}")
        
        return session
    
    def _initialize_models(self):
        """Initialize models with corporate network support"""
        try:
            # Try to import and initialize sentence transformers
            from sentence_transformers import SentenceTransformer
            
            # Use simple model that's more likely to download successfully
            model_name = "all-MiniLM-L6-v2"
            
            # Configure environment for model download
            if self.config['http_proxy']:
                os.environ['HTTP_PROXY'] = self.config['http_proxy']
                os.environ['HTTPS_PROXY'] = self.config['https_proxy'] or self.config['http_proxy']
            
            self.sentence_model = SentenceTransformer(model_name)
            self.embedding_dimension = self.sentence_model.get_sentence_embedding_dimension()
            
            # Initialize other models
            self._initialize_other_models()
            
            self.models_loaded = True
            logger.info(f"Sentence transformer loaded successfully with dimension {self.embedding_dimension}")
            
        except Exception as e:
            logger.warning(f"Failed to load sentence transformers: {e}")
            raise
    
    def _initialize_other_models(self):
        """Initialize other NLP models with error handling"""
        try:
            # Try to initialize BERT and other models
            from transformers import AutoTokenizer, AutoModel
            
            self.bert_tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
            self.bert_model = AutoModel.from_pretrained('bert-base-uncased')
            
        except Exception as e:
            logger.debug(f"Could not load transformers models: {e}")
            self.bert_tokenizer = None
            self.bert_model = None
    
    def _initialize_fallback(self):
        """Initialize with fallback functionality if main models fail"""
        logger.warning("Using fallback models - limited functionality")
        
        # Create simple embedding using basic text features
        self.sentence_model = None
        self.bert_model = None
        self.bert_tokenizer = None
        self.embedding_dimension = 100  # Reduced dimension for fallback
        self.models_loaded = False
    
    def get_sentence_embedding(self, text, encoder_type='default'):
        """Get sentence embedding with fallback"""
        try:
            if self.models_loaded and self.sentence_model:
                embedding = self.sentence_model.encode(text)
                return np.array(embedding)
            else:
                # Fallback to simple embedding
                return self._create_simple_embedding(text)
                
        except Exception as e:
            logger.warning(f"Sentence embedding failed: {e}")
            return self._create_simple_embedding(text)
    
    def get_bert_embedding(self, text):
        """Get BERT embedding with fallback"""
        try:
            if self.bert_model and self.bert_tokenizer:
                import torch
                
                inputs = self.bert_tokenizer(text, return_tensors='pt', max_length=512, truncation=True)
                with torch.no_grad():
                    outputs = self.bert_model(**inputs)
                    # Use [CLS] token embedding
                    embedding = outputs.last_hidden_state[:, 0, :].squeeze().numpy()
                    
                # Resize to match expected dimensions
                if len(embedding) > self.embedding_dimension:
                    embedding = embedding[:self.embedding_dimension]
                elif len(embedding) < self.embedding_dimension:
                    padding = np.zeros(self.embedding_dimension - len(embedding))
                    embedding = np.concatenate([embedding, padding])
                
                return embedding
            else:
                return self._create_simple_embedding(text)
                
        except Exception as e:
            logger.warning(f"BERT embedding failed: {e}")
            return self._create_simple_embedding(text)
    
    def _create_simple_embedding(self, text):
        """Create simple fallback embedding"""
        try:
            # Simple hash-based embedding
            import hashlib
            
            # Create multiple hash features
            features = []
            
            # Character frequency features
            char_counts = {}
            for char in text.lower():
                if char.isalnum():
                    char_counts[char] = char_counts.get(char, 0) + 1
            
            # Add top character frequencies (normalized)
            total_chars = len(text)
            for i in range(26):
                char = chr(ord('a') + i)
                freq = char_counts.get(char, 0) / max(total_chars, 1)
                features.append(freq)
            
            # Text statistics
            features.extend([
                len(text) / 100.0,  # Length (normalized)
                text.count(' ') / max(len(text), 1),  # Space ratio
                text.count('.') / max(len(text), 1),  # Dot ratio
                text.count('_') / max(len(text), 1),  # Underscore ratio
                sum(1 for c in text if c.isupper()) / max(len(text), 1),  # Uppercase ratio
                sum(1 for c in text if c.isdigit()) / max(len(text), 1),  # Digit ratio
            ])
            
            # Hash features for semantic content
            hash_obj = hashlib.md5(text.encode())
            hash_bytes = hash_obj.digest()
            
            # Convert hash to features
            hash_features = []
            for byte in hash_bytes[:20]:  # Use first 20 bytes
                hash_features.append(byte / 255.0)
            
            features.extend(hash_features)
            
            # Pad or truncate to desired dimension
            while len(features) < self.embedding_dimension:
                features.append(0.0)
            
            features = features[:self.embedding_dimension]
            
            return np.array(features, dtype=np.float32)
            
        except Exception as e:
            logger.warning(f"Simple embedding creation failed: {e}")
            return np.zeros(self.embedding_dimension, dtype=np.float32)
    
    def get_configuration_summary(self):
        """Get configuration summary for diagnostics"""
        return {
            'models_loaded': self.models_loaded,
            'embedding_dimension': self.embedding_dimension,
            'authentication_configured': self.config['authentication_configured'],
            'chronicle_configured': self.config['chronicle_configured'],
            'proxy_configured': self.config['proxy_configured'],
            'has_sentence_model': self.sentence_model is not None,
            'has_bert_model': self.bert_model is not None
        }
    
    def test_connectivity(self):
        """Test connectivity to required services"""
        test_results = {}
        
        # Test Hugging Face
        try:
            response = self.session.head('https://huggingface.co', timeout=10)
            test_results['huggingface'] = response.status_code < 400
        except:
            test_results['huggingface'] = False
        
        # Test PyPI
        try:
            response = self.session.head('https://pypi.org', timeout=10)
            test_results['pypi'] = response.status_code < 400
        except:
            test_results['pypi'] = False
        
        # Test Chronicle if configured
        if self.config['chronicle_endpoint']:
            try:
                response = self.session.head(self.config['chronicle_endpoint'], timeout=10)
                test_results['chronicle'] = response.status_code < 500
            except:
                test_results['chronicle'] = False
        
        return test_results#!/usr/bin/env python3

import os
import sys
import logging
import requests
import numpy as np
import warnings
from pathlib import Path

# Suppress warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

class FoundationModels:
    def __init__(self):
        self.models_loaded = False
        self.embedding_dimension = 384
        self.config = self._load_environment_config()
        
        # Session configuration like your working script
        self.session = self._create_session()
        
        try:
            self._initialize_models()
            logger.info("Foundation models initialized successfully")
        except Exception as e:
            logger.warning(f"Failed to initialize full models: {e}")
            self._initialize_fallback()
    
    def _load_environment_config(self):
        """Load configuration from environment variables like your working script"""
        return {
            'authority': os.getenv('AUTHORITY'),
            'client_id': os.getenv('CLIENT_ID'),
            'client_secret': os.getenv('CLIENT_SECRET'),
            'chronicle_api_key': os.getenv('CHRONICLE_API_KEY'),
            'chronicle_secret_key': os.getenv('CHRONICLE_SECRET_KEY'),
            'chronicle_endpoint': os.getenv('CHRONICLE_ENDPOINT'),
            'http_proxy': os.getenv('HTTP_PROXY'),
            'https_proxy': os.getenv('HTTPS_PROXY'),
            'authentication_configured': bool(os.getenv('CLIENT_ID')),
            'chronicle_configured': bool(os.getenv('CHRONICLE_API_KEY')),
            'proxy_configured': bool(os.getenv('HTTP_PROXY') or os.getenv('HTTPS_PROXY'))
        }
    
    def _create_session(self):
        """Create requests session with proxy configuration like your working script"""
        session = requests.Session()
        
        # Configure proxy like your working script
        if self.config['http_proxy'] or self.config['https_proxy']:
            session.proxies = {
                'http': self.config['http_proxy'],
                'https': self.config['https_proxy']
            }
            logger.info(f"Using proxy: {self.config['https_proxy'] or self.config['http_proxy']}")
        
        # Set timeouts
        session.timeout = 30
        
        # Configure SSL verification
        ssl_cert_file = os.getenv('SSL_CERT_FILE') or os.getenv('REQUESTS_CA_BUNDLE')
        if ssl_cert_file and os.path.exists(ssl_cert_file):
            session.verify = ssl_cert_file
            logger.info(f"Using SSL cert file: {ssl_cert_file}")
        
        return session
    
    def _initialize_models(self):
        """Initialize models with corporate network support"""
        try:
            # Try to import and initialize sentence transformers
            from sentence_transformers import SentenceTransformer
            
            # Use simple model that's more likely to download successfully
            model_name = "all-MiniLM-L6-v2"
            
            # Configure environment for model download
            if self.config['http_proxy']:
                os.environ['HTTP_PROXY'] = self.config['http_proxy']
                os.environ['HTTPS_PROXY'] = self.config['https_proxy'] or self.config['http_proxy']
            
            self.sentence_model = SentenceTransformer(model_name)
            self.embedding_dimension = self.sentence_model.get_sentence_embedding_dimension()
            
            # Initialize other models
            self._initialize_other_models()
            
            self.models_loaded = True
            logger.info(f"Sentence transformer loaded successfully with dimension {self.embedding_dimension}")
            
        except Exception as e:
            logger.warning(f"Failed to load sentence transformers: {e}")
            raise
    
    def _initialize_other_models(self):
        """Initialize other NLP models with error handling"""
        try:
            # Try to initialize BERT and other models
            from transformers import AutoTokenizer, AutoModel
            
            self.bert_tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
            self.bert_model = AutoModel.from_pretrained('bert-base-uncased')
            
        except Exception as e:
            logger.debug(f"Could not load transformers models: {e}")
            self.bert_tokenizer = None
            self.bert_model = None
    
    def _initialize_fallback(self):
        """Initialize with fallback functionality if main models fail"""
        logger.warning("Using fallback models - limited functionality")
        
        # Create simple embedding using basic text features
        self.sentence_model = None
        self.bert_model = None
        self.bert_tokenizer = None
        self.embedding_dimension = 100  # Reduced dimension for fallback
        self.models_loaded = False
    
    def get_sentence_embedding(self, text, encoder_type='default'):
        """Get sentence embedding with fallback"""
        try:
            if self.models_loaded and self.sentence_model:
                embedding = self.sentence_model.encode(text)
                return np.array(embedding)
            else:
                # Fallback to simple embedding
                return self._create_simple_embedding(text)
                
        except Exception as e:
            logger.warning(f"Sentence embedding failed: {e}")
            return self._create_simple_embedding(text)
    
    def get_bert_embedding(self, text):
        """Get BERT embedding with fallback"""
        try:
            if self.bert_model and self.bert_tokenizer:
                import torch
                
                inputs = self.bert_tokenizer(text, return_tensors='pt', max_length=512, truncation=True)
                with torch.no_grad():
                    outputs = self.bert_model(**inputs)
                    # Use [CLS] token embedding
                    embedding = outputs.last_hidden_state[:, 0, :].squeeze().numpy()
                    
                # Resize to match expected dimensions
                if len(embedding) > self.embedding_dimension:
                    embedding = embedding[:self.embedding_dimension]
                elif len(embedding) < self.embedding_dimension:
                    padding = np.zeros(self.embedding_dimension - len(embedding))
                    embedding = np.concatenate([embedding, padding])
                
                return embedding
            else:
                return self._create_simple_embedding(text)
                
        except Exception as e:
            logger.warning(f"BERT embedding failed: {e}")
            return self._create_simple_embedding(text)
    
    def _create_simple_embedding(self, text):
        """Create simple fallback embedding"""
        try:
            # Simple hash-based embedding
            import hashlib
            
            # Create multiple hash features
            features = []
            
            # Character frequency features
            char_counts = {}
            for char in text.lower():
                if char.isalnum():
                    char_counts[char] = char_counts.get(char, 0) + 1
            
            # Add top character frequencies (normalized)
            total_chars = len(text)
            for i in range(26):
                char = chr(ord('a') + i)
                freq = char_counts.get(char, 0) / max(total_chars, 1)
                features.append(freq)
            
            # Text statistics
            features.extend([
                len(text) / 100.0,  # Length (normalized)
                text.count(' ') / max(len(text), 1),  # Space ratio
                text.count('.') / max(len(text), 1),  # Dot ratio
                text.count('_') / max(len(text), 1),  # Underscore ratio
                sum(1 for c in text if c.isupper()) / max(len(text), 1),  # Uppercase ratio
                sum(1 for c in text if c.isdigit()) / max(len(text), 1),  # Digit ratio
            ])
            
            # Hash features for semantic content
            hash_obj = hashlib.md5(text.encode())
            hash_bytes = hash_obj.digest()
            
            # Convert hash to features
            hash_features = []
            for byte in hash_bytes[:20]:  # Use first 20 bytes
                hash_features.append(byte / 255.0)
            
            features.extend(hash_features)
            
            # Pad or truncate to desired dimension
            while len(features) < self.embedding_dimension:
                features.append(0.0)
            
            features = features[:self.embedding_dimension]
            
            return np.array(features, dtype=np.float32)
            
        except Exception as e:
            logger.warning(f"Simple embedding creation failed: {e}")
            return np.zeros(self.embedding_dimension, dtype=np.float32)
    
    def get_configuration_summary(self):
        """Get configuration summary for diagnostics"""
        return {
            'models_loaded': self.models_loaded,
            'embedding_dimension': self.embedding_dimension,
            'authentication_configured': self.config['authentication_configured'],
            'chronicle_configured': self.config['chronicle_configured'],
            'proxy_configured': self.config['proxy_configured'],
            'has_sentence_model': self.sentence_model is not None,
            'has_bert_model': self.bert_model is not None
        }
    
    def test_connectivity(self):
        """Test connectivity to required services"""
        test_results = {}
        
        # Test Hugging Face
        try:
            response = self.session.head('https://huggingface.co', timeout=10)
            test_results['huggingface'] = response.status_code < 400
        except:
            test_results['huggingface'] = False
        
        # Test PyPI
        try:
            response = self.session.head('https://pypi.org', timeout=10)
            test_results['pypi'] = response.status_code < 400
        except:
            test_results['pypi'] = False
        
        # Test Chronicle if configured
        if self.config['chronicle_endpoint']:
            try:
                response = self.session.head(self.config['chronicle_endpoint'], timeout=10)
                test_results['chronicle'] = response.status_code < 500
            except:
                test_results['chronicle'] = False
        
        return test_results