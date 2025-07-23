import logging
import numpy as np
import os
import ssl
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

logger = logging.getLogger(__name__)

class FoundationModels:
    def __init__(self):
        self.models = {}
        self.tokenizers = {}
        self.embedding_dimension = 384  # Standardize on 384 dimensions
        self.fallback_mode = False
        self._initialize_models()
    
    def _initialize_models(self):
        """Initialize models with corporate proxy support"""
        try:
            # Configure for corporate environment
            self._configure_corporate_environment()
            
            # Try lightweight models first
            logger.info("Initializing sentence transformers for corporate environment...")
            self._try_sentence_transformers()
            
        except Exception as e:
            logger.warning(f"SentenceTransformer initialization failed: {e}")
            logger.info("Falling back to simple embedding generation")
            self._initialize_simple_embeddings()
    
    def _configure_corporate_environment(self):
        """Configure for corporate proxy environment"""
        try:
            # Read proxy settings from environment
            http_proxy = os.environ.get('HTTP_PROXY') or os.environ.get('http_proxy')
            https_proxy = os.environ.get('HTTPS_PROXY') or os.environ.get('https_proxy')
            
            if http_proxy or https_proxy:
                logger.info(f"Detected corporate proxy environment")
                if http_proxy:
                    logger.info(f"HTTP proxy: {http_proxy}")
                if https_proxy:
                    logger.info(f"HTTPS proxy: {https_proxy}")
            
            # Configure requests to use corporate certificates
            # Check for corporate CA bundle
            ca_bundle_paths = [
                os.environ.get('REQUESTS_CA_BUNDLE'),
                os.environ.get('CURL_CA_BUNDLE'),
                '/etc/ssl/certs/ca-certificates.crt',  # Ubuntu/Debian
                '/etc/ssl/certs/ca-bundle.crt',        # CentOS/RHEL
                '/etc/pki/tls/certs/ca-bundle.crt',    # Fedora/CentOS
                '/usr/local/share/certs/ca-root-nss.crt',  # FreeBSD
                '/etc/ssl/cert.pem'                     # macOS
            ]
            
            ca_bundle = None
            for path in ca_bundle_paths:
                if path and os.path.exists(path):
                    ca_bundle = path
                    logger.info(f"Using CA bundle: {ca_bundle}")
                    break
            
            # Configure session for corporate environment
            self.session = requests.Session()
            
            # Set up retries
            retry_strategy = Retry(
                total=3,
                backoff_factor=1,
                status_forcelist=[429, 500, 502, 503, 504],
            )
            
            adapter = HTTPAdapter(max_retries=retry_strategy)
            self.session.mount("http://", adapter)
            self.session.mount("https://", adapter)
            
            # Configure proxy settings
            if http_proxy or https_proxy:
                proxies = {}
                if http_proxy:
                    proxies['http'] = http_proxy
                if https_proxy:
                    proxies['https'] = https_proxy
                self.session.proxies.update(proxies)
            
            # Configure SSL verification
            if ca_bundle:
                self.session.verify = ca_bundle
                os.environ['REQUESTS_CA_BUNDLE'] = ca_bundle
                os.environ['CURL_CA_BUNDLE'] = ca_bundle
            
            # Set timeout
            self.session.timeout = 30
            
        except Exception as e:
            logger.warning(f"Corporate environment configuration failed: {e}")
            # Create basic session
            self.session = requests.Session()
    
    def _try_sentence_transformers(self):
        """Try to initialize SentenceTransformer models with corporate settings"""
        try:
            # Import with retry
            import sentence_transformers
            from sentence_transformers import SentenceTransformer
            
            # Configure sentence-transformers to use our session
            # Monkey patch the default session if possible
            if hasattr(sentence_transformers.util, 'http_get'):
                original_http_get = sentence_transformers.util.http_get
                
                def corporate_http_get(url, temp_file, proxies=None, resume_size=0, headers=None):
                    """Use our configured session for downloads"""
                    try:
                        return original_http_get(url, temp_file, 
                                               proxies=self.session.proxies if hasattr(self, 'session') else proxies,
                                               resume_size=resume_size, headers=headers)
                    except Exception as e:
                        logger.warning(f"Download with corporate settings failed: {e}")
                        return original_http_get(url, temp_file, proxies=proxies, 
                                               resume_size=resume_size, headers=headers)
                
                sentence_transformers.util.http_get = corporate_http_get
            
            # Try to initialize models
            logger.info("Downloading sentence transformer model (may take a moment)...")
            
            # Use the smallest reliable model
            model_name = 'all-MiniLM-L6-v2'
            
            # Initialize with timeout and retry
            self.models['sentence_transformer'] = SentenceTransformer(
                model_name, 
                cache_folder=os.path.join(os.path.expanduser('~'), '.cache', 'sentence_transformers')
            )
            
            # Use same model for all encoders to ensure consistency
            self.models['domain_encoder'] = self.models['sentence_transformer']
            self.models['code_encoder'] = self.models['sentence_transformer']
            
            # Test embedding generation
            test_embedding = self.models['sentence_transformer'].encode(["test connection"])
            self.embedding_dimension = test_embedding.shape[1]
            
            logger.info(f"✅ SentenceTransformer models initialized successfully")
            logger.info(f"   Model: {model_name}")
            logger.info(f"   Embedding dimension: {self.embedding_dimension}")
            
        except Exception as e:
            logger.error(f"SentenceTransformer initialization failed: {e}")
            logger.error("This could be due to:")
            logger.error("  • Corporate firewall blocking model downloads")
            logger.error("  • Proxy configuration issues") 
            logger.error("  • SSL certificate verification problems")
            raise
    
    def _initialize_simple_embeddings(self):
        """Initialize simple hash-based embeddings as fallback"""
        logger.warning("⚠️  Using simple hash-based embeddings as fallback")
        logger.warning("   Performance will be reduced but scanner will still work")
        self.fallback_mode = True
        self.models['sentence_transformer'] = None
        self.models['domain_encoder'] = None
        self.models['code_encoder'] = None
        self.embedding_dimension = 384
    
    def get_sentence_embedding(self, text, model_type='sentence_transformer'):
        """Get consistent-dimension embeddings with fallback"""
        try:
            if not self.fallback_mode and self.models.get(model_type):
                # Use actual SentenceTransformer
                embedding = self.models[model_type].encode([text])[0]
                
                # Ensure consistent dimensionality
                if len(embedding) != self.embedding_dimension:
                    if len(embedding) < self.embedding_dimension:
                        padding = np.zeros(self.embedding_dimension - len(embedding))
                        embedding = np.concatenate([embedding, padding])
                    else:
                        embedding = embedding[:self.embedding_dimension]
                
                return embedding
            else:
                # Use simple hash-based embedding
                return self._generate_simple_embedding(text)
                
        except Exception as e:
            logger.debug(f"Failed to get embedding, using fallback: {e}")
            return self._generate_simple_embedding(text)
    
    def _generate_simple_embedding(self, text):
        """Generate simple deterministic embedding based on text features"""
        try:
            # Simple feature extraction that works well for field analysis
            text_lower = text.lower().strip()
            
            # Create a simple feature vector
            features = np.zeros(self.embedding_dimension)
            
            # Character-based features (first 100 dimensions)
            for i, char in enumerate(text_lower[:100]):
                if i < 100:
                    features[i] = ord(char) / 127.0  # Normalize to 0-1
            
            # Word-based features (next 100 dimensions)
            words = text_lower.split()[:25]  # Max 25 words
            for i, word in enumerate(words):
                if i < 25:
                    word_hash = hash(word) % 1000 / 1000.0  # Normalize hash
                    features[100 + i*4] = word_hash
                    features[101 + i*4] = len(word) / 20.0  # Word length feature
                    features[102 + i*4] = 1.0 if word.isalpha() else 0.0  # Alpha feature
                    features[103 + i*4] = 1.0 if word.isdigit() else 0.0  # Digit feature
            
            # Length and pattern features (next 84 dimensions)
            features[200] = min(len(text) / 100.0, 1.0)  # Text length
            features[201] = min(len(words) / 50.0, 1.0)   # Word count
            features[202] = 1.0 if any(c.isdigit() for c in text) else 0.0  # Has digits
            features[203] = 1.0 if any(c.isupper() for c in text) else 0.0  # Has uppercase
            features[204] = 1.0 if '_' in text else 0.0  # Has underscore
            features[205] = 1.0 if '-' in text else 0.0  # Has dash
            features[206] = 1.0 if '.' in text else 0.0  # Has dot
            features[207] = min(text.count('_') / 10.0, 1.0)  # Underscore count
            
            # Domain-specific keyword features (remaining dimensions)
            ao1_keywords = {
                'host': 208, 'hostname': 209, 'device': 210, 'machine': 211, 'server': 212, 'asset': 213,
                'ip': 214, 'network': 215, 'address': 216, 'dns': 217, 'url': 218, 'domain': 219,
                'security': 220, 'agent': 221, 'tool': 222, 'sensor': 223, 'edr': 224, 'crowdstrike': 225,
                'log': 226, 'event': 227, 'audit': 228, 'source': 229, 'siem': 230, 'chronicle': 231,
                'country': 232, 'region': 233, 'location': 234, 'site': 235, 'zone': 236, 'datacenter': 237,
                'business': 238, 'unit': 239, 'department': 240, 'owner': 241, 'app': 242, 'application': 243,
                'cloud': 244, 'onprem': 245, 'saas': 246, 'api': 247, 'platform': 248, 'infrastructure': 249,
                'system': 250, 'type': 251, 'category': 252, 'class': 253, 'critical': 254, 'production': 255,
                'coverage': 256, 'metric': 257, 'score': 258, 'percentage': 259, 'count': 260, 'total': 261
            }
            
            for keyword, index in ao1_keywords.items():
                if index < self.embedding_dimension and keyword in text_lower:
                    features[index] = 1.0
            
            # Add some randomness based on text hash for better separation
            np.random.seed(hash(text) % 2**31)  # Use positive seed
            noise = np.random.normal(0, 0.01, self.embedding_dimension)
            features = features + noise
            
            # Normalize the vector
            norm = np.linalg.norm(features)
            if norm > 0:
                features = features / norm
            
            return features
            
        except Exception as e:
            logger.warning(f"Simple embedding generation failed: {e}")
            # Return random but deterministic vector based on text hash
            np.random.seed(abs(hash(text)) % 2**31)
            return np.random.normal(0, 0.1, self.embedding_dimension)
    
    def get_bert_embedding(self, text):
        """Get BERT-style embedding (same as sentence embedding for consistency)"""
        return self.get_sentence_embedding(text, 'sentence_transformer')
    
    def get_roberta_embedding(self, text):
        """Get RoBERTa-style embedding (same as sentence embedding for consistency)"""
        return self.get_sentence_embedding(text, 'sentence_transformer')
    
    def is_fallback_mode(self):
        """Check if running in fallback mode"""
        return self.fallback_mode