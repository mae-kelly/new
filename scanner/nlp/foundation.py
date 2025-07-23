import logging
import torch
import numpy as np
import os
import ssl
import requests
import certifi
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import urllib3

logger = logging.getLogger(__name__)

class FoundationModels:
    def __init__(self):
        self.models = {}
        self.tokenizers = {}
        self.embedding_dimension = 384
        self._configure_corporate_ssl()
        self._initialize_models()
    
    def _configure_corporate_ssl(self):
        """Configure SSL/TLS for corporate environments"""
        try:
            logger.info("Configuring SSL for corporate environment...")
            
            # Method 1: Try to use corporate proxy with authentication
            self._setup_corporate_proxy()
            
            # Method 2: Configure SSL context with corporate certificates
            self._setup_corporate_ssl_context()
            
            # Method 3: Configure requests and urllib3 for corporate environment
            self._setup_corporate_requests()
            
        except Exception as e:
            logger.warning(f"Corporate SSL configuration failed: {e}")
    
    def _setup_corporate_proxy(self):
        """Setup corporate proxy configuration"""
        # Get proxy from environment
        http_proxy = os.environ.get('HTTP_PROXY') or os.environ.get('http_proxy')
        https_proxy = os.environ.get('HTTPS_PROXY') or os.environ.get('https_proxy')
        
        if http_proxy or https_proxy:
            logger.info("Corporate proxy detected")
            
            # Set up proxy for all Python requests
            proxies = {}
            if http_proxy:
                proxies['http'] = http_proxy
                logger.info(f"HTTP proxy: {http_proxy}")
            if https_proxy:
                proxies['https'] = https_proxy
                logger.info(f"HTTPS proxy: {https_proxy}")
            
            # Configure environment for subprocess calls
            if http_proxy:
                os.environ['HTTP_PROXY'] = http_proxy
                os.environ['http_proxy'] = http_proxy
            if https_proxy:
                os.environ['HTTPS_PROXY'] = https_proxy
                os.environ['https_proxy'] = https_proxy
    
    def _setup_corporate_ssl_context(self):
        """Setup SSL context for corporate certificates"""
        
        # Find corporate CA bundle
        ca_bundle_paths = [
            os.environ.get('REQUESTS_CA_BUNDLE'),
            os.environ.get('CURL_CA_BUNDLE'),
            os.environ.get('SSL_CERT_FILE'),
            '/etc/ssl/certs/ca-certificates.crt',  # Ubuntu/Debian
            '/etc/ssl/certs/ca-bundle.crt',        # CentOS/RHEL
            '/etc/pki/tls/certs/ca-bundle.crt',    # Fedora/CentOS
            '/usr/local/share/certs/ca-root-nss.crt',  # FreeBSD
            '/etc/ssl/cert.pem',                   # macOS
            certifi.where()                        # Fallback to certifi
        ]
        
        ca_bundle = None
        for path in ca_bundle_paths:
            if path and os.path.exists(path):
                ca_bundle = path
                logger.info(f"Using CA bundle: {ca_bundle}")
                break
        
        if ca_bundle:
            # Set for all SSL operations
            os.environ['REQUESTS_CA_BUNDLE'] = ca_bundle
            os.environ['CURL_CA_BUNDLE'] = ca_bundle
            os.environ['SSL_CERT_FILE'] = ca_bundle
            
            # Configure SSL context
            ssl_context = ssl.create_default_context(cafile=ca_bundle)
            ssl._create_default_https_context = lambda: ssl_context
        else:
            logger.warning("No corporate CA bundle found, using system default")
    
    def _setup_corporate_requests(self):
        """Configure requests library for corporate environment"""
        
        # Configure urllib3 to handle corporate proxies
        urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
        
        # Create custom session for downloads
        self.session = requests.Session()
        
        # Set up retry strategy
        retry_strategy = Retry(
            total=5,
            backoff_factor=2,
            status_forcelist=[429, 500, 502, 503, 504],
        )
        
        adapter = HTTPAdapter(max_retries=retry_strategy)
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)
        
        # Configure proxy
        http_proxy = os.environ.get('HTTP_PROXY') or os.environ.get('http_proxy')
        https_proxy = os.environ.get('HTTPS_PROXY') or os.environ.get('https_proxy')
        
        if http_proxy or https_proxy:
            proxies = {}
            if http_proxy:
                proxies['http'] = http_proxy
            if https_proxy:
                proxies['https'] = https_proxy
            self.session.proxies.update(proxies)
        
        # Set CA bundle
        ca_bundle = os.environ.get('REQUESTS_CA_BUNDLE')
        if ca_bundle and os.path.exists(ca_bundle):
            self.session.verify = ca_bundle
        
        # Set timeout
        self.session.timeout = 60
        
        # Patch requests to use our session
        self._patch_requests_for_transformers()
    
    def _patch_requests_for_transformers(self):
        """Patch requests/urllib to work with transformers library"""
        try:
            # Patch the session into transformers and sentence-transformers
            import transformers.utils.hub
            import sentence_transformers.util
            
            # Store original functions
            original_http_get = getattr(sentence_transformers.util, 'http_get', None)
            original_hf_hub_download = getattr(transformers.utils.hub, 'hf_hub_download', None)
            
            # Create patched version that uses our session
            def patched_http_get(url, temp_file, proxies=None, resume_size=0, headers=None):
                """Use our configured session for sentence-transformers downloads"""
                try:
                    logger.info(f"Downloading via corporate proxy: {url}")
                    
                    # Use our configured session
                    headers = headers or {}
                    if resume_size > 0:
                        headers['Range'] = f'bytes={resume_size}-'
                    
                    response = self.session.get(url, headers=headers, stream=True)
                    response.raise_for_status()
                    
                    # Write to temp file
                    with open(temp_file, 'ab' if resume_size > 0 else 'wb') as f:
                        for chunk in response.iter_content(chunk_size=8192):
                            if chunk:
                                f.write(chunk)
                    
                    return temp_file
                    
                except Exception as e:
                    logger.error(f"Corporate download failed: {e}")
                    # Fall back to original function if available
                    if original_http_get:
                        return original_http_get(url, temp_file, proxies, resume_size, headers)
                    raise
            
            # Apply patches
            if hasattr(sentence_transformers.util, 'http_get'):
                sentence_transformers.util.http_get = patched_http_get
                logger.info("Patched sentence_transformers for corporate proxy")
            
        except ImportError:
            logger.debug("Transformers not yet imported, will patch later")
        except Exception as e:
            logger.warning(f"Failed to patch transformers: {e}")
    
    def _initialize_models(self):
        """Initialize models with corporate network support"""
        try:
            logger.info("Initializing sentence transformers with corporate network support...")
            
            # Import after SSL configuration
            from sentence_transformers import SentenceTransformer
            
            # Ensure patches are applied
            self._patch_requests_for_transformers()
            
            # Download with corporate network settings
            logger.info("Downloading all-MiniLM-L6-v2 model...")
            self.models['sentence_transformer'] = SentenceTransformer(
                'all-MiniLM-L6-v2',
                cache_folder=os.path.expanduser('~/.cache/sentence_transformers_corporate')
            )
            
            # Use same model for consistency
            self.models['domain_encoder'] = self.models['sentence_transformer']
            self.models['code_encoder'] = self.models['sentence_transformer']
            
            # Test embedding
            test_embedding = self.models['sentence_transformer'].encode(["corporate test"])
            self.embedding_dimension = test_embedding.shape[1]
            
            logger.info(f"✅ SentenceTransformer models loaded successfully!")
            logger.info(f"   Model: all-MiniLM-L6-v2")
            logger.info(f"   Embedding dimension: {self.embedding_dimension}")
            
        except Exception as e:
            logger.error(f"❌ Model initialization failed: {e}")
            logger.error("This could be due to:")
            logger.error("  • Corporate firewall blocking model downloads")
            logger.error("  • Proxy authentication required")
            logger.error("  • SSL certificate issues")
            logger.error("  • Network connectivity problems")
            
            # Provide specific troubleshooting
            self._provide_troubleshooting_info()
            raise
    
    def _provide_troubleshooting_info(self):
        """Provide specific troubleshooting information"""
        logger.error("\n🔧 TROUBLESHOOTING STEPS:")
        
        # Check proxy
        http_proxy = os.environ.get('HTTP_PROXY') or os.environ.get('http_proxy')
        https_proxy = os.environ.get('HTTPS_PROXY') or os.environ.get('https_proxy')
        
        if not (http_proxy or https_proxy):
            logger.error("1. SET PROXY ENVIRONMENT VARIABLES:")
            logger.error("   export HTTP_PROXY=http://proxy.company.com:8080")
            logger.error("   export HTTPS_PROXY=http://proxy.company.com:8080")
        else:
            logger.error(f"1. PROXY CONFIGURED: HTTP={http_proxy}, HTTPS={https_proxy}")
        
        # Check CA bundle
        ca_bundle = os.environ.get('REQUESTS_CA_BUNDLE')
        if not ca_bundle or not os.path.exists(ca_bundle):
            logger.error("2. SET CORPORATE CA BUNDLE:")
            logger.error("   export REQUESTS_CA_BUNDLE=/path/to/corporate-ca-bundle.crt")
        else:
            logger.error(f"2. CA BUNDLE CONFIGURED: {ca_bundle}")
        
        # Test connectivity
        logger.error("3. TEST CONNECTIVITY:")
        logger.error("   curl -v https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2")
        
        # Alternative solutions
        logger.error("4. ALTERNATIVE SOLUTIONS:")
        logger.error("   • Download models manually to ~/.cache/sentence_transformers/")
        logger.error("   • Use pip install with --trusted-host flags")
        logger.error("   • Contact IT for firewall whitelist:")
        logger.error("     - huggingface.co")
        logger.error("     - cdn-lfs.huggingface.co")
        logger.error("     - pypi.org")
    
    def get_sentence_embedding(self, text, model_type='sentence_transformer'):
        """Get sentence embedding"""
        try:
            model = self.models.get(model_type)
            if model:
                embedding = model.encode([text])[0]
                
                # Ensure consistent dimensionality
                if len(embedding) != self.embedding_dimension:
                    if len(embedding) < self.embedding_dimension:
                        padding = np.zeros(self.embedding_dimension - len(embedding))
                        embedding = np.concatenate([embedding, padding])
                    else:
                        embedding = embedding[:self.embedding_dimension]
                
                return embedding
            else:
                logger.error("Model not available")
                raise RuntimeError("Models not properly initialized")
                
        except Exception as e:
            logger.error(f"Failed to get embedding: {e}")
            raise
    
    def get_bert_embedding(self, text):
        """Get BERT-style embedding"""
        return self.get_sentence_embedding(text, 'sentence_transformer')
    
    def get_roberta_embedding(self, text):
        """Get RoBERTa-style embedding"""
        return self.get_sentence_embedding(text, 'sentence_transformer')
    
    def is_fallback_mode(self):
        """Check if running in fallback mode"""
        return False  # We don't use fallback mode - we fix the corporate network issues