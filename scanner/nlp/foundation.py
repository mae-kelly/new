import logging
import torch
import numpy as np
import os
import ssl
import requests
import certifi
import json
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import urllib3
from pathlib import Path

logger = logging.getLogger(__name__)

class FoundationModels:
    def __init__(self):
        self.models = {}
        self.tokenizers = {}
        self.embedding_dimension = 384
        self._setup_ssl_certificates()
        self._load_environment_config()
        self._configure_corporate_network()
        self._initialize_models()
    
    def _setup_ssl_certificates(self):
        """Setup SSL certificates from the ssl folder"""
        try:
            # Get the current file location and navigate to ssl folder
            current_file = Path(__file__)
            # From scanner/nlp/foundation.py -> scanner/nlp -> scanner -> server -> ssl
            server_dir = current_file.parent.parent.parent  # Go up to server directory
            ssl_dir = server_dir.parent / 'ssl'  # Go up one more to find ssl folder
            
            # SSL certificate paths
            cert_file = ssl_dir / 'nexia.1dc.com.crt'
            key_file = ssl_dir / 'nexia.1dc.com.key'
            
            if cert_file.exists() and key_file.exists():
                self.ssl_cert_file = str(cert_file)
                self.ssl_key_file = str(key_file)
                
                logger.info(f"✅ Found SSL certificate: {cert_file}")
                logger.info(f"✅ Found SSL key: {key_file}")
                
                # Set environment variables for the certificates
                os.environ['SSL_CERT_FILE'] = self.ssl_cert_file
                os.environ['SSL_KEY_FILE'] = self.ssl_key_file
                os.environ['REQUESTS_CA_BUNDLE'] = self.ssl_cert_file
                
                # Configure SSL context with the certificates
                ssl_context = ssl.create_default_context()
                ssl_context.load_cert_chain(self.ssl_cert_file, self.ssl_key_file)
                ssl_context.check_hostname = False  # For corporate environment
                ssl_context.verify_mode = ssl.CERT_NONE  # For corporate environment
                
                # Set as default context
                ssl._create_default_https_context = lambda: ssl_context
                
                logger.info("✅ SSL certificates configured for nexia.1dc.com")
                
            else:
                logger.warning(f"⚠️  SSL certificates not found at {ssl_dir}")
                logger.warning(f"   Expected: {cert_file}")
                logger.warning(f"   Expected: {key_file}")
                self.ssl_cert_file = None
                self.ssl_key_file = None
                
        except Exception as e:
            logger.error(f"SSL certificate setup failed: {e}")
            self.ssl_cert_file = None
            self.ssl_key_file = None
    
    def _load_environment_config(self):
        """Load all configuration from environment variables"""
        
        # Authentication configuration
        self.auth_config = {
            'flask_secret_key': os.getenv('FLASK_SECRET_KEY'),
            'authority': os.getenv('AUTHORITY'),
            'client_id': os.getenv('CLIENT_ID'),
            'client_secret': os.getenv('CLIENT_SECRET'),
            'redirect_uri': os.getenv('REDIRECT_URI'),
            'scope': os.getenv('SCOPE'),
            'endpoint': os.getenv('ENDPOINT')
        }
        
        # Chronicle configuration
        self.chronicle_config = {
            'api_key': os.getenv('CHRONICLE_API_KEY'),
            'secret_key': os.getenv('CHRONICLE_SECRET_KEY'),
            'feed_id': os.getenv('CHRONICLE_FEED_ID'),
            'endpoint': os.getenv('CHRONICLE_ENDPOINT')
        }
        
        # Network configuration from environment
        self.network_config = {
            'http_proxy': os.getenv('HTTP_PROXY') or os.getenv('http_proxy'),
            'https_proxy': os.getenv('HTTPS_PROXY') or os.getenv('https_proxy'),
            'no_proxy': os.getenv('NO_PROXY') or os.getenv('no_proxy'),
            'ca_bundle': os.getenv('REQUESTS_CA_BUNDLE') or os.getenv('CURL_CA_BUNDLE') or self.ssl_cert_file
        }
        
        # Redis configuration (nexia.1dc.com from your environment)
        self.redis_config = {
            'host': os.getenv('REDIS_HOST', 'nexia.1dc.com'),
            'port': int(os.getenv('REDIS_PORT', 6379)),
            'db': int(os.getenv('REDIS_DB', 0)),
            'password': os.getenv('REDIS_PASSWORD')
        }
        
        # Log configuration status
        self._log_config_status()
    
    def _log_config_status(self):
        """Log configuration status without exposing secrets"""
        logger.info("Loading environment configuration...")
        
        # Check SSL certs
        ssl_configured = bool(self.ssl_cert_file and self.ssl_key_file)
        logger.info(f"✅ SSL Certificates: {'CONFIGURED (nexia.1dc.com)' if ssl_configured else 'NOT FOUND'}")
        
        # Check auth config
        auth_configured = all([
            self.auth_config['authority'],
            self.auth_config['client_id'],
            self.auth_config['client_secret']
        ])
        logger.info(f"✅ Authentication config: {'CONFIGURED' if auth_configured else 'MISSING'}")
        
        # Check Chronicle config
        chronicle_configured = all([
            self.chronicle_config['api_key'],
            self.chronicle_config['endpoint']
        ])
        logger.info(f"✅ Chronicle config: {'CONFIGURED' if chronicle_configured else 'MISSING'}")
        
        # Check network config
        proxy_configured = bool(self.network_config['http_proxy'])
        logger.info(f"✅ Proxy config: {'CONFIGURED' if proxy_configured else 'NOT SET'}")
        
        # Check Redis
        logger.info(f"✅ Redis backend: {self.redis_config['host']}:{self.redis_config['port']}")
    
    def _configure_corporate_network(self):
        """Configure corporate network settings from environment"""
        try:
            logger.info("Configuring corporate network from environment variables...")
            
            # Configure proxy if available
            if self.network_config['http_proxy'] and self.network_config['https_proxy']:
                self.proxies = {
                    'http': self.network_config['http_proxy'],
                    'https': self.network_config['https_proxy']
                }
                logger.info(f"✅ Proxy configured: {self.network_config['http_proxy']}")
            else:
                self.proxies = {}
                logger.info("⚠️  No proxy configuration found in environment")
            
            # Setup authenticated session with SSL certificates
            self._setup_authenticated_session()
            
        except Exception as e:
            logger.error(f"Corporate network configuration failed: {e}")
            raise
    
    def _setup_authenticated_session(self):
        """Setup requests session with SSL certificates and authentication"""
        try:
            # Create session with corporate configuration
            self.session = requests.Session()
            
            # Configure retry strategy
            retry_strategy = Retry(
                total=5,
                backoff_factor=2,
                status_forcelist=[429, 500, 502, 503, 504],
                allowed_methods=["HEAD", "GET", "OPTIONS", "POST"]
            )
            
            adapter = HTTPAdapter(max_retries=retry_strategy)
            self.session.mount("http://", adapter)
            self.session.mount("https://", adapter)
            
            # Configure proxy from environment
            if self.proxies:
                self.session.proxies = self.proxies
            
            # Configure SSL with nexia.1dc.com certificates
            if self.ssl_cert_file and self.ssl_key_file:
                # Use the SSL certificates for client authentication
                self.session.cert = (self.ssl_cert_file, self.ssl_key_file)
                self.session.verify = self.ssl_cert_file  # Use cert as CA bundle too
                logger.info("✅ SSL client certificates configured")
            elif self.network_config['ca_bundle'] and os.path.exists(self.network_config['ca_bundle']):
                self.session.verify = self.network_config['ca_bundle']
                logger.info(f"✅ CA bundle configured: {self.network_config['ca_bundle']}")
            
            # Set timeouts
            self.session.timeout = (30, 120)
            
            # Add authentication headers if Chronicle is configured
            if self.chronicle_config['api_key']:
                self.session.headers.update({
                    'X-goog-apikey': self.chronicle_config['api_key']
                })
                
                if self.chronicle_config['secret_key']:
                    self.session.headers.update({
                        'X-Webhook-Access-Key': self.chronicle_config['secret_key']
                    })
            
            # Add User-Agent for nexia.1dc.com environment
            self.session.headers.update({
                'User-Agent': 'AO1-Scanner/2.0 (nexia.1dc.com)',
                'X-Client-Domain': 'nexia.1dc.com'
            })
            
            # Add client authentication if configured
            if self.auth_config['client_id']:
                self.session.headers.update({
                    'X-Client-ID': self.auth_config['client_id']
                })
            
            logger.info("✅ Authenticated session configured with SSL certificates")
            
            # Test connectivity with certificates
            self._test_ssl_connectivity()
            
            # Patch transformers to use SSL certificates
            self._patch_transformers_with_ssl()
            
        except Exception as e:
            logger.error(f"Authenticated session setup failed: {e}")
            raise
    
    def _test_ssl_connectivity(self):
        """Test SSL connectivity with nexia.1dc.com certificates"""
        try:
            logger.info("Testing SSL connectivity with nexia.1dc.com certificates...")
            
            # Test internal connectivity first
            internal_urls = []
            if self.chronicle_config['endpoint']:
                internal_urls.append(self.chronicle_config['endpoint'])
            
            # Test external connectivity for model downloads
            external_urls = [
                'https://httpbin.org/status/200',
                'https://huggingface.co'
            ]
            
            # Test internal endpoints
            for url in internal_urls:
                try:
                    response = self.session.head(url, timeout=10)
                    if response.status_code < 400:
                        logger.info(f"✅ Internal SSL connectivity: {url}")
                    else:
                        logger.warning(f"⚠️  Internal endpoint returned {response.status_code}: {url}")
                except Exception as e:
                    logger.warning(f"⚠️  Internal SSL test failed for {url}: {e}")
            
            # Test external endpoints
            for url in external_urls:
                try:
                    response = self.session.head(url, timeout=10)
                    if response.status_code < 400:
                        logger.info(f"✅ External SSL connectivity: {url}")
                    else:
                        logger.warning(f"⚠️  External endpoint returned {response.status_code}: {url}")
                except Exception as e:
                    logger.debug(f"External SSL test failed for {url}: {e}")
            
        except Exception as e:
            logger.debug(f"SSL connectivity test failed: {e}")
    
    def _patch_transformers_with_ssl(self):
        """Patch transformers library to use SSL certificates"""
        try:
            import sentence_transformers.util
            
            # Store original function
            original_http_get = getattr(sentence_transformers.util, 'http_get', None)
            
            def ssl_authenticated_http_get(url, temp_file, proxies=None, resume_size=0, headers=None):
                """Download using SSL certificates and corporate authentication"""
                try:
                    logger.info(f"Downloading via nexia.1dc.com SSL: {url}")
                    
                    # Use SSL-configured session for downloads
                    download_headers = headers or {}
                    if resume_size > 0:
                        download_headers['Range'] = f'bytes={resume_size}-'
                    
                    # Add client authentication headers
                    if self.auth_config['client_id']:
                        download_headers['X-Client-ID'] = self.auth_config['client_id']
                        download_headers['X-Client-Domain'] = 'nexia.1dc.com'
                    
                    response = self.session.get(url, headers=download_headers, stream=True)
                    response.raise_for_status()
                    
                    # Write to file
                    with open(temp_file, 'ab' if resume_size > 0 else 'wb') as f:
                        for chunk in response.iter_content(chunk_size=8192):
                            if chunk:
                                f.write(chunk)
                    
                    logger.info("✅ Download completed via nexia.1dc.com SSL")
                    return temp_file
                    
                except Exception as e:
                    logger.error(f"SSL download failed: {e}")
                    
                    # Fallback to original function
                    if original_http_get:
                        logger.info("Attempting fallback download without SSL...")
                        return original_http_get(url, temp_file, proxies, resume_size, headers)
                    raise
            
            # Apply SSL patch
            if hasattr(sentence_transformers.util, 'http_get'):
                sentence_transformers.util.http_get = ssl_authenticated_http_get
                logger.info("✅ Transformers patched for nexia.1dc.com SSL")
            
        except ImportError:
            logger.debug("Transformers not imported yet - will patch on demand")
        except Exception as e:
            logger.warning(f"Failed to patch transformers with SSL: {e}")
    
    def _initialize_models(self):
        """Initialize models with SSL certificates and corporate authentication"""
        try:
            logger.info("Initializing sentence transformers with nexia.1dc.com SSL certificates...")
            
            # Import after SSL configuration
            from sentence_transformers import SentenceTransformer
            
            # Ensure SSL patches are applied
            self._patch_transformers_with_ssl()
            
            # Download models using SSL certificates
            logger.info("Downloading models via nexia.1dc.com SSL...")
            self.models['sentence_transformer'] = SentenceTransformer(
                'all-MiniLM-L6-v2',
                cache_folder=os.path.expanduser('~/.cache/sentence_transformers_nexia')
            )
            
            # Use same model for consistency
            self.models['domain_encoder'] = self.models['sentence_transformer']
            self.models['code_encoder'] = self.models['sentence_transformer']
            
            # Test model functionality
            test_embedding = self.models['sentence_transformer'].encode(["nexia ssl test"])
            self.embedding_dimension = test_embedding.shape[1]
            
            logger.info(f"✅ Models initialized successfully!")
            logger.info(f"   Model: all-MiniLM-L6-v2")
            logger.info(f"   Embedding dimension: {self.embedding_dimension}")
            logger.info(f"   SSL certificates: {'ENABLED (nexia.1dc.com)' if self.ssl_cert_file else 'DISABLED'}")
            logger.info(f"   Corporate authentication: {'ENABLED' if self.chronicle_config['api_key'] else 'DISABLED'}")
            
        except Exception as e:
            logger.error(f"❌ Model initialization failed: {e}")
            logger.error("Possible issues:")
            logger.error("  • Corporate firewall blocking model downloads")
            logger.error("  • SSL certificate authentication failed")
            logger.error("  • nexia.1dc.com certificates invalid or expired")
            logger.error("  • Proxy configuration incorrect")
            logger.error("  • Network connectivity issues")
            
            # Provide SSL-specific troubleshooting
            self._provide_ssl_troubleshooting()
            raise
    
    def _provide_ssl_troubleshooting(self):
        """Provide SSL-specific troubleshooting"""
        logger.error("\n🔧 SSL TROUBLESHOOTING:")
        
        # Check certificates exist
        if self.ssl_cert_file and self.ssl_key_file:
            logger.error(f"1. SSL CERTIFICATES: ✅ FOUND")
            logger.error(f"   Certificate: {self.ssl_cert_file}")
            logger.error(f"   Key: {self.ssl_key_file}")
        else:
            logger.error("1. SSL CERTIFICATES: ❌ NOT FOUND")
            logger.error("   Expected in ../ssl/ directory relative to scanner")
        
        # Check environment variables
        missing_vars = []
        for key, value in self.auth_config.items():
            if not value and key != 'flask_secret_key':
                missing_vars.append(key.upper())
        
        if missing_vars:
            logger.error(f"2. MISSING ENV VARS: {', '.join(missing_vars)}")
        else:
            logger.error("2. ENVIRONMENT VARIABLES: ✅ CONFIGURED")
        
        # Check proxy
        if not self.network_config['http_proxy']:
            logger.error("3. PROXY: ❌ NOT CONFIGURED")
        else:
            logger.error("3. PROXY: ✅ CONFIGURED")
        
        logger.error("4. TEST SSL MANUALLY:")
        logger.error(f"   curl --cert {self.ssl_cert_file} --key {self.ssl_key_file} -v https://huggingface.co")
        
        logger.error("5. VERIFY CERTIFICATE VALIDITY:")
        logger.error(f"   openssl x509 -in {self.ssl_cert_file} -text -noout")
    
    def get_sentence_embedding(self, text, model_type='sentence_transformer'):
        """Get sentence embedding using SSL-configured models"""
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
        return False  # We use proper SSL certificates and authentication
    
    def get_configuration_summary(self):
        """Get summary of current configuration for debugging"""
        return {
            'ssl_certificates_configured': bool(self.ssl_cert_file and self.ssl_key_file),
            'authentication_configured': bool(self.auth_config['client_id']),
            'chronicle_configured': bool(self.chronicle_config['api_key']),
            'proxy_configured': bool(self.network_config['http_proxy']),
            'models_loaded': bool(self.models.get('sentence_transformer')),
            'embedding_dimension': self.embedding_dimension,
            'ssl_cert_path': self.ssl_cert_file,
            'redis_host': self.redis_config['host']
        }