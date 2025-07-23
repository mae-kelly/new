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
        self._load_environment_config()
        self._configure_corporate_network()
        self._initialize_models()
    
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
            'ca_bundle': os.getenv('REQUESTS_CA_BUNDLE') or os.getenv('CURL_CA_BUNDLE') or os.getenv('SSL_CERT_FILE')
        }
        
        # Redis configuration (if needed)
        self.redis_config = {
            'host': os.getenv('REDIS_HOST'),
            'port': int(os.getenv('REDIS_PORT', 6379)),
            'db': int(os.getenv('REDIS_DB', 0)),
            'password': os.getenv('REDIS_PASSWORD')
        }
        
        # Log configuration status
        self._log_config_status()
    
    def _log_config_status(self):
        """Log configuration status without exposing secrets"""
        logger.info("Loading environment configuration...")
        
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
        
        ca_configured = bool(self.network_config['ca_bundle'])
        logger.info(f"✅ CA bundle: {'CONFIGURED' if ca_configured else 'SYSTEM DEFAULT'}")
    
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
            
            # Configure SSL/TLS
            self._setup_ssl_from_env()
            
            # Setup authenticated session
            self._setup_authenticated_session()
            
        except Exception as e:
            logger.error(f"Corporate network configuration failed: {e}")
            raise
    
    def _setup_ssl_from_env(self):
        """Setup SSL configuration from environment variables"""
        try:
            # Use CA bundle from environment if specified
            if self.network_config['ca_bundle'] and os.path.exists(self.network_config['ca_bundle']):
                ca_bundle = self.network_config['ca_bundle']
                logger.info(f"Using CA bundle from environment: {ca_bundle}")
            else:
                # Fall back to system locations
                ca_locations = [
                    '/etc/ssl/certs/ca-certificates.crt',
                    '/etc/ssl/certs/ca-bundle.crt',
                    '/etc/pki/tls/certs/ca-bundle.crt',
                    '/usr/local/share/certs/ca-root-nss.crt',
                    '/etc/ssl/cert.pem',
                    certifi.where()
                ]
                
                ca_bundle = None
                for location in ca_locations:
                    if os.path.exists(location):
                        ca_bundle = location
                        break
                
                if ca_bundle:
                    logger.info(f"Using system CA bundle: {ca_bundle}")
                else:
                    logger.warning("No CA bundle found - using defaults")
            
            # Configure SSL context
            if ca_bundle:
                ssl_context = ssl.create_default_context(cafile=ca_bundle)
                ssl._create_default_https_context = lambda: ssl_context
                self.ca_bundle = ca_bundle
            else:
                self.ca_bundle = None
            
        except Exception as e:
            logger.warning(f"SSL configuration failed: {e}")
            self.ca_bundle = None
    
    def _setup_authenticated_session(self):
        """Setup requests session with authentication and corporate settings"""
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
            
            # Configure SSL verification
            if self.ca_bundle:
                self.session.verify = self.ca_bundle
            
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
            
            # Add User-Agent
            self.session.headers.update({
                'User-Agent': 'AO1-Scanner/2.0 (Corporate-Environment)'
            })
            
            logger.info("✅ Authenticated session configured")
            
            # Test connectivity
            self._test_corporate_connectivity()
            
            # Patch transformers to use authenticated session
            self._patch_transformers_with_auth()
            
        except Exception as e:
            logger.error(f"Authenticated session setup failed: {e}")
            raise
    
    def _test_corporate_connectivity(self):
        """Test connectivity through corporate network"""
        try:
            # Test basic connectivity
            test_urls = ['https://httpbin.org/status/200']
            
            if self.chronicle_config['endpoint']:
                test_urls.append(self.chronicle_config['endpoint'])
            
            for url in test_urls:
                try:
                    response = self.session.head(url, timeout=10)
                    if response.status_code < 400:
                        logger.info(f"✅ Connectivity test passed: {url}")
                    else:
                        logger.warning(f"⚠️  Connectivity test returned {response.status_code}: {url}")
                except Exception as e:
                    logger.warning(f"⚠️  Connectivity test failed for {url}: {e}")
            
        except Exception as e:
            logger.debug(f"Connectivity test failed: {e}")
    
    def _patch_transformers_with_auth(self):
        """Patch transformers library to use authenticated corporate session"""
        try:
            import sentence_transformers.util
            
            # Store original function
            original_http_get = getattr(sentence_transformers.util, 'http_get', None)
            
            def authenticated_http_get(url, temp_file, proxies=None, resume_size=0, headers=None):
                """Download using authenticated corporate session"""
                try:
                    logger.info(f"Downloading via corporate network: {url}")
                    
                    # Use corporate session for downloads
                    download_headers = headers or {}
                    if resume_size > 0:
                        download_headers['Range'] = f'bytes={resume_size}-'
                    
                    # Add any additional authentication if needed
                    if self.auth_config['client_id']:
                        download_headers['X-Client-ID'] = self.auth_config['client_id']
                    
                    response = self.session.get(url, headers=download_headers, stream=True)
                    response.raise_for_status()
                    
                    # Write to file
                    with open(temp_file, 'ab' if resume_size > 0 else 'wb') as f:
                        for chunk in response.iter_content(chunk_size=8192):
                            if chunk:
                                f.write(chunk)
                    
                    logger.info("✅ Download completed via corporate network")
                    return temp_file
                    
                except Exception as e:
                    logger.error(f"Corporate download failed: {e}")
                    
                    # Fallback to original function
                    if original_http_get:
                        logger.info("Attempting fallback download...")
                        return original_http_get(url, temp_file, proxies, resume_size, headers)
                    raise
            
            # Apply patch
            if hasattr(sentence_transformers.util, 'http_get'):
                sentence_transformers.util.http_get = authenticated_http_get
                logger.info("✅ Transformers patched for corporate authentication")
            
        except ImportError:
            logger.debug("Transformers not imported yet - will patch on demand")
        except Exception as e:
            logger.warning(f"Failed to patch transformers: {e}")
    
    def _initialize_models(self):
        """Initialize models with corporate authentication"""
        try:
            logger.info("Initializing sentence transformers with corporate authentication...")
            
            # Import after corporate configuration
            from sentence_transformers import SentenceTransformer
            
            # Ensure patches are applied
            self._patch_transformers_with_auth()
            
            # Download models using corporate session
            logger.info("Downloading models via corporate network...")
            self.models['sentence_transformer'] = SentenceTransformer(
                'all-MiniLM-L6-v2',
                cache_folder=os.path.expanduser('~/.cache/sentence_transformers_corporate')
            )
            
            # Use same model for consistency
            self.models['domain_encoder'] = self.models['sentence_transformer']
            self.models['code_encoder'] = self.models['sentence_transformer']
            
            # Test model functionality
            test_embedding = self.models['sentence_transformer'].encode(["corporate network test"])
            self.embedding_dimension = test_embedding.shape[1]
            
            logger.info(f"✅ Models initialized successfully!")
            logger.info(f"   Model: all-MiniLM-L6-v2")
            logger.info(f"   Embedding dimension: {self.embedding_dimension}")
            logger.info(f"   Corporate authentication: {'ENABLED' if self.chronicle_config['api_key'] else 'DISABLED'}")
            
        except Exception as e:
            logger.error(f"❌ Model initialization failed: {e}")
            logger.error("Possible issues:")
            logger.error("  • Corporate firewall blocking model downloads")
            logger.error("  • Authentication credentials invalid or expired")
            logger.error("  • Proxy configuration incorrect")
            logger.error("  • Network connectivity issues")
            
            # Provide environment-specific troubleshooting
            self._provide_env_troubleshooting()
            raise
    
    def _provide_env_troubleshooting(self):
        """Provide troubleshooting based on environment configuration"""
        logger.error("\n🔧 TROUBLESHOOTING CHECKLIST:")
        
        # Check required environment variables
        missing_vars = []
        if not self.network_config['http_proxy']:
            missing_vars.append('HTTP_PROXY')
        if not self.network_config['https_proxy']:
            missing_vars.append('HTTPS_PROXY')
        
        if missing_vars:
            logger.error(f"1. MISSING ENVIRONMENT VARIABLES: {', '.join(missing_vars)}")
        else:
            logger.error("1. PROXY CONFIGURATION: ✅ CONFIGURED")
        
        # Check authentication
        if not self.auth_config['client_id']:
            logger.error("2. AUTHENTICATION: ⚠️  CLIENT_ID not set")
        else:
            logger.error("2. AUTHENTICATION: ✅ CONFIGURED")
        
        # Check Chronicle
        if not self.chronicle_config['api_key']:
            logger.error("3. CHRONICLE API: ⚠️  CHRONICLE_API_KEY not set")
        else:
            logger.error("3. CHRONICLE API: ✅ CONFIGURED")
        
        logger.error("4. VERIFY NETWORK ACCESS:")
        logger.error("   curl -v https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2")
        
        logger.error("5. CONTACT IT TO WHITELIST:")
        logger.error("   • huggingface.co")
        logger.error("   • cdn-lfs.huggingface.co")
        logger.error("   • files.pythonhosted.org")
    
    def get_sentence_embedding(self, text, model_type='sentence_transformer'):
        """Get sentence embedding using corporate-configured models"""
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
        return False  # We use proper corporate authentication
    
    def get_configuration_summary(self):
        """Get summary of current configuration for debugging"""
        return {
            'authentication_configured': bool(self.auth_config['client_id']),
            'chronicle_configured': bool(self.chronicle_config['api_key']),
            'proxy_configured': bool(self.network_config['http_proxy']),
            'ca_bundle_configured': bool(self.ca_bundle),
            'models_loaded': bool(self.models.get('sentence_transformer')),
            'embedding_dimension': self.embedding_dimension
        }