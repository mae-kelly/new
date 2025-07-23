import logging
import numpy as np
import pickle
import os
from sklearn.metrics.pairwise import cosine_similarity
from ..nlp import FoundationModels, LinguisticResources
from ..config import SEMANTIC_MODELS_CACHE, AO1_CONCEPTS
from .types import ConceptEmbedding

logger = logging.getLogger(__name__)

class EmbeddingManager:
    def __init__(self):
        self.foundation_models = FoundationModels()
        self.linguistic_resources = LinguisticResources()
        self.concept_embeddings = {}
        self.field_embeddings_cache = {}
        self.embedding_dimension = self.foundation_models.embedding_dimension
        
        os.makedirs(SEMANTIC_MODELS_CACHE, exist_ok=True)
        self._initialize_concept_embeddings()
    
    def _initialize_concept_embeddings(self):
        """Initialize concept embeddings with error handling"""
        embeddings_file = os.path.join(SEMANTIC_MODELS_CACHE, 'concept_embeddings.pkl')
        
        try:
            if os.path.exists(embeddings_file):
                with open(embeddings_file, 'rb') as f:
                    loaded_embeddings = pickle.load(f)
                    # Validate dimensions
                    if self._validate_embedding_dimensions(loaded_embeddings):
                        self.concept_embeddings = loaded_embeddings
                        logger.info("Loaded existing concept embeddings")
                        return
                    else:
                        logger.warning("Existing embeddings have incompatible dimensions, regenerating")
        except Exception as e:
            logger.warning(f"Failed to load concept embeddings: {e}")
        
        self._generate_concept_embeddings()
        self._save_concept_embeddings()
    
    def _validate_embedding_dimensions(self, embeddings):
        """Check if loaded embeddings have consistent dimensions"""
        try:
            for concept, emb in embeddings.items():
                if hasattr(emb, 'base_embedding'):
                    if len(emb.base_embedding) != self.embedding_dimension:
                        return False
            return True
        except:
            return False
    
    def _generate_concept_embeddings(self):
        """Generate concept embeddings with consistent dimensions"""
        logger.info("Generating concept embeddings...")
        
        for concept, data in AO1_CONCEPTS.items():
            try:
                concept_text = self._generate_concept_text(concept, data)
                contextual_text = self._generate_contextual_text(concept, data)
                
                # All embeddings will have consistent dimensions now
                base_embedding = self.foundation_models.get_sentence_embedding(concept_text)
                contextual_embedding = self.foundation_models.get_sentence_embedding(contextual_text, 'domain_encoder')
                domain_embedding = self.foundation_models.get_bert_embedding(concept_text)
                
                # Simple linguistic profile to avoid complex dependencies
                linguistic_profile = {'concept': concept, 'indicators': data['content_indicators'][:5]}
                
                self.concept_embeddings[concept] = ConceptEmbedding(
                    concept_name=concept,
                    base_embedding=base_embedding,
                    contextual_embedding=contextual_embedding,
                    domain_embedding=domain_embedding,
                    linguistic_profile=linguistic_profile
                )
                
                logger.debug(f"Generated embedding for {concept} with dimension {len(base_embedding)}")
                
            except Exception as e:
                logger.warning(f"Failed to create embedding for {concept}: {e}")
                # Create fallback embedding
                self.concept_embeddings[concept] = ConceptEmbedding(
                    concept_name=concept,
                    base_embedding=np.zeros(self.embedding_dimension),
                    contextual_embedding=np.zeros(self.embedding_dimension),
                    domain_embedding=np.zeros(self.embedding_dimension),
                    linguistic_profile={'concept': concept}
                )
    
    def _generate_concept_text(self, concept, data):
        """Generate concept text for embedding"""
        base_indicators = data['content_indicators'][:10]  # Limit to avoid complexity
        
        # Simple synonym expansion
        expanded_terms = []
        for indicator in base_indicators[:5]:  # Limit processing
            try:
                synonyms = self.linguistic_resources.get_wordnet_synonyms(indicator)
                expanded_terms.extend(synonyms[:3])  # Limit synonyms
            except:
                pass
        
        domain_terms = self._get_domain_terms(concept)
        technical_terms = self._get_technical_terms(concept)
        
        all_terms = base_indicators + expanded_terms + domain_terms + technical_terms
        return ' '.join(set(all_terms))
    
    def _generate_contextual_text(self, concept, data):
        """Generate contextual text for embedding"""
        context_templates = {
            'asset_identity': "computing devices servers workstations endpoints infrastructure management identification",
            'network_identity': "network addressing IP addresses domains URLs connectivity infrastructure",
            'security_tools': "cybersecurity monitoring protection detection response security software agents",
            'log_sources': "logging audit trails events monitoring data security analysis operational",
            'geographic_data': "location geography regions countries datacenters facilities physical sites",
            'business_context': "organizational structure business units departments applications services ownership",
            'infrastructure_type': "cloud computing platforms deployment virtualization containers infrastructure",
            'system_classification': "system categorization device types endpoint classification server categories",
            'coverage_metrics': "measurement statistics coverage analysis visibility gaps performance metrics"
        }
        
        base_context = context_templates.get(concept, f"{concept} related terms")
        indicators = ' '.join(data['content_indicators'][:5])  # Limit indicators
        return f"{base_context} {indicators}"
    
    def _get_domain_terms(self, concept):
        """Get domain-specific terms"""
        domain_mappings = {
            'asset_identity': ['cmdb', 'inventory', 'hardware', 'infrastructure'],
            'network_identity': ['tcp', 'ip', 'dns', 'dhcp', 'routing'],
            'security_tools': ['edr', 'siem', 'ids', 'ips', 'antivirus'],
            'log_sources': ['syslog', 'eventlog', 'audit', 'monitoring'],
            'geographic_data': ['datacenter', 'region', 'zone', 'location'],
            'business_context': ['department', 'division', 'organization'],
            'infrastructure_type': ['cloud', 'onprem', 'hybrid', 'saas'],
            'system_classification': ['server', 'workstation', 'mobile', 'iot'],
            'coverage_metrics': ['kpi', 'metrics', 'dashboard', 'reporting']
        }
        return domain_mappings.get(concept, [])
    
    def _get_technical_terms(self, concept):
        """Get technical terms"""
        technical_mappings = {
            'asset_identity': ['uuid', 'guid', 'serial', 'mac'],
            'network_identity': ['ipv4', 'ipv6', 'cidr', 'vlan'],
            'security_tools': ['api', 'sdk', 'agent', 'sensor'],
            'log_sources': ['json', 'xml', 'csv', 'syslog'],
            'geographic_data': ['gps', 'coordinates', 'timezone'],
            'business_context': ['ldap', 'ad', 'rbac', 'identity'],
            'infrastructure_type': ['vm', 'container', 'kubernetes', 'docker'],
            'system_classification': ['os', 'platform', 'architecture'],
            'coverage_metrics': ['percentage', 'ratio', 'score', 'index']
        }
        return technical_mappings.get(concept, [])
    
    def get_field_embedding(self, field_name, sample_values, table_context):
        """Get field embedding with consistent dimensions"""
        cache_key = f"{field_name}_{len(sample_values)}_{hash(str(table_context))}"
        
        if cache_key in self.field_embeddings_cache:
            return self.field_embeddings_cache[cache_key]
        
        try:
            field_text = self._prepare_field_text(field_name, sample_values, table_context)
            embedding = self.foundation_models.get_sentence_embedding(field_text)
            
            # Ensure correct dimensions
            if len(embedding) != self.embedding_dimension:
                if len(embedding) < self.embedding_dimension:
                    padding = np.zeros(self.embedding_dimension - len(embedding))
                    embedding = np.concatenate([embedding, padding])
                else:
                    embedding = embedding[:self.embedding_dimension]
            
            self.field_embeddings_cache[cache_key] = embedding
            return embedding
            
        except Exception as e:
            logger.warning(f"Failed to get field embedding for {field_name}: {e}")
            return np.zeros(self.embedding_dimension)
    
    def _prepare_field_text(self, field_name, sample_values, table_context):
        """Prepare field text for embedding"""
        try:
            cleaned_name = field_name.replace('_', ' ').lower()
            
            # Simple synonym expansion with error handling
            synonyms = []
            try:
                for word in cleaned_name.split()[:3]:  # Limit words
                    word_synonyms = self.linguistic_resources.get_wordnet_synonyms(word)
                    synonyms.extend(word_synonyms[:2])  # Limit synonyms per word
            except:
                pass
            
            sample_text = ' '.join(str(v) for v in sample_values[:3] if v)  # Limit samples
            context_themes = ' '.join(table_context.get('semantic_themes', [])[:3])  # Limit themes
            
            return f"{cleaned_name} {' '.join(synonyms)} {sample_text} {context_themes}"
            
        except Exception as e:
            logger.warning(f"Error preparing field text: {e}")
            return field_name.replace('_', ' ').lower()
    
    def compute_semantic_similarity(self, field_embedding, concept_name):
        """Compute semantic similarity with dimension validation"""
        if concept_name not in self.concept_embeddings:
            return 0.0
        
        try:
            concept_emb = self.concept_embeddings[concept_name]
            
            # Ensure embeddings have same dimensions
            field_emb = field_embedding.reshape(1, -1)
            base_emb = concept_emb.base_embedding.reshape(1, -1)
            contextual_emb = concept_emb.contextual_embedding.reshape(1, -1)
            domain_emb = concept_emb.domain_embedding.reshape(1, -1)
            
            # Validate dimensions before computing similarity
            if (field_emb.shape[1] != base_emb.shape[1] or 
                field_emb.shape[1] != contextual_emb.shape[1] or 
                field_emb.shape[1] != domain_emb.shape[1]):
                logger.warning(f"Dimension mismatch for {concept_name}: field={field_emb.shape}, base={base_emb.shape}")
                return 0.0
            
            base_sim = cosine_similarity(field_emb, base_emb)[0][0]
            contextual_sim = cosine_similarity(field_emb, contextual_emb)[0][0]
            domain_sim = cosine_similarity(field_emb, domain_emb)[0][0]
            
            weighted_sim = (base_sim * 0.5 + contextual_sim * 0.3 + domain_sim * 0.2)
            return max(0.0, weighted_sim)
            
        except Exception as e:
            logger.warning(f"Failed to compute similarity for {concept_name}: {e}")
            return 0.0
    
    def _save_concept_embeddings(self):
        """Save concept embeddings"""
        embeddings_file = os.path.join(SEMANTIC_MODELS_CACHE, 'concept_embeddings.pkl')
        try:
            with open(embeddings_file, 'wb') as f:
                pickle.dump(self.concept_embeddings, f)
            logger.info("Concept embeddings saved successfully")
        except Exception as e:
            logger.warning(f"Failed to save concept embeddings: {e}")