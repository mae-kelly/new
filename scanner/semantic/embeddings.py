import logging
import numpy as np
import pickle
import os
import torch.nn.functional as F
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
        
        os.makedirs(SEMANTIC_MODELS_CACHE, exist_ok=True)
        self._initialize_concept_embeddings()
    
    def _initialize_concept_embeddings(self):
        embeddings_file = os.path.join(SEMANTIC_MODELS_CACHE, 'concept_embeddings.pkl')
        
        if os.path.exists(embeddings_file):
            try:
                with open(embeddings_file, 'rb') as f:
                    self.concept_embeddings = pickle.load(f)
                return
            except Exception as e:
                logger.warning(f"Failed to load concept embeddings: {e}")
        
        self._generate_concept_embeddings()
        self._save_concept_embeddings()
    
    def _generate_concept_embeddings(self):
        for concept, data in AO1_CONCEPTS.items():
            try:
                concept_text = self._generate_concept_text(concept, data)
                contextual_text = self._generate_contextual_text(concept, data)
                
                base_embedding = self.foundation_models.get_sentence_embedding(concept_text)
                contextual_embedding = self.foundation_models.get_sentence_embedding(contextual_text, 'domain_encoder')
                domain_embedding = self.foundation_models.get_bert_embedding(concept_text)
                
                linguistic_profile = self.linguistic_resources.analyze_linguistic_features(concept_text)
                
                self.concept_embeddings[concept] = ConceptEmbedding(
                    concept_name=concept,
                    base_embedding=base_embedding,
                    contextual_embedding=contextual_embedding,
                    domain_embedding=domain_embedding,
                    linguistic_profile=linguistic_profile
                )
                
            except Exception as e:
                logger.warning(f"Failed to create embedding for {concept}: {e}")
    
    def _generate_concept_text(self, concept, data):
        base_indicators = data['content_indicators']
        
        expanded_terms = []
        for indicator in base_indicators:
            synonyms = self.linguistic_resources.get_wordnet_synonyms(indicator)
            expanded_terms.extend(synonyms)
        
        domain_terms = self._get_domain_terms(concept)
        technical_terms = self._get_technical_terms(concept)
        
        all_terms = base_indicators + expanded_terms + domain_terms + technical_terms
        return ' '.join(set(all_terms))
    
    def _generate_contextual_text(self, concept, data):
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
        indicators = ' '.join(data['content_indicators'])
        return f"{base_context} {indicators}"
    
    def _get_domain_terms(self, concept):
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
        cache_key = f"{field_name}_{len(sample_values)}_{hash(str(table_context))}"
        
        if cache_key in self.field_embeddings_cache:
            return self.field_embeddings_cache[cache_key]
        
        field_text = self._prepare_field_text(field_name, sample_values, table_context)
        embedding = self.foundation_models.get_sentence_embedding(field_text)
        
        self.field_embeddings_cache[cache_key] = embedding
        return embedding
    
    def _prepare_field_text(self, field_name, sample_values, table_context):
        cleaned_name = field_name.replace('_', ' ').lower()
        
        synonyms = []
        for word in cleaned_name.split():
            synonyms.extend(self.linguistic_resources.get_wordnet_synonyms(word))
        
        sample_text = ' '.join(str(v) for v in sample_values[:5] if v)
        context_themes = ' '.join(table_context.get('semantic_themes', []))
        
        return f"{cleaned_name} {' '.join(synonyms)} {sample_text} {context_themes}"
    
    def compute_semantic_similarity(self, field_embedding, concept_name):
        if concept_name not in self.concept_embeddings:
            return 0.0
        
        concept_emb = self.concept_embeddings[concept_name]
        
        base_sim = cosine_similarity([field_embedding], [concept_emb.base_embedding])[0][0]
        contextual_sim = cosine_similarity([field_embedding], [concept_emb.contextual_embedding])[0][0]
        domain_sim = cosine_similarity([field_embedding], [concept_emb.domain_embedding])[0][0]
        
        weighted_sim = (base_sim * 0.5 + contextual_sim * 0.3 + domain_sim * 0.2)
        return max(0.0, weighted_sim)
    
    def _save_concept_embeddings(self):
        embeddings_file = os.path.join(SEMANTIC_MODELS_CACHE, 'concept_embeddings.pkl')
        try:
            with open(embeddings_file, 'wb') as f:
                pickle.dump(self.concept_embeddings, f)
        except Exception as e:
            logger.warning(f"Failed to save concept embeddings: {e}")