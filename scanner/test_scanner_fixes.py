#!/usr/bin/env python3

import os
import sys
import logging

# Add the scanner directory to the path
sys.path.insert(0, os.path.dirname(__file__))

def test_components():
    """Test individual components to verify fixes"""
    
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    print("Testing AO1 Scanner Components...")
    print("="*50)
    
    # Test 1: Foundation Models
    try:
        from scanner.nlp.foundation import FoundationModels
        models = FoundationModels()
        
        # Test embedding generation
        test_embedding = models.get_sentence_embedding("test hostname device")
        print(f"✅ Foundation Models: {test_embedding.shape} dimensions")
        
    except Exception as e:
        print(f"❌ Foundation Models failed: {e}")
    
    # Test 2: Embedding Manager
    try:
        from scanner.semantic.embeddings import EmbeddingManager
        emb_manager = EmbeddingManager()
        
        # Test field embedding
        field_emb = emb_manager.get_field_embedding("hostname", ["server01", "workstation02"], {})
        print(f"✅ Embedding Manager: {field_emb.shape} dimensions")
        
        # Test semantic similarity
        similarity = emb_manager.compute_semantic_similarity(field_emb, "asset_identity")
        print(f"✅ Semantic Similarity: {similarity:.3f}")
        
    except Exception as e:
        print(f"❌ Embedding Manager failed: {e}")
    
    # Test 3: Semantic Analyzer
    try:
        from scanner.semantic_analyzer import AdvancedSemanticAnalyzer
        analyzer = AdvancedSemanticAnalyzer()
        
        # Create mock schema field
        class MockField:
            def __init__(self, name, field_type):
                self.name = name
                self.field_type = field_type
        
        # Test analysis
        mock_data = [
            {"hostname": "server01.company.com", "ip_address": "192.168.1.10"},
            {"hostname": "workstation02.local", "ip_address": "192.168.1.11"}
        ]
        
        mock_schema = [
            MockField("hostname", "STRING"),
            MockField("ip_address", "STRING")
        ]
        
        analyses = analyzer.analyze_batch_fields(mock_data, mock_schema)
        print(f"✅ Semantic Analyzer: Analyzed {len(analyses)} fields")
        
        for analysis in analyses:
            print(f"   - {analysis.field_name}: {analysis.ao1_category} ({analysis.confidence_score:.3f})")
        
    except Exception as e:
        print(f"❌ Semantic Analyzer failed: {e}")
    
    # Test 4: BigQuery Connection (if credentials available)
    try:
        service_account_file = "gcp_prod_key.json"
        if os.path.exists(service_account_file):
            from scanner.connection import BigQueryConnection
            conn = BigQueryConnection(service_account_file)
            datasets = conn.list_datasets()
            print(f"✅ BigQuery Connection: Found {len(datasets)} datasets")
        else:
            print("⚠️  BigQuery Connection: No credentials file found (expected)")
            
    except Exception as e:
        print(f"❌ BigQuery Connection failed: {e}")
    
    print("="*50)
    print("Component testing complete!")

if __name__ == "__main__":
    test_components()