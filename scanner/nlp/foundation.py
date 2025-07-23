import logging
import torch
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import (
    BertTokenizer, BertModel, RobertaTokenizer, RobertaModel,
    T5Tokenizer, T5ForConditionalGeneration, GPT2LMHeadModel, GPT2Tokenizer
)

logger = logging.getLogger(__name__)

class FoundationModels:
    def __init__(self):
        self.models = {}
        self.tokenizers = {}
        self._initialize_models()
    
    def _initialize_models(self):
        try:
            self.models['sentence_transformer'] = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
            self.models['domain_encoder'] = SentenceTransformer('sentence-transformers/all-roberta-large-v1')
            self.models['code_encoder'] = SentenceTransformer('microsoft/codebert-base')
            
            self.models['bert'] = BertModel.from_pretrained('bert-large-uncased')
            self.tokenizers['bert'] = BertTokenizer.from_pretrained('bert-large-uncased')
            
            self.models['roberta'] = RobertaModel.from_pretrained('roberta-large')
            self.tokenizers['roberta'] = RobertaTokenizer.from_pretrained('roberta-large')
            
            self.models['t5'] = T5ForConditionalGeneration.from_pretrained('t5-base')
            self.tokenizers['t5'] = T5Tokenizer.from_pretrained('t5-base')
            
            self.models['gpt2'] = GPT2LMHeadModel.from_pretrained('gpt2')
            self.tokenizers['gpt2'] = GPT2Tokenizer.from_pretrained('gpt2')
            self.tokenizers['gpt2'].pad_token = self.tokenizers['gpt2'].eos_token
            
            logger.info("Foundation models initialized successfully")
            
        except Exception as e:
            logger.warning(f"Failed to initialize foundation models: {e}")
            self._initialize_fallback_models()
    
    def _initialize_fallback_models(self):
        try:
            self.models['sentence_transformer'] = SentenceTransformer('all-MiniLM-L6-v2')
            self.models['domain_encoder'] = self.models['sentence_transformer']
            self.models['code_encoder'] = self.models['sentence_transformer']
            logger.info("Fallback models initialized")
        except Exception as e:
            logger.error(f"Failed to initialize fallback models: {e}")
    
    def get_sentence_embedding(self, text, model_type='sentence_transformer'):
        try:
            model = self.models.get(model_type)
            if model:
                return model.encode([text])[0]
            else:
                return np.zeros(768)
        except Exception as e:
            logger.warning(f"Failed to get embedding: {e}")
            return np.zeros(768)
    
    def get_bert_embedding(self, text):
        try:
            tokenizer = self.tokenizers['bert']
            model = self.models['bert']
            inputs = tokenizer(text, return_tensors='pt', truncation=True, max_length=512)
            with torch.no_grad():
                outputs = model(**inputs)
            return outputs.last_hidden_state.mean(dim=1).squeeze().numpy()
        except Exception as e:
            logger.warning(f"Failed to get BERT embedding: {e}")
            return np.zeros(768)
    
    def get_roberta_embedding(self, text):
        try:
            tokenizer = self.tokenizers['roberta']
            model = self.models['roberta']
            inputs = tokenizer(text, return_tensors='pt', truncation=True, max_length=512)
            with torch.no_grad():
                outputs = model(**inputs)
            return outputs.last_hidden_state.mean(dim=1).squeeze().numpy()
        except Exception as e:
            logger.warning(f"Failed to get RoBERTa embedding: {e}")
            return np.zeros(1024)