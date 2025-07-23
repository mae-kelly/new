import logging
from transformers import pipeline

logger = logging.getLogger(__name__)

class SpecializedModels:
    def __init__(self):
        self.pipelines = {}
        self._initialize_pipelines()
    
    def _initialize_pipelines(self):
        try:
            self.pipelines['ner_general'] = pipeline(
                "ner", 
                model="dbmdz/bert-large-cased-finetuned-conll03-english", 
                aggregation_strategy="simple"
            )
            
            self.pipelines['ner_technical'] = pipeline(
                "ner", 
                model="allenai/scibert_scivocab_uncased", 
                aggregation_strategy="simple"
            )
            
            self.pipelines['zero_shot'] = pipeline(
                "zero-shot-classification", 
                model="facebook/bart-large-mnli"
            )
            
            self.pipelines['qa'] = pipeline(
                "question-answering", 
                model="deepset/roberta-base-squad2"
            )
            
            self.pipelines['text_generation'] = pipeline(
                "text-generation", 
                model="microsoft/DialoGPT-medium", 
                max_length=100
            )
            
            logger.info("Specialized models initialized successfully")
            
        except Exception as e:
            logger.warning(f"Failed to initialize specialized models: {e}")
            self._initialize_basic_pipelines()
    
    def _initialize_basic_pipelines(self):
        try:
            self.pipelines['ner_general'] = pipeline("ner", aggregation_strategy="simple")
            logger.info("Basic pipelines initialized")
        except Exception as e:
            logger.error(f"Failed to initialize basic pipelines: {e}")
    
    def analyze_entities(self, text, model_type='general'):
        try:
            pipeline_key = f'ner_{model_type}'
            if pipeline_key in self.pipelines:
                return self.pipelines[pipeline_key](text)
            else:
                return []
        except Exception as e:
            logger.warning(f"Entity analysis failed: {e}")
            return []
    
    def classify_zero_shot(self, text, candidate_labels):
        try:
            if 'zero_shot' in self.pipelines:
                return self.pipelines['zero_shot'](text, candidate_labels)
            else:
                return None
        except Exception as e:
            logger.warning(f"Zero-shot classification failed: {e}")
            return None
    
    def answer_question(self, question, context):
        try:
            if 'qa' in self.pipelines:
                return self.pipelines['qa'](question=question, context=context)
            else:
                return None
        except Exception as e:
            logger.warning(f"Question answering failed: {e}")
            return None
    
    def generate_text(self, prompt, max_length=50):
        try:
            if 'text_generation' in self.pipelines:
                return self.pipelines['text_generation'](prompt, max_length=max_length)
            else:
                return None
        except Exception as e:
            logger.warning(f"Text generation failed: {e}")
            return None