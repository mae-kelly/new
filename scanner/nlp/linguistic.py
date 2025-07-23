import logging
import nltk
import spacy
from nltk.corpus import wordnet, stopwords
from nltk.stem import WordNetLemmatizer, PorterStemmer
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.tag import pos_tag
from spacy.matcher import Matcher, PhraseMatcher
import textstat
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

logger = logging.getLogger(__name__)

class LinguisticResources:
    def __init__(self):
        self._download_nltk_resources()
        self._initialize_spacy()
        self._initialize_tools()
    
    def _download_nltk_resources(self):
        resources = [
            'punkt', 'stopwords', 'wordnet', 'omw-1.4', 'averaged_perceptron_tagger',
            'maxent_ne_chunker', 'words', 'brown', 'reuters'
        ]
        
        for resource in resources:
            try:
                nltk.download(resource, quiet=True)
            except:
                pass
    
    def _initialize_spacy(self):
        self.spacy_models = {}
        models = ['en_core_web_lg', 'en_core_web_md', 'en_core_web_sm']
        
        for model_name in models:
            try:
                self.spacy_models[model_name] = spacy.load(model_name)
                break
            except:
                continue
        
        if self.spacy_models:
            self.nlp = list(self.spacy_models.values())[0]
            self.matcher = Matcher(self.nlp.vocab)
            self.phrase_matcher = PhraseMatcher(self.nlp.vocab, attr="LOWER")
        else:
            self.nlp = None
            logger.warning("No spaCy models available")
    
    def _initialize_tools(self):
        self.lemmatizer = WordNetLemmatizer()
        self.stemmer = PorterStemmer()
        self.sentiment_analyzer = SentimentIntensityAnalyzer()
    
    def get_wordnet_synonyms(self, word):
        synonyms = []
        try:
            for syn in wordnet.synsets(word):
                for lemma in syn.lemmas():
                    synonym = lemma.name().replace('_', ' ')
                    if synonym != word and len(synonym) > 2:
                        synonyms.append(synonym)
        except Exception:
            pass
        return list(set(synonyms))[:5]
    
    def analyze_linguistic_features(self, text):
        features = {
            'pos_distribution': {},
            'dependency_patterns': [],
            'entities': [],
            'readability_scores': {},
            'sentiment': {}
        }
        
        try:
            if self.nlp:
                doc = self.nlp(text)
                
                pos_counts = {}
                for token in doc:
                    pos_counts[token.pos_] = pos_counts.get(token.pos_, 0) + 1
                features['pos_distribution'] = pos_counts
                
                for token in doc:
                    if token.dep_ != 'ROOT':
                        features['dependency_patterns'].append((token.text, token.dep_, token.head.text))
                
                features['entities'] = [(ent.text, ent.label_) for ent in doc.ents]
            
            features['readability_scores'] = {
                'flesch_kincaid': textstat.flesch_kincaid_grade(text),
                'gunning_fog': textstat.gunning_fog(text),
                'coleman_liau': textstat.coleman_liau_index(text)
            }
            
            features['sentiment'] = self.sentiment_analyzer.polarity_scores(text)
            
        except Exception as e:
            logger.debug(f"Linguistic analysis error: {e}")
        
        return features
    
    def lemmatize_text(self, text):
        try:
            tokens = word_tokenize(text.lower())
            return [self.lemmatizer.lemmatize(token) for token in tokens if len(token) > 2]
        except Exception:
            return text.split()
    
    def stem_text(self, text):
        try:
            tokens = word_tokenize(text.lower())
            return [self.stemmer.stem(token) for token in tokens]
        except Exception:
            return text.split()