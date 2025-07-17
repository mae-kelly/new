import numpy as np
import re
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from textdistance import levenshtein, jaro_winkler

class AdvancedFeatureEngine:
    def __init__(self):
        self.char_vectorizer = TfidfVectorizer(analyzer='char', ngram_range=(2,4), max_features=200)
        self.word_vectorizer = TfidfVectorizer(analyzer='word', ngram_range=(1,3), max_features=300)
        
    def extract_linguistic_features(self, texts):
        features = []
        
        for text in texts:
            text_str = str(text)
            
            # Linguistic patterns
            feat = [
                # Entropy (randomness)
                self.calculate_entropy(text_str),
                
                # Pattern regularity
                len(set(text_str)) / max(len(text_str), 1),
                
                # Consecutive character patterns
                max([len(list(g)) for k, g in __import__('itertools').groupby(text_str)] + [0]),
                
                # Vowel/consonant ratio
                vowels = sum(1 for c in text_str.lower() if c in 'aeiou')
                consonants = sum(1 for c in text_str.lower() if c.isalpha() and c not in 'aeiou')
                vowels / max(consonants + vowels, 1),
                
                # Special character density
                sum(1 for c in text_str if not c.isalnum()) / max(len(text_str), 1),
                
                # Capitalization patterns
                sum(1 for i, c in enumerate(text_str[:-1]) if c.islower() and text_str[i+1].isupper()),
                
                # Number sequences
                len(re.findall(r'\d+', text_str)),
                max([len(m) for m in re.findall(r'\d+', text_str)] + [0]),
                
                # Common IT patterns
                bool(re.search(r'[a-z]+-[a-z]+-\d+', text_str.lower())),
                bool(re.search(r'[A-Z]{2,4}-[A-Z]+-\d+', text_str)),
                bool(re.search(r'v\d+\.\d+(\.\d+)?', text_str.lower())),
                bool(re.search(r'\b\d{1,3}(\.\d{1,3}){3}\b', text_str)),
                bool(re.search(r'[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}', text_str.lower())),
                
                # Semantic indicators
                any(word in text_str.lower() for word in ['srv', 'server', 'host', 'node']),
                any(word in text_str.lower() for word in ['prod', 'production', 'live']),
                any(word in text_str.lower() for word in ['test', 'dev', 'staging']),
                any(word in text_str.lower() for word in ['corp', 'company',
