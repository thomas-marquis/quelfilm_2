import nltk
from nltk.stem.snowball import FrenchStemmer

from quelfilm import utils
from quelfilm.nlp.load import DataLoader, InputData


class Preprocessing:
    def __init__(self, data_loader: DataLoader):
        nltk.download('french')
        self.tokenizer = nltk.RegexpTokenizer(r'\w+')
        self.stop_words = nltk.corpus.stopwords.words('french')
        self.stemmer = FrenchStemmer()
        
        # TODO compose it
        inputs: InputData = data_loader.load()
        data = inputs.examples
        self.responses = inputs.responses
        data = self.to_lower_case_all(data)
        data = self.tokenize_all_examples(data)
        data = self.remove_stop_words_for_all(data)
        data = self.lemmatize_all(data)
        self.data = data
        
    def to_lower_case_one(self, example: str):
        return example.lower()
        
    def to_lower_case_all(self, data):
        to_lower_case = lambda examples: [self.to_lower_case_one(ex) for ex in examples]
        return utils.apply_for_each_key(data, to_lower_case)
    
    def tokenize_one_example(self, example):
        return self.tokenizer.tokenize(example)
    
    def tokenize_all_examples(self, data):
        tokenize = lambda data_list: [self.tokenize_one_example(d) for d in data_list]
        return utils.apply_for_each_key(data, tokenize)
    
    def remove_stop_words(self, example):
        return [w for w in example if not w in self.stop_words]
    
    def remove_stop_words_for_all(self, data):
        remove_sw = lambda examples: [self.remove_stop_words(ex) for ex in examples]
        return utils.apply_for_each_key(data, remove_sw)
    
    def lemmatize(self, example):
        return [self.stemmer.stem(w) for w in example]
    
    def lemmatize_all(self, data):
        get_lems = lambda examples: [self.lemmatize(ex) for ex in examples]
        return utils.apply_for_each_key(data, get_lems)
    
    def process_sentence(self, sentence):
        # TODO compose
        data = self.to_lower_case_one(sentence)
        data = self.tokenize_one_example(data)
        data = self.remove_stop_words(data)
        data = self.lemmatize(data)
        return data
