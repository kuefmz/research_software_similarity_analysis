import argparse
from typing import Iterable, List
import logthis
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re
import numpy as np
import inflect
import contractions
from bs4 import BeautifulSoup
import re, unicodedata
from nltk import word_tokenize
from nltk.stem import LancasterStemmer, WordNetLemmatizer
from nltk.stem.api import StemmerI

class Preprocessor:
    def __init__(self, data: pd.DataFrame, TEXT: str = 'Text') -> None:
        self.data = data
        self.TEXT = TEXT

    def denoise_text(self, text: str) -> str:
        soup = BeautifulSoup(text, "html.parser")
        text = soup.get_text()
        text = contractions.fix(text)
        text.replace("""404: Not Found""", '')
        return text

    def remove_stop_words(self, text: str) -> List[str]:
        stop_words = stopwords.words('english')
        stop_words += ['zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine', 'ten', 'hundred', 'thousand', 'and']
        stop_words += ['network', 'install', 'run', 'file', 'use', 'result', 'paper', 'python', 'using', 'code', 'model', 'train', 'implementation', 'use']
        stop_words += ['data', 'dataset', 'example', 'build', 'learn', 'download', 'obj']
        return [word for word in text if word not in stop_words]

    def remove_codeblocks(self, text: str) -> str:
        return re.sub('```.*?```', ' ', text)

    def remove_punctuation(self, text: str) -> str:
        res = re.sub(r'[^\w\s]|\_', ' ', text)
        return res

    def remove_non_ascii(self, words: Iterable[str]) -> List[str]:
        new_words = []
        for word in words:
            new_word = unicodedata.normalize('NFKD', word).encode('ascii', 'ignore').decode('utf-8', 'ignore')
            new_words.append(new_word)
        return new_words

    def replace_numbers(self, words: Iterable[str]) -> List[str]:
        p = inflect.engine()
        new_words = []
        for word in words:
            if word.isdigit():
                new_word = p.number_to_words(word)
                new_words += new_word.split(' ')
            else:
                new_words.append(word)
        return new_words

    def stemming(self, text: Iterable[str], porter_stemmer: StemmerI) -> List[str]:
        stem_text = [porter_stemmer.stem(word) for word in text]
        return stem_text

    def stem_words(self, words: Iterable[str]) -> List[str]:
        stemmer = LancasterStemmer()
        stems = []
        for word in words:
            stem = stemmer.stem(word)
            stems.append(stem)
        return stems

    def lemmatize_verbs(self, words: Iterable[str]) -> List[str]:
        lemmatizer = WordNetLemmatizer()
        lemmas = []
        for word in words:
            lemma = lemmatizer.lemmatize(word, pos='v')
            lemmas.append(lemma)
        return lemmas

    def lemmatize_nouns(self, words: Iterable[str]) -> List[str]:
        lemmatizer = WordNetLemmatizer()
        lemmas = []
        for word in words:
            lemma = lemmatizer.lemmatize(word, pos='n')
            lemmas.append(lemma)
        return lemmas

    def lemmatize_adjectives(self, words: Iterable[str]) -> List[str]:
        lemmatizer = WordNetLemmatizer()
        lemmas = []
        for word in words:
            lemma = lemmatizer.lemmatize(word, pos='a')
            lemmas.append(lemma)
        return lemmas

    def remove_short_and_number_words(self, text: Iterable[str]) -> List[str]:
        res = [word for word in text if word.isdigit() == False and len(word) > 2]
        return res

    def remove_links(self, text: str) -> str:
        regex = r"(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:'\".,<>?«»“”‘’]))"
        return (re.sub(regex, '', text))

    def remove_links2(self, text: str) -> str:
        return ' '.join([token for token in text.split(' ') if 'http' not in token])

    def run(self):
        pipeline = {
            'remove codeblocks': lambda x: self.remove_codeblocks(x),
            'remove links': lambda x : self.remove_links2(x),
            'remove tags': lambda x : self.denoise_text(x),
            'remove punctuations': lambda x: self.remove_punctuation(x),
            'transform to lowercase': lambda x: x.lower(),
            'remove non-ascii characters': lambda x : self.remove_non_ascii(word_tokenize(x)),
            'lemmatize verbs': lambda x : self.lemmatize_verbs(x),
            'lemmatize nouns': lambda x : self.lemmatize_nouns(x),
            'lemmatize adjectives': lambda x : self.lemmatize_adjectives(x),
            'remove stop_words': lambda x : self.remove_stop_words(x),
            'remove tokens only containing numbers or two char': lambda x : self.remove_short_and_number_words(x),
            'join tokens': lambda x: ' '.join(x),
        }

        for key, val in pipeline.items():
            logthis.say(f'Preprocessing: Process name: "{key}".')
            self.data[self.TEXT] = self.data[self.TEXT].apply(val)

        logthis.say("Preprocessing: drop empty rows.")
        self.data.drop(self.data[self.data[self.TEXT] == np.nan].index, inplace=True)
        self.data.drop(self.data[self.data[self.TEXT] == ''].index, inplace=True)


def preprocess_file(filename: str) -> None:
    """
    Reads the data from the given JSON file, runs preprocessing methods on the data, 
    and saves the preprocessed data next to the original file with "_preprocessed.json" suffix.

    Params
    ---------		
    filename: (str) Path to the input JSON file.
    """
    logthis.say('Preprocessing starts.')
    df = pd.read_json(filename)
    Preprocessor(df).run()
    df.to_json(filename.replace('.json', '_preprocessed.json'), orient='records', lines=True)
    logthis.say('Preprocessing done.')
