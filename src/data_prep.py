import email
from bs4 import BeautifulSoup
from sklearn.base import BaseEstimator, TransformerMixin
import re
from nltk.stem import PorterStemmer
from collections import Counter 
from scipy.sparse import csr_matrix
from sklearn.pipeline import Pipeline



url_pattern = r"https?://(?:www\.)?[-a-zA-Z0-9@:%._\+~#=]{1,256}\.[a-zA-Z0-9()]{1,6}\b(?:[-a-zA-Z0-9()@:%_\+.~#?&/=]*)"
number_pattern = r'[0-9]+'
punctuation_pattern = r'[^\w\s]'
stemmer = PorterStemmer()


def get_email_content(msg) -> str:
    if not msg.is_multipart():
        content_type= msg.get_content_type()
        if content_type == "text/plain":
            return msg.get_payload()
        elif content_type== "text/html":
            html = msg.get_payload()
            return BeautifulSoup(html,"html.parser").get_text()
        else: return ""
    else:
        plain_text = None
        html_text = None
        for part in msg.walk():
            if part.is_multipart():
             continue 
            part_type = part.get_content_type()
            if part_type == "text/plain" and plain_text is None:
                plain_text = part.get_payload()
            elif part_type == "text/html" and html_text is None:
                html = part.get_payload()
                html_text= BeautifulSoup(html,"html.parser").get_text()
        if plain_text is not None:
            return plain_text
        elif html_text is not None:
            return html_text
        else:
            return ""

class EmailToWordCountTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, strip_headers=True, lowercase=True,
                 remove_punctuation=True, replace_urls=True,
                 replace_numbers=True, do_stemming=True):
        self.strip_headers = strip_headers
        self.lowercase = lowercase
        self.remove_punctuation = remove_punctuation
        self.replace_urls = replace_urls
        self.replace_numbers = replace_numbers
        self.do_stemming = do_stemming

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        contents = []
        for item in X:
            if self.input_type == "path":
                with open(item, encoding="latin1") as f:
                    msg = email.message_from_file(f)
            else:  # "text"
                msg = email.message_from_string(item)

            content = get_email_content(msg)
            if self.lowercase:
                content = content.lower()
            if self.replace_urls:
                content = re.sub(url_pattern, "URL", content)
            if self.replace_numbers:
                content = re.sub(number_pattern, "NUMBER", content)
            if self.remove_punctuation:
                content = re.sub(punctuation_pattern, " ", content)
            if self.do_stemming:
                content = ' '.join(map(stemmer.stem, content.split()))
            word_count = Counter(content.split())
            contents.append(word_count)
        return contents

class WordCounterToVectorTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, vocabulary_size=1000):
        self.vocabulary_size = vocabulary_size
    
    def fit(self, X, y=None):
        total_count = sum(X,Counter())
        most_common = total_count.most_common(self.vocabulary_size)
        self.vocabulary_=[word for word,count in most_common]
        self.vocabulary_index_ = {word: index for index , word in enumerate(self.vocabulary_) }
        return self
    
    def transform(self, X):
        rows= []
        cols = []
        data = []
        for row_index , word_count in enumerate(X):
            for word, count in word_count.items():
                if word in self.vocabulary_index_:
                    col_index = self.vocabulary_index_[word]
                    rows.append(row_index)
                    cols.append(col_index)
                    data.append(count)
        return csr_matrix((data,(rows,cols)),shape=(len(X),self.vocabulary_size))

def build_pipeline():
    return Pipeline([
        ("email_to_wordcount", EmailToWordCountTransformer()),
        ("wordcount_to_vector", WordCounterToVectorTransformer()),
    ])