import re

import utils.io_utils as iu


class NltkFactory:
    lemm = stem = stop = None

    @staticmethod
    def lemmatize(word):
        if not NltkFactory.lemm:
            from nltk.stem import WordNetLemmatizer
            NltkFactory.lemm = WordNetLemmatizer()
        return NltkFactory.lemm.lemmatize(word)

    @staticmethod
    def stemming(word):
        if not NltkFactory.stem:
            from nltk.stem.snowball import EnglishStemmer
            NltkFactory.stem = EnglishStemmer()
        return NltkFactory.stem.stem(word)

    @staticmethod
    def get_stop_words():
        if not NltkFactory.stop:
            from nltk.corpus import stopwords
            NltkFactory.stop = set(stopwords.words('english'))
        return NltkFactory.stop


lemmatize = NltkFactory.lemmatize
stemming = NltkFactory.stemming

nltk_stop_words = None  # NltkFactory.get_stop_words()
# file_path = os.path.abspath(os.path.dirname(__file__))
# my_stop_words_file = os.path.join(file_path, "stopwords.txt")
my_stop_words_file = iu.join(iu.parent_name(__file__), "stopwords.txt")
my_stop_words = set(iu.read_lines(my_stop_words_file, newline='\r\n'))
tokenize_pattern = r"[a-zA-Z0-9]+(?:[_-][a-zA-Z0-9]+)*"


def tokenize(text, pattern):
    return re.findall(pattern, text)


def has_enough_alpha(text, threshold):
    text = re.sub('\s', '', text)
    if len(text) == 0:
        return False
    alphas = re.findall('[a-zA-Z]', text)
    return len(alphas) / len(text) >= threshold


def is_valid_word(word):
    not_all_repeat = len(set(list(word))) > 2
    not_stop_word = word not in my_stop_words
    return not_all_repeat and not_stop_word
