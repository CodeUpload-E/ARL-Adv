import utils.io_utils as iu

K_ID, K_FREQ = 1, 2


class IdFreqDict:
    def __init__(self):
        self._word2id = dict()
        self._id2word = dict()
        self._freq_sum = 0
        self._word_freq_enum = None

    def clear(self):
        self._word2id.clear()
        self._id2word.clear()
        self._freq_sum = 0
        self._word_freq_enum = None

    def __contains__(self, word):
        return self.has_word(word)

    def has_word(self, word):
        return word in self._word2id

    def vocabulary(self):
        return list(self._word2id.keys())

    def vocabulary_size(self):
        return self._word2id.__len__()

    def word2id(self, word):
        return self._word2id[word][K_ID]

    def id2word(self, _id):
        return self._id2word[_id]

    def freq_of_word(self, word):
        return self._word2id[word][K_FREQ]

    def get_freq_sum(self):
        return self._freq_sum

    def get_word2id(self):
        return dict((w, self.word2id(w)) for w in self.vocabulary())

    def calc_freq_sum(self):
        self._freq_sum = sum([freq for word, freq in self.word_freq_enumerate(newest=True)])
        return self._freq_sum

    def reset_id(self):
        for idx, word in enumerate(sorted(self.vocabulary())):
            self._word2id[word][K_ID] = idx
            self._id2word[idx] = word

    def drop_word(self, word):
        if self.has_word(word):
            self._freq_sum -= self.freq_of_word(word)
            return self._word2id.pop(word)
        return None

    def drop_words(self, words):
        for word in words:
            self.drop_word(word)

    def reserve_words(self, condition):
        if condition is None:
            return
        elif callable(condition):
            cond = condition
        elif isinstance(condition, int):
            def cond(_, freq):
                return freq >= condition
        else:
            raise TypeError('condition type invalid: {}'.format(type(condition)))

        for word in self.vocabulary():
            if not cond(word, self.freq_of_word(word)):
                self.drop_word(word)
        self.reset_id()

    def count_words(self, words):
        for word in words:
            self.count_word(word)

    def count_word(self, word, freq=None):
        freq = 1 if freq is None else freq
        if self.has_word(word):
            self._word2id[word][K_FREQ] += freq
        else:
            self._word2id[word] = {K_FREQ: freq}
        self._freq_sum += freq

    def uncount_word(self, word, freq=None):
        if not self.has_word(word):
            print('non existing word "{}"'.format(word))
            raise ValueError('non existing word "{}"'.format(word))
        freq = 1 if freq is None else freq
        self._word2id[word][K_FREQ] -= freq
        self._freq_sum -= freq
        post_freq = self.freq_of_word(word)
        if post_freq == 0:
            self.drop_word(word)
        elif post_freq < 0:
            raise ValueError('word {} freq {} less than 0'.format(word, post_freq))

    def word_freq_enumerate(self, newest):
        if self._word_freq_enum is None or newest:
            self._word_freq_enum = [(word, self._word2id[word][K_FREQ]) for word in
                                    self.vocabulary()]
        return self._word_freq_enum

    def most_common(self, k=None, newest=True):
        word_freq_enum = self.word_freq_enumerate(newest)
        word_freq_enum = sorted(word_freq_enum, key=lambda item: item[1], reverse=True)
        return word_freq_enum[:k] if k is not None else word_freq_enum

    def merge_freq_from(self, other_ifd, newest=True):
        pre_freq = self._freq_sum
        for other_word, other_freq in other_ifd.word_freq_enumerate(newest):
            self.count_word(other_word, other_freq)
        return self._freq_sum - pre_freq

    def drop_freq_from(self, other_ifd, newest=True):
        pre_freq = self._freq_sum
        for other_word, other_freq in other_ifd.word_freq_enumerate(newest):
            self.uncount_word(other_word, other_freq)
        return pre_freq - self._freq_sum

    def dump_dict(self, file_name):
        self.reset_id()
        for word in self.vocabulary():
            if type(word) is not str:
                self.drop_word(word)
        word_id_freq_arr = [(word.strip(), int(self.word2id(word)), int(self.freq_of_word(word)))
                            for word in sorted(self.vocabulary())]
        iu.dump_array(file_name, word_id_freq_arr)

    def load_dict(self, file_name):
        self.clear()
        word_id_freq_arr = iu.load_array(file_name)
        for word, wid, freq in word_id_freq_arr:
            self._word2id[word] = {K_FREQ: int(freq), K_ID: int(wid)}
            self._id2word[wid] = word
            self._freq_sum += freq
        self.calc_freq_sum()
        return self
