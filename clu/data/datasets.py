from collections import Counter
from typing import List, Dict

import numpy as np
from utils import au, du, iu, pu, mu
from utils.doc_utils import Document


class Data:
    stem: bool = True
    name: str = None
    topic_num: int = None
    seq_len: int = None
    orgn = None
    wf_flt_func = doc_flt_func = topic_flt_func = None

    def __init__(self):
        # self.orgn_file, = self.fill(self.orgn)
        # self.temp_file, self.docarr_file, self.dict_file = \
        #     self.fill([self.name + s for s in ['_temp.txt', '_docarr.txt', '_dict.txt']])
        # self.embed_init_file, self.word2vec_file = \
        # fill([self.name + s for s in ['_clu_init_300.npy', '_word2vec_300.npy']])
        self.support_dims = [64, 128, 256, 300]
        self.support_ratios = list(np.arange(1, 6) / 5) + list(np.arange(2, 6))
        self.word2wid_file = self.fill_new(self.name + '_word2wid.txt')
        self.new_docarr_file = self.fill_new(self.name + '_docarr.txt')
        self.docarr = self.word2wid = None

    def __call__(self, *args, **kwargs):
        return self

    # def fill(self, files):
    #     base = '/home/cdong/works/clu/data/input_and_outputs/short_text_corpus'
    #     if isinstance(files, list):
    #         return [iu.join(base, self.name, f) for f in files]
    #     elif isinstance(files, str):
    #         return iu.join(base, self.name, files)
    #     else:
    #         raise ValueError('wtf')

    def fill_new(self, files):
        base = '/home/cdong/works/clu/data/corpus'
        if isinstance(files, list):
            return [iu.join(base, self.name, f) for f in files]
        elif isinstance(files, str):
            return iu.join(base, self.name, files)
        else:
            raise ValueError('wtf')

    def get_word2vec_file(self, embed_dim: int, required=True) -> str:
        assert embed_dim in self.support_dims
        file = self.fill_new(self.name + '_word2vec_{}.pkl'.format(embed_dim))
        if required and not iu.exists(file):
            raise ValueError('%s does not exist, the embedding may have not been trained' % file)
        return file

    def get_clu_init_file(self, embed_dim: int, topic_ratio: float, required=True) -> str:
        assert embed_dim in self.support_dims
        file = self.fill_new(self.name + '_clu_init_{}_{:.2f}.pkl'.format(embed_dim, topic_ratio))
        if required and not iu.exists(file):
            raise ValueError('%s does not exist, the embedding may have not been trained' % file)
        return file

    # def load_ifd_and_docarr(self):
    #     return self.load_ifd(), self.load_docarr()
    #
    # def load_docarr(self) -> t.List:
    #     if self.docarr is None:
    #         self.docarr = du.load_docarr(self.docarr_file)
    #     return self.docarr
    #
    # def load_ifd(self):
    #     from utils.id_freq_dict import IdFreqDict
    #     return IdFreqDict().load_dict(self.dict_file)
    #
    # def load_word2vec(self):
    #     wv_list = np.load(self.word2vec_file)
    #     print('len(wv_array) {}, len(wv_array[0]) {}, embed dim {}'.format(
    #         len(wv_list), len(wv_list[0]), len(wv_list[0][1])))
    #     return dict(wv_list)

    def load_new_docarr(self, renew: bool = False) -> List[Document]:
        if self.docarr is None or renew:
            self.docarr = du.load_docarr(self.new_docarr_file)
        return self.docarr

    def load_word2wid(self, renew: bool = False) -> Dict[str, int]:
        if self.word2wid is None or renew:
            self.word2wid = iu.load_json(self.word2wid_file)
        return self.word2wid

    def load_docarr_and_word2wid(self):
        return self.load_new_docarr(), self.load_word2wid()

    def load_word2vec(self, embed_dim: int) -> Dict[str, np.ndarray]:
        word2vec_file = self.get_word2vec_file(embed_dim)
        return iu.load_pickle(word2vec_file)

    def load_clu_init(self, embed_dim: int, topic_ratio: float = 1) -> np.ndarray:
        clu_init_file = self.get_clu_init_file(embed_dim, topic_ratio)
        return iu.load_pickle(clu_init_file)

    def train_word2vec(self, embed_dim: int):
        from utils.node_utils import Nodes
        from gensim.models.word2vec import Word2Vec
        docarr = self.load_new_docarr()
        word2wid = self.load_word2wid()
        word2vec_file = self.get_word2vec_file(embed_dim, required=False)
        tokens_list = [d.tokens for d in docarr]
        print('start training word2vec')
        model = Word2Vec(min_count=0, workers=Nodes.max_cpu_num(), iter=300, size=embed_dim)
        model.build_vocab(tokens_list)
        model.train(tokens_list, total_examples=model.corpus_count, epochs=300)
        word2vec = {w: model[w] for w in word2wid}
        print('word2vec over, vocab size:', len(word2vec))
        iu.dump_pickle(word2vec_file, word2vec)

    def train_centroids(self, embed_dim: int, topic_ratio: float):
        from clu.baselines.kmeans import fit_kmeans
        docarr = self.load_new_docarr()
        word2vec = self.load_word2vec(embed_dim)
        clu_init_file = self.get_clu_init_file(embed_dim, topic_ratio, required=False)
        avg_embeds = list()
        for doc in docarr:
            avg = np.mean([word2vec[word] for word in doc.tokens], axis=0)
            avg_embeds.append(avg)
        centroids = fit_kmeans(avg_embeds, int(self.topic_num * topic_ratio)).cluster_centers_
        print('centroids.shape:{}, to file:{}'.format(centroids.shape, clu_init_file))
        iu.dump_pickle(clu_init_file, centroids)

    def get_topics(self):
        return np.array([d.topic for d in self.load_new_docarr()])

    def get_matrix_topics(self, using):
        assert using in {'tf', 'tfidf'}
        docarr, word2wid = self.load_docarr_and_word2wid()
        matrix = None
        if using == 'tf':
            matrix = [d.tf for d in du.docarr_fit_tf(word2wid, docarr)]
        if using == 'tfidf':
            matrix = [d.tfidf for d in du.docarr_fit_tfidf(word2wid, docarr)]
        return np.array(matrix), self.get_topics()

    def get_matrix_topics_for_dec(self):
        from sklearn.feature_extraction.text import TfidfTransformer
        matrix, topics = self.get_matrix_topics(using='tf')
        topics = np.array(au.reindex(topics))
        matrix = TfidfTransformer(norm='l2', sublinear_tf=True).fit_transform(matrix)
        matrix = matrix.astype(np.float32)
        print(matrix.shape, matrix.dtype, matrix.size)
        matrix = np.asarray(matrix.todense()) * np.sqrt(matrix.shape[1])
        print('todense succeed')
        p = np.random.permutation(matrix.shape[0])
        matrix = matrix[p]
        topics = topics[p]
        print('permutation finished')
        assert matrix.shape[0] == topics.shape[0]
        return matrix, topics

    def get_matrix_topics_for_vade(self):
        docarr, word2wid = self.load_docarr_and_word2wid()
        matrix = [d.tfidf for d in du.docarr_fit_tfidf(word2wid, docarr)]
        return np.array(matrix), self.get_topics()

    def get_avg_embeds_and_topics(self, embed_dim):
        docarr = self.load_new_docarr()
        word2vec = self.load_word2vec(embed_dim)
        e_dim = len(list(word2vec.values())[0])
        vs_list = [[word2vec[w] for w in d.tokens if w in word2vec] for d in docarr]
        avg_embeds = [np.zeros(e_dim) if len(vs) == 0 else np.mean(vs, axis=0) for vs in vs_list]
        print('get avg: 0-len doc: {}/{}'.format(
            sum(1 for v in vs_list if len(v) > 0), len(docarr)))
        return np.array(avg_embeds), np.array([d.topic for d in docarr])

    def filter_from_temp(self):
        c = self.__class__
        topic_flt_func, wf_flt_func, doc_flt_func = c.topic_flt_func, c.wf_flt_func, c.doc_flt_func
        docarr = du.load_docarr(self.temp_file)
        docarr = du.filter_duplicate_docid(docarr)
        docarr = du.tokenize_docarr(docarr, stemming=self.stem)
        print('data prepare (filter duplicate id, tokenize) over')
        acc_topics, rej_topics_1, docarr = du.filter_docarr_by_topic(docarr, topic_flt_func)
        docarr, ifd = du.docarr_bootstrap_ifd(docarr, wf_flt_func)
        # docarr = du.filter_docarr(docarr, doc_flt_func)
        # docarr, ifd = du.docarr_bootstrap_ifd(docarr, wf_flt_func)
        # acc_topics, rej_topics_2, docarr = du.filter_docarr_by_topic(docarr, topic_flt_func)
        # docarr, ifd = du.docarr_bootstrap_ifd(docarr, wf_flt_func)
        # docarr = du.filter_docarr(docarr, doc_flt_func)

        # rej_topics = rej_topics_1 + rej_topics_2
        rej_topics = rej_topics_1
        print('get {} docs\n'.format(len(docarr)))
        print('{} suff topic:{}\n'.format(len(acc_topics), acc_topics.most_common()))
        print('{} insuff topic:{}'.format(len(rej_topics), rej_topics.most_common()[:20]))
        docarr = sorted(docarr, key=lambda d: d.topic)
        ifd.dump_dict(self.dict_file)
        du.dump_docarr(self.docarr_file, docarr)

    def filter_into_temp(self):
        raise RuntimeError('This function should be implemented in sub classes')


class DataTREC(Data):
    name = 'TREC'
    orgn = ['Tweets.txt']
    seq_len = 14
    topic_num = 128
    w_verify_func = du.word_verify(3, 14, 0.8, pu.my_stop_words)
    wf_flt_func = lambda word, freq: freq >= 3
    doc_flt_func = lambda d: len(d.tokens) >= 5 and d.topic is not None
    topic_flt_func = lambda rank, freq: 10 <= freq

    def filter_into_temp(self):
        twarr = iu.load_array(self.orgn_file)
        outrows = list()
        for idx, tw in enumerate(twarr):
            if tw['relevance'] > 1:
                continue
            docid, topic, text = tw['tweetId'], tw['clusterNo'], tw['text']
            if not 10 < len(' '.join(pu.tokenize(text, pu.tokenize_pattern))):
                continue
            outrows.append([docid, topic, text])
        topics = Counter([r[1] for r in outrows])
        print('get {} rows'.format(len(outrows)))
        print('{} topics, {}'.format(len(topics), topics))
        du.dump_docarr(self.temp_file, du.make_docarr(outrows))


class DataGoogle(Data):
    name = 'Google'
    orgn = ['News.txt']
    seq_len = 10
    topic_num = 152
    w_verify_func = du.word_verify(None, None, 0.0, None)
    wf_flt_func = lambda word, freq: freq >= 0
    doc_flt_func = lambda d: len(d.tokens) >= 3 and d.topic is not None
    topic_flt_func = lambda rank, freq: True

    def filter_into_temp(self):
        twarr = iu.load_array(self.orgn_file)
        print(len(twarr), type(twarr[0]))
        docarr = du.make_docarr(
            [[tw[k] for k in ('tweetId', 'clusterNo', 'textCleaned')] for tw in twarr])
        du.dump_docarr(self.temp_file, docarr)


class DataEvent(Data):
    name = 'Event'
    orgn = ['Terrorist']
    seq_len = 14
    topic_num = 69
    w_verify_func = du.word_verify(2, 16, 0.8, pu.nltk_stop_words)
    wf_flt_func = lambda word, freq: freq >= 3
    doc_flt_func = lambda d: len(d.tokens) >= 3 and d.topic is not None
    topic_flt_func = lambda rank, freq: True

    def filter_into_temp(self):
        file_list = iu.list_children(self.orgn_file, full_path=True)
        twarr_list = [iu.load_array(file) for file in file_list]
        doclist = list()
        for topic_id, twarr in enumerate(twarr_list):
            for tw in twarr:
                doclist.append((str(tw['id']), topic_id, tw['text'].replace('#', '')))
        docarr = du.make_docarr(doclist)
        du.dump_docarr(self.temp_file, docarr)


class Data20ng(Data):
    name = '20ng'
    orgn = ['20ng']
    seq_len = 100
    topic_num = 20
    wf_flt_func = lambda word, freq: freq >= 8 and 3 <= len(word) <= 14 and pu.is_valid_word(word)
    doc_flt_func = lambda d: len(d.tokens) >= 4
    topic_flt_func = lambda rank, freq: True

    def filter_into_temp(self):
        json_list = iu.load_array(self.orgn_file)
        item_list = list()
        for i, o in enumerate(json_list):
            text = ' '.join(pu.tokenize(o['text'], pu.tokenize_pattern)[:1200])
            # text = ' '.join(pu.tokenize(o['text'], pu.tokenize_pattern)[:3000])
            # text = o['text']
            item_list.append((i, o['cluster'], text))
        docarr = du.make_docarr(item_list)
        du.dump_docarr(self.temp_file, docarr)


class DataReuters(Data):
    name = 'Reuters'
    orgn = ['segments']
    seq_len = 100
    topic_num = 31
    w_verify_func = du.word_verify(3, 16, 0.8, pu.nltk_stop_words)
    wf_flt_func = lambda word, freq: freq >= 3
    doc_flt_func = lambda d: len(d.tokens) >= 3 and d.topic is not None
    topic_flt_func = lambda rank, freq: freq >= 20

    def filter_into_temp(self):
        from bs4 import BeautifulSoup
        files = iu.list_children(self.orgn_file, full_path=True)
        array = list()
        for fidx, file in enumerate(files):
            print(fidx)
            tree = BeautifulSoup(''.join(iu.read_lines(file)), "html.parser")
            for article in tree.find_all("reuters"):
                topics = list(article.topics.children)
                if not len(topics) == 1:
                    continue
                topic = str(topics[0].text.encode('ascii', 'ignore'))
                text = article.find('text')
                if text is None or text.body is None:
                    continue
                title = str(
                    text.title.text.encode('utf-8', 'ignore')) if text.title is not None else ''
                title = ' '.join(pu.tokenize(title, pu.tokenize_pattern))
                body = str(text.body.text.encode('utf-8', 'ignore'))
                body = ' '.join(pu.tokenize(body, pu.tokenize_pattern))
                array.append((topic, '{}, {}'.format(title, body)))
        docarr = du.make_docarr([(idx, topic, body) for idx, (topic, body) in enumerate(array)])
        print(len(docarr))
        print(Counter([d.topic for d in docarr]))
        print(len(sorted(set([d.topic for d in docarr]))))
        du.dump_docarr(self.temp_file, docarr)


class_list = [DataTREC, DataGoogle, DataEvent, DataReuters, Data20ng]
object_list = [_c() for _c in class_list]
name_list = [_c.name for _c in class_list]
name2d_class = dict(zip(name_list, class_list))
name2object = dict(zip(name_list, object_list))
dft = object()


class Sampler:
    def __init__(self, d_cls):
        if d_cls in name2d_class:
            d_cls = name2d_class[d_cls]
        self.d_obj: Data = d_cls()
        self.seq_len: int = self.d_obj.seq_len
        self.docarr: List[Document] = None
        self.word2wid: dict = None
        self.word2vec: dict = None
        self.eval_batches: List[List[Document]] = None
        self.clu_embed_init = self.word_embed_init = None

    def pad_docarr(self, seq_len):
        assert seq_len is not None and seq_len > 0
        for doc in self.docarr:
            tokenids = doc.tokenids
            n_tokens = len(tokenids)
            if n_tokens == seq_len:
                doc.tokenids = tokenids
            elif n_tokens < seq_len:
                doc.tokenids = tokenids + [0] * (seq_len - n_tokens)
            else:
                doc.tokenids = tokenids[:seq_len]

    def load(self, embed_dim: int = 64, topic_ratio: float = 1, use_pad: bool = True):
        self.docarr = self.d_obj.load_new_docarr()
        if use_pad:
            self.pad_docarr(self.d_obj.seq_len)
        self.word2wid = self.d_obj.load_word2wid()
        self.word2vec = self.d_obj.load_word2vec(embed_dim)
        self.clu_embed_init = self.d_obj.load_clu_init(embed_dim, topic_ratio)
        self.word_embed_init = [None] * len(self.word2vec)
        for word, wid in self.word2wid.items():
            self.word_embed_init[wid - 1] = self.word2vec[word]
        self.word_embed_init = np.array(self.word_embed_init)
        for e in self.word_embed_init:
            assert e is not None
        # max_wid, min_wid = -1, 1e9
        # for d in self.docarr:
        #     max_wid = max(max_wid, max(d.tokenids))
        #     min_wid = min(min_wid, min(d.tokenids))
        # print('wid in docs:', max_wid, min_wid)
        # wids = self.word2wid.values()
        # print('wid in word2wid:', max(wids), min(wids))
        # self.ifd, self.docarr = self.d_obj.load_ifd_and_docarr()
        # assert self.v_size == max(max(d.tokenids) for d in self.docarr) + 1
        self.eval_batches = [pos for pos, negs in self.generate(128, 0, False)]
        # self.eval_batches = self.split_length(self.docarr, 256)

    @staticmethod
    def split_length(docarr: List[Document], batch_size: int) -> List[List[Document]]:
        docarr = sorted(docarr, key=lambda x: len(x.tokenids))
        batches, batch = list(), list()
        prev_len = len(docarr[0].tokenids)
        for i, doc in enumerate(docarr):
            doc_len = len(doc.tokenids)
            if doc_len != prev_len or len(batch) >= batch_size:
                batches.append(batch)
                batch = [doc]
            else:
                batch.append(doc)
            if i >= len(docarr) - 1:
                batches.append(batch)
                break
            prev_len = doc_len
        return batches

    def generate(self, batch_size: int, neg_batch_num: int, shuffle: bool):
        docarr = au.shuffle(self.docarr) if shuffle else self.docarr
        docarr_list = au.split_slices(docarr, batch_size)
        for docarr in docarr_list:
            yield docarr, None
        # batches = self.split_length(docarr, batch_size)
        # print('shuffle_generate - batch num:', len(batches))
        # for i in range(len(batches)):
        #     p_batch: List[Document] = batches[i]
        #     yield p_batch, None
        # n_idxes: List[int] = np.random.choice([j for j in i_range if j != i], neg_batch_num)
        # n_batches: List[List[Document]] = [batches[j] for j in n_idxes]
        # yield p_batch, n_batches


def summary_datasets():
    import pandas as pd
    df = pd.DataFrame(columns=['K', 'D', 'V', 'Avg-len', 'Max-len', 'Min-len'])
    for idx, cls in enumerate([DataTREC, DataGoogle, DataEvent, Data20ng, DataReuters]):
        d_obj = cls()
        print(d_obj.new_docarr_file)
        docarr, word2wid = d_obj.load_docarr_and_word2wid()
        len_arr = [len(d.tokenids) for d in docarr]
        K = len(set([d.topic for d in docarr]))
        D = len(docarr)
        V = len(word2wid)
        avg_len = round(float(np.mean(len_arr)), 2)
        max_len = round(float(np.max(len_arr)), 2)
        min_len = round(float(np.min(len_arr)), 2)
        df.loc[d_obj.name] = [K, D, V, avg_len, max_len, min_len]
    print(df)
    df.to_csv('summary.csv')


def transfer_files(d_class: Data):
    data = d_class()
    print(data.name)
    print(data.new_docarr_file)
    print(data.word2wid_file)
    print('train word2vec & centroids')
    print()
    for dim in [256]:
        for ratio in [1]:
            print(data.name, dim, ratio)
            data.train_word2vec(dim)
            data.train_centroids(dim, ratio)
    # data_path = iu.parent_name(data.new_docarr_file)
    # iu.mkprnts(data.new_docarr_file)
    #
    # doc_arr = du.load_docarr(data.docarr_file)
    # for doc in doc_arr:
    #     doc.tokenids = [i + 1 for i in doc.tokenids]
    # du.dump_docarr(data.new_docarr_file, doc_arr)
    #
    # ifd = data.load_ifd()
    # orgn_word2wid = ifd.get_word2id()
    # assert min(wid for wid in orgn_word2wid.values()) == 0
    # new_word2wid = {word: wid + 1 for word, wid in orgn_word2wid.items()}
    # iu.dump_json(data.word2wid_file, new_word2wid)


if __name__ == '__main__':
    # for d in class_list:
    #     print(d.name)
    #     Sampler(d).load()
    # exit()
    mu.multi_process(transfer_files, args_list=[(c,) for c in class_list])
    # summary_datasets()
    # exit()
    # to_btm()
    # exit()
    # for _d in [Data20ng]:
    #     print('Going to process: {}, continue?'.format(_d.name))
    #     input()
    #     _d = _d()
    #     _d.filter_into_temp()
    #     _d.filter_from_temp()
    #     _d.train_word2vec(embed_dim=300)
    #     _d.train_centroids()
