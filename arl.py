from .utils import *


class ARL:
    def __init__(self, args: dict):
        # negative sample num
        self.neg_num = args['ns']
        # learning rate
        self.lr = args['lr']
        self.x_init = tf.contrib.layers.xavier_initializer()

    def set_session(self, sess):
        self.sess = sess

    def build(self, w_embed, c_embed):
        self.define_word_embed(w_embed)
        self.define_cluster_embed(c_embed)
        self.define_inputs()
        self.forward()
        self.define_optimizer()

    @staticmethod
    def get_variable(name, init, train):
        initer = tf.constant_initializer(init)
        var = tf.get_variable(name=name, shape=init.shape, initializer=initer, trainable=train)
        return var

    def define_word_embed(self, init):
        pad = tf.zeros(shape=(1, tf.shape(init)[1]), name='w_padding', dtype=tf.float32)
        emb = self.get_variable('w_embed', init, True)
        self.w_embed = tf.concat([pad, emb], axis=0, name='w_embed_concat')

    def define_cluster_embed(self, init):
        self.c_embed = self.get_variable('c_embed', init, True)

    def define_inputs(self):
        self.is_train = tf.placeholder_with_default(True, (), name='is_train')
        lk, ph = tf.nn.embedding_lookup, tf.placeholder
        shape = (None, None)
        self.p_seq = ph(tf.int32, shape, name='p_seq')
        self.n_seqs = [ph(tf.int32, shape, name='n_seq_{}'.format(i)) for i in range(self.neg_num)]
        with tf.name_scope('lookup_pos'):
            self.p_lk = lk(self.w_embed, self.p_seq)
            self.p_rep = tf.reduce_mean(self.p_lk, axis=1, keepdims=False)
        with tf.name_scope('lookup_neg'):
            self.n_lks = [lk(self.w_embed, n, name='n_rep_{}'.format(i)) for i, n in enumerate(self.n_seqs)]
            self.n_reps = tf.concat([tf.reduce_mean(n_lk, axis=1, keepdims=False) for n_lk in self.n_lks], axis=0)

    def define_optimizer(self):
        self.optimizer = tf.train.AdamOptimizer(tf.train.exponential_decay(
            learning_rate=self.lr, global_step=0, decay_steps=50, decay_rate=0.99))
        self.sim_gvs = self.optimizer.compute_gradients(self.sim_loss)
        self.sim_opt = self.optimizer.apply_gradients(self.sim_gvs, name='sim_op')

    def get_loss_with(self):
        with tf.name_scope('get_loss'):
            c_emb = self.c_embed
            pos_d, neg_d = self.p_rep, self.n_reps
            pc_probs, pos_recon = self.get_c_probs_and_recon_d(pos_d, c_emb, name='recon_d')
            pos_d, pos_recon, neg_d = l2_norm_tensors(pos_d, pos_recon, neg_d)
            pos_dr_sim = inner_dot(pos_d, pos_recon, keepdims=True)
            pos_neg_sim = tf.matmul(pos_d, tf.transpose(neg_d))
            pairwise_margin = tf.maximum(0., 1. - pos_dr_sim + pos_neg_sim)
            pairwise = tf.reduce_mean(pairwise_margin)
            pointwise = tf.reduce_mean(pos_dr_sim)
        return pc_probs, pairwise, pointwise

    def get_c_probs_and_recon_d(self, d_emb, c_emb, name):
        with tf.name_scope(name):
            c_score = tf.matmul(d_emb, tf.transpose(c_emb))
            c_probs = tf.nn.softmax(c_score, axis=1)
            recon = tf.matmul(c_probs, c_emb)
        return c_probs, recon

    def forward(self):
        self.pc_probs, pairwise, pointwise = self.get_loss_with()
        J1 = pairwise - pointwise
        self.sim_loss = J1

    def get_fd_by_batch(self, p_batch, n_batches=None):
        p_seq = [d.tokenids for d in p_batch]
        pairs = [(self.p_seq, p_seq)]
        if n_batches is not None:
            n_seqs = [[d.tokenids for d in n_batch] for n_batch in n_batches]
            pairs += list(zip(self.n_seqs, n_seqs))
        return dict(pairs)

    def train_step(self, pos, negs):
        fd = self.get_fd_by_batch(pos, negs)
        self.sess.run(self.sim_opt, feed_dict=fd)

    def predict(self, batch):
        fd = self.get_fd_by_batch(batch)
        fd[self.is_train] = False
        return np.argmax(self.sess.run(self.pc_probs, feed_dict=fd), axis=1).reshape(-1)

    def save(self, file):
        tf.train.Saver().save(self.sess, file)

    def load(self, file):
        tf.train.Saver().restore(self.sess, file)
