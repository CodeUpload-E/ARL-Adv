from clu.me import *
from utils.deep.layers import *


# noinspection PyAttributeOutsideInit,PyPep8Naming
class N1:
    """  """
    file = __file__

    def __init__(self, args: dict):
        self.m_dim = args[md_]
        self.n_num = args[ns_]
        self.use_bn = args[bn_]
        self.smooth = args[smt_]
        self.margin = args[mgn_]
        self.l_reg = [args[l1_], args[l2_], args[l3_], args[l4_]]
        # self.w_train = args[wtrn_]
        # self.c_train = args[ctrn_]
        self.x_init = tf.contrib.layers.xavier_initializer()
        self.n_init = tf.random_normal_initializer(mean=0., stddev=args[sc_])

    @staticmethod
    def _get_variable(name, init, train):
        shape = init.shape
        initer = tf.constant_initializer(init)
        var = tf.get_variable(name=name, shape=shape, initializer=initer, trainable=train)
        return shape, var

    def define_word_embed(self, init):
        self.w_num, self.w_dim = init.shape
        # trainable = [False, True][self.w_train]
        trainable = True
        self.w_embed = tf.Variable(init, name='w_embed', trainable=trainable, dtype=f32)

    def define_cluster_embed(self, init):
        self.c_num, self.c_dim = init.shape
        # trainable = [False, True][self.c_train]
        trainable = True
        self.c_embed = tf.Variable(init, name='c_embed', trainable=trainable, dtype=f32)

    def define_inputs(self):
        self.is_train = tf.placeholder_with_default(True, (), name='is_train')
        self.dropout = tf.placeholder_with_default(1., (), name='dropout')
        shape = (None, None)
        ph, lk = tf.placeholder, tf.nn.embedding_lookup
        self.p_seq = ph(i32, shape, name='p_seq')
        self.n_seqs = [ph(i32, shape, name='n_seq_{}'.format(i)) for i in range(self.n_num)]
        with tf.name_scope('lookup_pos'):
            self.p_rep = lk(self.w_embed, self.p_seq, name='p_rep')
        with tf.name_scope('lookup_neg'):
            self.n_reps = [lk(self.w_embed, n, name='n_rep_{}'.format(i))
                           for i, n in enumerate(self.n_seqs)]

    def define_denses(self):
        assert self.w_dim == self.c_dim
        ed, md = self.w_dim, self.m_dim
        initer = self.n_init

        self.W_1 = Dense(ed, ed, kernel=initer, name='W_1')
        self.W_2 = Dense(ed, ed, kernel=initer, name='W_2')
        self.W_3 = Dense(ed, ed, kernel=initer, name='W_3')
        self.Q = tf.get_variable(shape=(1, 1, ed), initializer=initer, name='Q')
        self.W_doc = [self.W_1, self.W_2, self.W_3]

        self.D_r = Denses(ed, [(ed, relu), (ed, None)], initer, name='D_r')
        self.Gen = Denses(ed, [(ed, relu), (ed, None)], initer, name='Generator')
        self.Dis = Denses(ed, [(ed, relu), (1, None)], initer, name='Discriminator')

    def multi_dim_att(self, w_emb):
        # return tf.reduce_mean(w_emb, axis=1, keepdims=False)
        # (batch_size, token_num, embed_dim) / (1, 1 , embed_dim)
        # bn_func = wrap_batch_norm(self.is_train) if self.use_bn else None
        bn_func = None
        w = self.W_1(w_emb, name='get_w', post_apply=bn_func)
        q = self.W_2(self.Q, name='get_q', post_apply=bn_func)
        # (batch_size, token_num, embed_dim)
        wq_score = self.W_3(tf.nn.sigmoid(w + q), name='wq_score', post_apply=bn_func)
        wq_alpha = tf.nn.softmax(wq_score, axis=1)
        wq_apply = tf.multiply(wq_alpha, w_emb)
        # (batch_size, embed_dim)
        md_att = tf.reduce_sum(wq_apply, axis=1, keepdims=False)
        return md_att

    def get_c_probs_r(self, e, c, name):
        with tf.name_scope(name):
            # batch_size * clu_num
            c_score = tf.matmul(e, tf.transpose(c))
            c_probs = tf.nn.softmax(c_score, axis=1)
            # batch_size * embed_dim
            r = tf.matmul(c_probs, c)
            # r = self.D_r(r, name='transform_r')
        return c_probs, r

    def get_cpn(self):
        with tf.name_scope('c_embed'):
            # (clu_num, embed_dim)
            c = self.c_embed
        with tf.name_scope('p_mdatt'):
            # (batch_size, embed_dim)
            p = self.multi_dim_att(self.p_rep)
        with tf.name_scope('n_mdatt'):
            # (neg size, embed_dim)
            n = tf.concat([self.multi_dim_att(x) for x in self.n_reps], axis=0)
        return c, p, n

    def forward(self):
        l1, l2, l3, l4 = self.l_reg
        c, Pd, Nd = self.get_cpn()

        with tf.name_scope('similarity'):
            pc_probs, Pr = self.get_c_probs_r(Pd, c, 'reconstruct_p')
            _, Nr = self.get_c_probs_r(Nd, c, 'reconstruct_n')
            with tf.name_scope('loss'):
                Pd_l2, Pr_l2, Nd_l2, Nr_l2 = l2_norm_tensors(Pd, Pr, Nd, Nr)
                PdPr_sim = inner_dot(Pd_l2, Pr_l2, keepdims=True)
                PdNd_sim = tf.matmul(Pd_l2, tf.transpose(Nd_l2))
                PdNd_sim_v = tf.reduce_mean(PdNd_sim, axis=1, keepdims=True)
                PdPr_PdNd_mgn = - PdPr_sim + PdNd_sim_v * l1
                # PdPr_PdNd_mgn = tf.maximum(0.0, 1.0 - PdPr_sim + PdNd_sim_v * l1)
                # DnRn_sim = inner_dot(Nd_l2, Nr_l2, keepdims=True)
                # DnDp_sim = tf.transpose(PdNd_sim_v)
                # DnDp_sim_v = tf.reduce_mean(DnDp_sim, axis=1, keepdims=True)
                # nr_np_mgn = tf.maximum(0.0, 1.0 - DnRn_sim + DnDp_sim_v * l1)
                """merge loss"""
                pairwise = tf.reduce_mean(PdPr_PdNd_mgn)  # + tf.reduce_mean(nr_np_mgn)
                pointwise = tf.reduce_mean(PdPr_sim)  # + tf.reduce_mean(GnRn_sim)
                reg_loss = sum(w.get_norm(order=2) for w in self.W_doc)
                sim_loss = pairwise - pointwise * l3 + reg_loss * l4

        with tf.name_scope('generate'):
            Gen_Pd_full = self.Gen(Pd, name='generate_Pd', last=False)
            Gen_Nd_full = self.Gen(Nd, name='generate_Nd', last=False)
            Gen_Pr_full = self.Gen(Pr, name='generate_Pr', last=False)
            Gen_Nr_full = self.Gen(Nr, name='generate_Nr', last=False)
            G_Pd = Gen_Pd_full[-1]
            G_Nd = Gen_Nd_full[-1]
            G_Pr = Gen_Pr_full[-1]
            G_Nr = Gen_Nr_full[-1]
            G_Pd_l2, G_Nd_l2, G_Pr_l2, G_Nr_l2 = l2_norm_tensors(G_Pd, G_Nd, G_Pr, G_Nr)

        with tf.name_scope('discriminate'):
            org = self.Dis(tf.concat([G_Pd_l2, G_Nd_l2], axis=0), name='dis_org')
            res = self.Dis(tf.concat([G_Pr_l2, G_Nr_l2], axis=0), name='dis_res')
            with tf.name_scope('pred_org_res'):
                D_pred_org = tf.squeeze(org, axis=1)
                D_pred_res = tf.squeeze(res, axis=1)
                D_pred = tf.concat([D_pred_org, D_pred_res], axis=0)
            with tf.name_scope('true_org_res'):
                D_true_org = tf.tile(tf.constant([1. - self.smooth]), tf.shape(D_pred_org))
                D_true_res = tf.tile(tf.constant([self.smooth]), tf.shape(D_pred_res))
                D_true = tf.concat([D_true_org, D_true_res], axis=0)
            with tf.name_scope('loss'):
                cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits
                cross = cross_entropy(labels=D_true, logits=D_pred)
                dis_loss = tf.reduce_mean(cross) * l2
                gen_loss = - dis_loss

        self.pc_probs = pc_probs
        self.sim_loss = sim_loss
        self.gen_loss = gen_loss
        self.dis_loss = dis_loss

        histogram(name='Pd', values=Pd, family='origin')
        histogram(name='Nd', values=Nd, family='origin')
        histogram(name='Pd_l2', values=Pd_l2, family='origin')
        histogram(name='Nd_l2', values=Nd_l2, family='origin')

        histogram(name='Pr', values=Pr, family='recons')
        histogram(name='Nr', values=Nr, family='recons')
        histogram(name='Pr_l2', values=Pr_l2, family='recons')
        histogram(name='Nr_l2', values=Nr_l2, family='recons')

        histogram(name='PdNd_sim', values=PdNd_sim, family='sim')
        histogram(name='PdPr_sim', values=PdPr_sim, family='sim')

        scalar(name='pairwise', tensor=pairwise, family='sim')
        scalar(name='pointwise', tensor=pointwise, family='sim')
        scalar(name='sim_loss', tensor=sim_loss, family='sim')

        histogram(name='G_Pd', values=G_Pd, family='gen')
        histogram(name='G_Nd', values=G_Nd, family='gen')
        histogram(name='G_Pd_l2', values=G_Pd_l2, family='gen')
        histogram(name='G_Nd_l2', values=G_Nd_l2, family='gen')
        histogram(name='PdPr_PdNd_mgn', values=PdPr_PdNd_mgn, family='gen')

        histogram(name='D_pred_org', values=D_pred_org, family='dis')
        histogram(name='D_pred_res', values=D_pred_res, family='dis')
        histogram(name='cross', values=cross, family='dis')
        scalar(name='dis_loss', tensor=dis_loss, family='dis')

    def define_optimizer(self):
        def record_grad_vars(prompt, grads_vars):
            print(prompt, 'var num:{}'.format(len(grads_vars)))
            for g, v in grads_vars:
                v_name = v.name.replace(':', '_')
                print('  {:20} : {}'.format(v_name, 'x' if g is None else 'o'))
                if g is not None:
                    histogram(name=v_name, values=v, family=prompt + '_vars')
                    histogram(name=v_name, values=g, family=prompt + '_grads')

        lr = dict(learning_rate=6e-3, global_step=0, decay_steps=100, decay_rate=0.99)
        vars_d = get_trainable(self.Dis, self.w_embed, self.c_embed)
        # vars_d = get_trainable(self.Dis)
        vars_g = get_trainable(self.D_r, self.Gen)
        vars_s = get_trainable(self.w_embed, self.c_embed, self.D_r, self.Q, *self.W_doc)
        self.dis_op, grad_vars_d = get_train_op(
            lr, self.dis_loss, vars_d, name='dis_op', grads=True)
        self.gen_op, grad_vars_g = get_train_op(
            lr, self.gen_loss, vars_g, name='gen_op', grads=True)
        self.sim_op, grad_vars_s = get_train_op(
            lr, self.sim_loss, vars_s, name='sim_op', grads=True)
        record_grad_vars('vars_d', grad_vars_d)
        record_grad_vars('vars_g', grad_vars_g)
        record_grad_vars('vars_s', grad_vars_s)

    def build(self, w_embed, c_embed):
        self.define_word_embed(w_embed)
        self.define_cluster_embed(c_embed)
        self.define_denses()
        self.define_inputs()
        self.forward()
        self.define_optimizer()

    """ runtime below """

    def set_session(self, sess):
        self.sess = sess

    def train_step(self, fd, epoch, max_epoch, batch_id, max_batch_id):
        # def train():
        #     for _ in range(self.dis_step_num):
        #         sess.run(self.dis_op, feed_dict=fd)
        #     for _ in range(self.gen_step_num):
        #         sess.run(self.gen_op, feed_dict=fd)
        #     for _ in range(self.sim_step_num):
        # control_dependencies(self.use_bn, train)
        self.sess.run(self.sim_op, feed_dict=fd)

    def get_fd_by_batch(self, p_batch, n_batches=None, dropout=1.):
        p_seq = [d.tokenids for d in p_batch]
        pairs = [(self.p_seq, p_seq), (self.dropout, dropout)]
        if n_batches is not None:
            n_seqs = [[d.tokenids for d in n_batch] for n_batch in n_batches]
            pairs += list(zip(self.n_seqs, n_seqs))
        return dict(pairs)

    def get_loss(self, fd):
        sl = self.sess.run([self.sim_loss], feed_dict=fd)
        return {'sim_loss': sl}

    def predict(self, batch):
        fd = self.get_fd_by_batch(batch)
        # fd[self.is_train] = False
        return np.argmax(self.sess.run(self.pc_probs, feed_dict=fd), axis=1).reshape(-1)

    def evaluate(self, batches):
        clusters, topics = list(), list()
        for batch in batches:
            clusters.extend(self.predict(batch))
            topics.extend(d.topic for d in batch)
        return au.scores(topics, clusters, au.eval_scores)
