from .n1 import *


# def get_norm_of_variables(order, variables):
#     assert len(variables) > 0
#     norm = list()
#     for v in variables:
#         if isinstance(v, (Denses, Dense)):
#             norm.append(v.get_norm(order))
#         elif isinstance(v, (tf.Variable,)):
#             norm.append(tf.norm(v, ord=order))
#         else:
#             raise TypeError('type {} of v {} invalid'.format(type(v), v))
#     return sum(norm)


# noinspection PyAttributeOutsideInit,PyPep8Naming
class N5(N1):
    """ 对抗噪声（聚类/单词embedding加噪声参数）"""
    file = __file__

    def __init__(self, args: dict):
        super(N5, self).__init__(args)
        self.eps = args[eps_]
        self.worc = args[worc_]

    def define_cluster_embed(self, init):
        super(N5, self).define_cluster_embed(init)
        self.c_noise = tf.get_variable(name='c_noise', shape=init.shape, initializer=self.x_init)

    def define_word_embed(self, init):
        super(N5, self).define_word_embed(init)
        self.w_noise = tf.get_variable(name='w_noise', shape=init.shape, initializer=self.x_init)

    def define_inputs(self):
        super(N5, self).define_inputs()
        lk = tf.nn.embedding_lookup
        w_embed_nis = self.w_embed + self.w_noise
        with tf.name_scope('lookup_pos_nis'):
            self.p_rep_nis = lk(w_embed_nis, self.p_seq, name='p_rep_nis')
        with tf.name_scope('lookup_neg_nis'):
            self.n_reps_nis = [lk(w_embed_nis, n, name='n_reps_nis_{}'.format(i))
                               for i, n in enumerate(self.n_seqs)]

    def define_denses(self):
        ed, md, init = self.w_dim, self.m_dim, self.x_init
        self.W_1 = Dense(ed, ed, kernel=init, name='W_1')
        self.W_2 = Dense(ed, ed, kernel=init, name='W_2')
        self.W_3 = Dense(ed, ed, kernel=init, name='W_3')
        self.Q = tf.get_variable(shape=(1, 1, ed), initializer=init, name='Q')
        self.W_doc = [self.W_1, self.W_2, self.W_3]

    def multi_dim_att(self, w_emb):
        w = self.W_1(w_emb, name='get_w')
        q = self.W_2(self.Q, name='get_q')
        wq_score = self.W_3(tf.nn.sigmoid(w + q), name='wq_score')
        wq_alpha = tf.nn.softmax(wq_score, axis=1)
        wq_apply = tf.multiply(wq_alpha, w_emb)
        md_att = tf.reduce_sum(wq_apply, axis=1, keepdims=False)
        return wq_alpha, md_att

    # def get_c_probs_r(self, e, c, name):
    #     with tf.name_scope(name):
    #         c_score = tf.matmul(e, tf.transpose(c))
    #         c_probs = tf.nn.softmax(c_score, axis=1)
    #         restore = tf.matmul(c_probs, c)
    #     return c_probs, restore

    def get_cpn_nis(self, w_nis, c_nis):
        with tf.name_scope('c_embed'):
            c = (self.c_embed + self.c_noise) if c_nis else self.c_embed
        with tf.name_scope('p_mdatt'):
            p_rep = self.p_rep_nis if w_nis else self.p_rep
            p_wqatt, p = self.multi_dim_att(p_rep)
        with tf.name_scope('n_mdatt'):
            n_reps = self.n_reps_nis if w_nis else self.n_reps
            n = tf.concat([self.multi_dim_att(n_rep)[-1] for n_rep in n_reps], axis=0)
        if not w_nis and not c_nis:
            print('first p_wqatt')
            self.p_wqatt = p_wqatt
        return c, p, n

    def get_loss_with(self, w_nis: bool, c_nis: bool):
        block_name = 'block' + ('_cnis' if c_nis else '') + ('_wnis' if w_nis else '')
        with tf.name_scope(block_name):
            c, Pd, Nd = self.get_cpn_nis(w_nis, c_nis)
            pc_probs, Pr = self.get_c_probs_r(Pd, c, name='reconstruct_p')
            Pd_l2, Pr_l2, Nd_l2 = l2_norm_tensors(Pd, Pr, Nd)
            PdPr_sim = inner_dot(Pd_l2, Pr_l2, keepdims=True)
            PdNd_sim = tf.matmul(Pd_l2, tf.transpose(Nd_l2))
            pairwise_margin = tf.maximum(0., 1. - PdPr_sim + PdNd_sim)
            pairwise = tf.reduce_mean(pairwise_margin)
            pointwise = tf.reduce_mean(PdPr_sim)
        if not w_nis and not c_nis:
            print('first Pd')
            self.Pd = Pd
        return pc_probs, pairwise, pointwise

    def forward(self):
        l1, l2, l3, l4 = self.l_reg
        pc_probs, pairwise, pointwise = self.get_loss_with(w_nis=False, c_nis=False)
        reg_loss = sum(w.get_norm(order=2, add_bias=False) for w in self.W_doc)
        # self.sim_loss = l1 * pairwise - l3 * pointwise + l4 * reg_loss
        self.sim_loss = tf.constant(0., f32)
        if l1 is not None and l1 > 1e-8:
            self.sim_loss += l1 * pairwise
        if l3 is not None and l3 > 1e-8:
            self.sim_loss -= l3 * pointwise
        if l4 is not None and l4 > 1e-8:
            self.sim_loss += l4 * reg_loss

        if l2 > 1e-8:
            wc_choise = [dict(w_nis=True, c_nis=False), dict(w_nis=False, c_nis=True),
                         dict(w_nis=True, c_nis=True)][self.worc]
            pc_probs_nis, pairwise_nis, _ = self.get_loss_with(**wc_choise)
            self.use_adv_nis = True
            self.adv_loss = self.sim_loss + l2 * pairwise_nis
            self.pairwise_nis = pairwise_nis
            self.pc_probs_nis = pc_probs_nis
        else:
            self.use_adv_nis = False
            self.adv_loss = self.pairwise_nis = self.pc_probs_nis = None

        self.pc_probs = pc_probs
        self.pairwise = pairwise
        self.pointwise = pointwise

    def define_optimizer(self):
        self.optimizer = tf.train.AdamOptimizer(tf.train.exponential_decay(
            learning_rate=5e-3, global_step=0, decay_steps=50, decay_rate=0.99))
        self.sim_gvs = self.optimizer.compute_gradients(self.sim_loss)
        self.sim_opt = self.optimizer.apply_gradients(self.sim_gvs, name='sim_op')
        if not self.use_adv_nis:
            return

        def reassign(noise, grad):
            noise_new = - self.eps * tf.nn.l2_normalize(grad)
            return tf.assign(noise, noise_new)

        adv_gvs = self.optimizer.compute_gradients(self.adv_loss)
        self.adv_v2g = {v: g for g, v in adv_gvs if g is not None}
        self.adv_asg = list()
        if self.w_noise in self.adv_v2g:
            self.adv_asg.append(reassign(self.w_noise, self.adv_v2g[self.w_noise]))
        if self.c_noise in self.adv_v2g:
            self.adv_asg.append(reassign(self.c_noise, self.adv_v2g[self.c_noise]))
        self.adv_gvs = [(g, v) for g, v in adv_gvs if v not in {self.w_noise, self.c_noise}]
        self.adv_opt = self.optimizer.apply_gradients(self.adv_gvs, name='adv_op')

    """ runtime below """

    # def manual_assign_adv(self, train_batches, eval_batches, ppp):
    #     if not self.use_adv_nis:
    #         return ppp('no adv noise used, cannot do this')
    #
    #     ppp('\n\nmanual_assign_adv(), noise type {}'.format(self.worc))
    #     # ppp(self.sess.run(self.c_embed))
    #     use_w_nis, use_c_nis = self.w_noise in self.adv_v2g, self.c_noise in self.adv_v2g
    #     w_grads, c_grads = list(), list()
    #     for bid, pos, neg in train_batches:
    #         fd = self.get_fd_by_batch(pos, neg)
    #         if use_w_nis:
    #             w_grads.append(self.sess.run(self.adv_v2g[self.w_noise], feed_dict=fd))
    #         if use_c_nis:
    #             c_grads.append(self.sess.run(self.adv_v2g[self.c_noise], feed_dict=fd))
    #
    #     def l2norm(x):
    #         return x / np.linalg.norm(x, ord=2)
    #
    #     w_nis_grad = c_nis_grad = None
    #     if use_w_nis:
    #         w_nis_grad = l2norm(np.mean(w_grads, axis=0, keepdims=False))
    #         # w_nis_grad = np.mean([l2norm(g) for g in w_grads], axis=0, keepdims=False)
    #     if use_c_nis:
    #         c_nis_grad = l2norm(np.mean(c_grads, axis=0, keepdims=False))
    #         # c_nis_grad = np.mean([l2norm(g) for g in c_grads], axis=0, keepdims=False)
    #
    #     import pandas as pd
    #     df = pd.DataFrame()
    #     for i, epsilon in enumerate(pow(10, a) for a in range(-1, 8)):
    #         if use_w_nis:
    #             w_noise_new = - epsilon * w_nis_grad
    #             self.sess.run(tf.assign(self.w_noise, w_noise_new))
    #         if use_c_nis:
    #             c_noise_new = - epsilon * c_nis_grad
    #             self.sess.run(tf.assign(self.c_noise, c_noise_new))
    #         adv_scores = list((k + '_adv', v) for k, v in self.evaluate(eval_batches).items()
    #                           if k not in au.eval_scores)
    #         if use_w_nis:
    #             w_noise_new = epsilon * l2norm(np.random.random(w_nis_grad.shape) - 0.5)
    #             self.sess.run(tf.assign(self.w_noise, w_noise_new))
    #         if use_c_nis:
    #             c_noise_new = epsilon * l2norm(np.random.random(c_nis_grad.shape) - 0.5)
    #             self.sess.run(tf.assign(self.c_noise, c_noise_new))
    #         rnd_scores = list((k + '_rnd', v) for k, v in self.evaluate(eval_batches).items()
    #                           if k not in au.eval_scores)
    #
    #         # ppp('{:<10}={}'.format(epsilon, adv_scores))
    #         df.loc[i, 'eps'] = epsilon
    #         for k, v in adv_scores + rnd_scores:
    #             df.loc[i, k] = v
    #     pd.set_option('display.max_columns', 1000)
    #     pd.set_option('display.width', 1000)
    #     ppp(df)

    def evaluate(self, batches):
        def get_scores(pred_target, add_on):
            preds, trues = list(), list()
            for batch in batches:
                c_probs = self.sess.run(pred_target, feed_dict=self.get_fd_by_batch(batch))
                preds.extend(np.argmax(c_probs, axis=1).reshape(-1))
                trues.extend(d.topic for d in batch)
            od = au.scores(trues, preds, au.eval_scores)
            return Od((k + add_on, v) for k, v in od.items())

        from collections import OrderedDict as Od
        scores = Od()
        scores.update(get_scores(self.pc_probs, add_on=''))
        if self.use_adv_nis:
            scores.update(get_scores(self.pc_probs_nis, add_on='_nis'))
        return scores

    def train_step(self, fd, epoch, max_epoch, batch_id, max_batch_id):
        if self.use_adv_nis:
            self.sess.run([self.adv_opt] + self.adv_asg, feed_dict=fd)
        else:
            self.sess.run(self.sim_opt, feed_dict=fd)

    def get_loss(self, fd):
        names = ['adv_loss' if self.use_adv_nis else 'sim_loss', 'pair', 'point']
        loss = [self.adv_loss if self.use_adv_nis else self.sim_loss, self.pairwise, self.pointwise]
        if self.use_adv_nis:
            names.append('pair_nis')
            loss.append(self.pairwise_nis)
        losses = [round(l, 4) for l in self.sess.run(loss, feed_dict=fd)]
        return dict(zip(names, losses))
