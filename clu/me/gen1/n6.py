from .n5 import *


# noinspection PyAttributeOutsideInit,PyPep8Naming
class N6(N5):
    """ mean pooling """
    file = __file__

    # def __init__(self, args):
    #     super(N6, self).__init__(args)
    #     self.topk = args[tpk_]

    def multi_dim_att(self, w_emb):
        return None, tf.reduce_mean(w_emb, axis=1, keepdims=False)

    def forward(self):
        def is_valid(v):
            return float(v) > 1e-6

        l1, l2, l3, _ = self.l_reg
        zero = tf.constant(0., f32)
        pc_probs, pairwise, pointwise = self.get_loss_with(w_nis=False, c_nis=False)
        J1 = zero
        if is_valid(l1):
            J1 += l1 * pairwise
        if is_valid(l2):
            J1 -= l2 * pointwise

        if is_valid(l3):
            wc_choise = [dict(w_nis=True, c_nis=False), dict(w_nis=False, c_nis=True),
                         dict(w_nis=True, c_nis=True)][self.worc]
            # wc_choise = [None, dict(w_nis=False, c_nis=True), None][self.worc]
            pc_probs_nis, pairwise_nis, pointwise_nis = self.get_loss_with(**wc_choise)
            J2 = zero
            if is_valid(l1):
                J2 += l1 * pairwise_nis
            if is_valid(l2):
                J2 -= l2 * pointwise_nis
            self.use_adv_nis = True
            self.adv_loss = J1 + l3 * J2
            self.pairwise_nis = pairwise_nis
            self.pc_probs_nis = pc_probs_nis
        else:
            self.use_adv_nis = False
            self.adv_loss = self.pairwise_nis = self.pc_probs_nis = None

        self.sim_loss = J1
        self.pc_probs = pc_probs
        self.pairwise = pairwise
        self.pointwise = pointwise

    # def forward(self):
    #     l1, l2, l3, l4 = self.l_reg
    #     pc_probs, pairwise, pointwise = self.get_loss_with(w_nis=False, c_nis=False)
    #     # self.sim_loss = l1 * pairwise - l3 * pointwise
    #     self.sim_loss = tf.constant(0., f32)
    #     if l1 is not None and l1 > 1e-8:
    #         self.sim_loss += l1 * pairwise
    #     if l3 is not None and l3 > 1e-8:
    #         self.sim_loss -= l3 * pointwise
    #
    #     if l2 > 1e-8:
    #         wc_choise = [dict(w_nis=True, c_nis=False), dict(w_nis=False, c_nis=True),
    #                      dict(w_nis=True, c_nis=True)][self.worc]
    #         pc_probs_nis, pairwise_nis, _ = self.get_loss_with(**wc_choise)
    #         self.use_adv_nis = True
    #         self.adv_loss = self.sim_loss + l2 * pairwise_nis
    #         self.pairwise_nis = pairwise_nis
    #         self.pc_probs_nis = pc_probs_nis
    #     else:
    #         self.use_adv_nis = False
    #         self.adv_loss = self.pairwise_nis = self.pc_probs_nis = None
    #
    #     self.pc_probs = pc_probs
    #     self.pairwise = pairwise
    #     self.pointwise = pointwise
