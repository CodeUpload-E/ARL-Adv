# ARL-Adv

Tensorflow implementations of short text clustering model [ARL-Adv](https://arxiv.org/abs/1912.03720) 

Word embeddings are initialized by training word2vec model on the corpus and we initialize centroids of the clusters by performing K-means on text embeddings, where  text embeddings are obtained by averaging the embeddings of the words they contain.

ARL-Adv will be published in IEEE TKDE. If you want to cite our work, please use the following information:

>> Wei Zhang, Chao Dong, Jianhua Ying, Jianyong Wang  
>> Attentive Representation Learning with Adversarial Training for Short Text Clustering  
>> IEEE Transactions on Knowledge and Data Engineering (TKDE) , 2021

We also make the dataset Event available.
Other datasets are publicly available, two of which could be accessed through the URL (https://github.com/jackyin12/GSDMM)
