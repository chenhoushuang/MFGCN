import numpy as np
import os
import tensorflow as tf
from tensorflow.python.client import device_lib
from math import ceil
from time import time

from utility.DataHelper import *
from utility.Metrics import *

COMBINER = 'mean'
EMB_DIM = 16
LAYERS = 2
ALPHA = 0.8
EPOCHS = 100
TEST_INTERVAL = 10
# PRINT_INTERVAL = 1
LR = 0.001
BATCH_SIZE = 128
TOPKS = [5, 10, 15]

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
gpus = [x.name for x in device_lib.list_local_devices() if x.device_type == 'GPU']
cpus = [x.name for x in device_lib.list_local_devices() if x.device_type == 'CPU']
data_generator = Data(BATCH_SIZE)


class MFGCN:
    def __init__(self, data, pretrain_data=None):
        self.pretrain_data = pretrain_data
        self.user_adj = data.norm_adj
        self.item_adj = data.norm_adj.T
        self.n_users = data.ftr_size['User']
        self.n_items = data.ftr_size['Movie']
        self.ftr_size = data.ftr_size
        self.userInfo = data.UserInfo
        self.itemInfo = data.MovieInfo
        self.rating = data.data
        self.weights = self._init_weights()
        self.sample_size = len(self.rating)

        # input argument
        self.pos_items = tf.placeholder(tf.int32, shape=(None,))
        self.neg_items = tf.placeholder(tf.int32, shape=(None,))
        self.users = tf.placeholder(tf.int32, shape=(None,))

        self.user_features, self.item_features = self.forward()
        self.emb_loss, self.mf_loss = self.loss_net()
        self.batch_ratings = self.pred()
        self.loss = self.emb_loss + self.mf_loss
        self.opt = tf.train.AdamOptimizer(learning_rate=LR).minimize(self.loss)

    def _init_weights(self):
        all_weights = dict()
        initializer = tf.contrib.layers.xavier_initializer()
        if self.pretrain_data is None:
            all_weights['user_id_embedding'] = tf.Variable(initializer([self.n_users, EMB_DIM]),
                                                           name='user_id_embedding')
            all_weights['user_gender_embedding'] = tf.Variable(initializer([self.ftr_size['Gender'], EMB_DIM]),
                                                               name='user_gender_embedding')
            all_weights['user_age_embedding'] = tf.Variable(initializer([self.ftr_size['Age'], EMB_DIM]),
                                                            name='user_age_embedding')
            all_weights['user_job_embedding'] = tf.Variable(initializer([self.ftr_size['Job'], EMB_DIM]),
                                                            name='user_job_embedding')
            all_weights['item_id_embedding'] = tf.Variable(initializer([self.ftr_size['Movie'], EMB_DIM]),
                                                           name='item_id_embedding')
            all_weights['item_category_embedding'] = tf.Variable(initializer([self.ftr_size['Genres'], EMB_DIM]),
                                                                 name='item_category_embedding')
        else:
            # To do
            pass
        return all_weights

    def get_user_embedding(self):
        id = self.weights['user_id_embedding']
        gender = tf.nn.embedding_lookup(self.weights['user_gender_embedding'], self.userInfo['Gender'])
        age = tf.nn.embedding_lookup(self.weights['user_age_embedding'], self.userInfo['Age'])
        job = tf.nn.embedding_lookup(self.weights['user_job_embedding'], self.userInfo['JobID'])
        user_embedding = tf.stack([id, gender, age, job], axis=1)
        return user_embedding

    def get_item_embedding(self):
        id = self.weights['item_id_embedding']
        category = []
        for idx in self.itemInfo['Genres']:
            cat = tf.nn.embedding_lookup(self.weights['item_category_embedding'], idx)
            if COMBINER == 'mean':
                cat = tf.reduce_mean(cat, axis=0)
            elif COMBINER == 'sum':
                cat = tf.reduce_sum(cat, axis=0)
            category.append(cat)
        category = tf.stack(category, axis=0)
        item_embedding = tf.stack([id, category], axis=1)
        return item_embedding

    def forward(self):
        user_embedding = self.get_user_embedding()
        item_embedding = self.get_item_embedding()
        user_features = [user_embedding]
        item_features = [item_embedding]

        ratio = ALPHA

        for layer_idx in range(LAYERS):
            # self attention
            user_neighbor_avg = tf.tensordot(self.user_adj, item_embedding, axes=[[-1], [0]])
            scores1 = tf.matmul(user_embedding, user_neighbor_avg, transpose_b=True)
            distribution1 = tf.nn.softmax(scores1)
            new_user_embedding = tf.matmul(distribution1, user_neighbor_avg)

            item_neigbor_avg = tf.tensordot(self.item_adj, user_embedding, axes=[[-1], [0]])
            scores2 = tf.matmul(item_embedding, item_neigbor_avg, transpose_b=True)
            distribution2 = tf.nn.softmax(scores2)
            new_item_embedding = tf.matmul(distribution2, item_neigbor_avg)

            user_features.append(ratio * new_user_embedding)
            item_features.append(ratio * new_item_embedding)
            ratio *= ALPHA
            user_embedding = new_user_embedding
            item_embedding = new_item_embedding
        return tf.reduce_sum(user_features, axis=0), tf.reduce_sum(item_features, axis=0)

    def loss_net(self):
        regular_loss = [tf.nn.l2_loss(weight) for weight in self.weights.values()]
        emb_loss = tf.reduce_mean(regular_loss)
        # user_features = tf.nn.embedding_lookup(user_embedding,self.rating['UserID'])
        user_features = tf.nn.embedding_lookup(self.user_features, self.users)
        item_pos_features = tf.nn.embedding_lookup(self.item_features, self.pos_items)
        item_neg_features = tf.nn.embedding_lookup(self.item_features, self.neg_items)
        # label = self.rating['Rating']
        pos_scores = tf.matmul(user_features, item_pos_features, transpose_b=True)
        pos_scores = tf.reduce_sum(pos_scores, axis=[1, 2])
        neg_scores = tf.matmul(user_features, item_neg_features, transpose_b=True)
        neg_scores = tf.reduce_sum(neg_scores, axis=[1, 2])
        mf_loss = tf.reduce_mean(tf.nn.softplus(-(pos_scores - neg_scores)))
        return emb_loss, mf_loss

    def pred(self):
        user_features = tf.nn.embedding_lookup(self.user_features, self.users)
        user_features = tf.reduce_sum(user_features, axis=1)
        item_features = tf.reduce_sum(self.item_features, axis=1)
        scores = tf.matmul(user_features, item_features, transpose_b=True)
        return scores


def rating_test(sess, model: MFGCN, users_to_test, train_set_flag=0):
    # B: batch size
    # N: the number of items
    top_show = np.sort(TOPKS)
    max_top = max(top_show)
    result = {'precision': np.zeros(len(TOPKS)), 'recall': np.zeros(len(TOPKS)), 'ndcg': np.zeros(len(TOPKS))}

    u_batch_size = BATCH_SIZE

    test_users = users_to_test
    n_test_users = len(test_users)
    # n_user_batchs = n_test_users // u_batch_size + 1
    n_user_batchs = ceil(n_test_users / u_batch_size)
    count = 0
    all_result = []
    for u_batch_id in range(n_user_batchs):
        start = u_batch_id * u_batch_size
        end = (u_batch_id + 1) * u_batch_size
        if end > n_test_users:
            end = n_test_users
        user_batch = test_users[start: end]
        if len(gpus):
            with tf.device(gpus[-1]):
                rate_batch = sess.run(model.batch_ratings, {model.users: user_batch})
        else:
            rate_batch = sess.run(model.batch_ratings, {model.users: user_batch})
        rate_batch = np.array(rate_batch)  # (B, N)
        test_items = []
        if train_set_flag == 0:
            for user in user_batch:
                test_items.append(data_generator.test_set[user])  # (B, #test_items)

            # set the ranking scores of training items to -inf,
            # then the training items will be sorted at the end of the ranking list.
            for idx, user in enumerate(user_batch):
                train_items_off = data_generator.train_set[user]
                rate_batch[idx][train_items_off] = -np.inf
        else:
            for user in user_batch:
                test_items.append(data_generator.train_set[user])

        batch_result = eval_score_matrix_foldout(rate_batch, test_items, max_top)  # (B,k*metric_num), max_top= 20
        count += len(batch_result)
        all_result.append(batch_result)

    assert count == n_test_users
    all_result = np.concatenate(all_result, axis=0)
    final_result = np.mean(all_result, axis=0)  # mean
    final_result = np.reshape(final_result, newshape=[5, max_top])
    final_result = final_result[:, top_show - 1]
    final_result = np.reshape(final_result, newshape=[5, len(top_show)])
    result['precision'] += final_result[0]
    result['recall'] += final_result[1]
    result['ndcg'] += final_result[3]
    return result


def train():
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)

    model = MFGCN(data_generator)
    sess.run(tf.global_variables_initializer())

    n_batch = ceil(data_generator.n_train / BATCH_SIZE)
    print('training...')
    for epoch in range(1, EPOCHS + 1):
        t1 = time()
        loss, mf_loss, emb_loss = 0., 0., 0.
        for idx in range(n_batch):
            users, pos_items, neg_items = data_generator.sample_train()
            if len(gpus):
                with tf.device(gpus[-1]):
                    _, batch_loss, batch_mf_loss, batch_emb_loss = sess.run(
                        [model.opt, model.loss, model.mf_loss, model.emb_loss],
                        feed_dict={model.users: users, model.pos_items: pos_items, model.neg_items: neg_items})
            else:
                _, batch_loss, batch_mf_loss, batch_emb_loss = sess.run(
                    [model.opt, model.loss, model.mf_loss, model.emb_loss],
                    feed_dict={model.users: users, model.pos_items: pos_items, model.neg_items: neg_items})
            loss += batch_loss / n_batch
            mf_loss += batch_mf_loss / n_batch
            emb_loss += batch_emb_loss / n_batch
        if np.isnan(loss):
            print('ERROR: loss is nan.')
            sys.exit()

        if (epoch % TEST_INTERVAL) != 0:
            perf_str = 'Epoch %d [%.1fs]: train==[%.5f=%.5f + %.5f]' % (
                    epoch, time() - t1, loss, mf_loss, emb_loss)
            print(perf_str)
            continue

        users_to_test = list(data_generator.train_set.keys())
        ret = rating_test(sess, model, users_to_test, train_set_flag=1)
        perf_str = 'Epoch %d: train==[%.5f=%.5f + %.5f], recall=[%s], precision=[%s], ndcg=[%s]' % \
                   (epoch, loss, mf_loss, emb_loss,
                    ', '.join(['%.5f' % r for r in ret['recall']]),
                    ', '.join(['%.5f' % r for r in ret['precision']]),
                    ', '.join(['%.5f' % r for r in ret['ndcg']]))
        print(perf_str)

        # test on test data
        loss_test,mf_loss_test,emb_loss_test=0.,0.,0.
        t1 = time()
        loss, mf_loss, emb_loss = 0., 0., 0.
        for idx in range(n_batch):
            users, pos_items, neg_items = data_generator.sample_test()
            if len(gpus):
                with tf.device(gpus[-1]):
                    _, batch_loss, batch_mf_loss, batch_emb_loss = sess.run(
                        [model.opt, model.loss, model.mf_loss, model.emb_loss],
                        feed_dict={model.users: users, model.pos_items: pos_items, model.neg_items: neg_items})
            else:
                _, batch_loss, batch_mf_loss, batch_emb_loss = sess.run(
                    [model.opt, model.loss, model.mf_loss, model.emb_loss],
                    feed_dict={model.users: users, model.pos_items: pos_items, model.neg_items: neg_items})
            loss_test += batch_loss / n_batch
            mf_loss_test += batch_mf_loss / n_batch
            emb_loss_test += batch_emb_loss / n_batch

        users_to_test = list(data_generator.test_set.keys())
        ret = rating_test(sess, model, users_to_test)
        perf_str = 'Epoch %d: test==[%.5f=%.5f + %.5f], recall=[%s], precision=[%s], ndcg=[%s]' % \
                   (epoch, loss_test, mf_loss_test, emb_loss_test,
                    ', '.join(['%.5f' % r for r in ret['recall']]),
                    ', '.join(['%.5f' % r for r in ret['precision']]),
                    ', '.join(['%.5f' % r for r in ret['ndcg']]))
        print(perf_str)

        # cur_best_pre_0, stopping_step, should_stop = early_stopping(ret['recall'][0], cur_best_pre_0,
        #                                                             stopping_step, expected_order='acc', flag_step=5)
        # if should_stop == True:
        #     break

    saver = tf.train.Saver(max_to_keep=1)
    saver.save(sess,'./weights',global_step=EPOCHS)
    print('save the weights')


if __name__=='__main__':
    train()

