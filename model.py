# -*- coding:utf-8 -*-
"""
Created on 2019/4/4 10:18 AM.

Author: Ruizhang1993 (zhang1rui4@foxmail.com)
"""
import tensorflow as tf

class CNN(object):

    def __init__(self,
                 vocab_size,
                 batch_size,
                 embedding_size,
                 num_hidden_size,
                 maxlen,
                 num_categories=2):

        self.vocab_size = vocab_size
        self.batch_size = batch_size
        self.embedding_size = embedding_size
        self.num_hidden_size = num_hidden_size
        self.maxlen = maxlen
        self.num_categories = num_categories

        self.filter_sizes = [1, 2, 3, 4, 5]
        self.filter_nums = [10, 20, 20, 20, 20]

        self._build_model()
        self._build_graph()

    def _build_model(self):

        self.keep_prob = tf.placeholder(tf.float32, name='keep_prob')

        with tf.device('/cpu:0'):
            with tf.variable_scope('embeddings', reuse=tf.AUTO_REUSE):
                self.embeddings = tf.get_variable('embedding_lookup',
                                                  [self.vocab_size, self.embedding_size],
                                                  dtype=tf.float32)

        self.hidden_proj = tf.layers.Dense(self.num_hidden_size, activation='linear')

        def distinguish(input_x):
            input_x_emb = self.hidden_proj(tf.nn.embedding_lookup(self.embeddings, input_x))
            input_x_emb_expand = tf.expand_dims(input_x_emb, -1)

            with tf.variable_scope('discriminator'):
                pooled_outputs = []
                for filter_size, num_filter in zip(self.filter_sizes, self.filter_nums):
                    with tf.variable_scope('conv_maxpool-%s' % filter_size):
                        filter_shape = [filter_size, self.num_hidden_size, 1, num_filter]
                        W = tf.get_variable(name='W',
                                            shape=filter_shape,
                                            initializer=tf.initializers.truncated_normal(stddev=0.1))
                        conv = tf.nn.conv2d(input_x_emb_expand,
                                            W,
                                            [1, 1, 1, 1],
                                            padding='VALID',
                                            name='conv')
                        h = tf.nn.relu(conv)
                        pooled = tf.nn.max_pool(h,
                                                [1, self.maxlen - filter_size + 1, 1, 1],
                                                [1, 1, 1, 1],
                                                padding='VALID',
                                                name='pool')
                        pooled_outputs.append(pooled)
                num_filter_total = sum(self.filter_nums)
                h_pool = tf.concat(pooled_outputs, 3)
                h_pool_flat = tf.reshape(h_pool, [-1, num_filter_total])
                h_pool_flat = tf.contrib.layers.dropout(h_pool_flat, keep_prob=self.keep_prob)

                with tf.name_scope('projection'):
                    dis_dense = tf.layers.dense(h_pool_flat, self.num_hidden_size, name='dis_dense')
                    dis_dense = tf.nn.tanh(dis_dense)
                    dis_logits = tf.layers.dense(dis_dense, 2, name='dis_proj')

            return dis_logits

        self.distinguish = distinguish

    def _build_graph(self):

        self.texts = tf.placeholder(tf.int32, [None, self.maxlen], name='input_texts')
        self.labels = tf.placeholder(tf.int64, [None], name='input_labels')

        self.output = self.distinguish(self.texts)

        self.ypred_for_auc = tf.nn.softmax(self.output)

        self.loss = tf.reduce_mean(
            tf.nn.sparse_softmax_cross_entropy_with_logits(
                logits=self.output+1e-10,
                labels=self.labels
            )
        )

        ypred = tf.cast(tf.argmax(self.output, 1), tf.int64)
        correct = tf.equal(ypred, self.labels)
        self.accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

        tvars = tf.trainable_variables()
        opt = tf.train.AdamOptimizer()
        self.train_op = opt.minimize(self.loss, var_list=tvars)