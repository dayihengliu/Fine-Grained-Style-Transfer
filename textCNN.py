import tensorflow as tf
import os
import random
import pickle as cPickle
import numpy as np
import time

def linear(input_, output_size, scope=None):
    '''
    Linear map: output[k] = sum_i(Matrix[k, i] * input_[i] ) + Bias[k]
    Args:
    input_: a tensor or a list of 2D, batch x n, Tensors.
    output_size: int, second dimension of W[i].
    scope: VariableScope for the created subgraph; defaults to "Linear".
  Returns:
    A 2D Tensor with shape [batch x output_size] equal to
    sum_i(input_[i] * W[i]), where W[i]s are newly created matrices.
  Raises:
    ValueError: if some of the arguments has unspecified or wrong shape.
  '''

    shape = input_.get_shape().as_list()
    if len(shape) != 2:
        raise ValueError("Linear is expecting 2D arguments: %s" % str(shape))
    if not shape[1]:
        raise ValueError("Linear expects shape[1] of arguments: %s" % str(shape))
    input_size = shape[1]

    # Now the computation.
    with tf.variable_scope(scope or "SimpleLinear"):
        matrix = tf.get_variable("Matrix", [output_size, input_size], dtype=input_.dtype)
        bias_term = tf.get_variable("Bias", [output_size], dtype=input_.dtype)

    return tf.matmul(input_, tf.transpose(matrix)) + bias_term


def highway(input_, size, num_layers=1, bias=-2.0, f=tf.nn.relu, scope='Highway'):
    """Highway Network (cf. http://arxiv.org/abs/1505.00387).
    t = sigmoid(Wy + b)
    z = t * g(Wy + b) + (1 - t) * y
    where g is nonlinearity, t is transform gate, and (1 - t) is carry gate.
    """

    with tf.variable_scope(scope):
        for idx in range(num_layers):
            g = f(linear(input_, size, scope='highway_lin_%d' % idx))

            t = tf.sigmoid(linear(input_, size, scope='highway_gate_%d' % idx) + bias)

            output = t * g + (1. - t) * input_
            input_ = output

    return output


class TextCNN(object):
    """
    A CNN for text classification.
    Uses an embedding layer, followed by a convolutional, max-pooling and softmax layer.
    """

    def __init__(
            self, sess, sequence_length, num_classes, vocab_size, dp,
            emd_dim, filter_sizes, num_filters, l2_reg_lambda=0.0, dropout_keep_prob = 1):
        # Placeholders for input, output and dropout
        self.input_x = tf.placeholder(tf.int32, [None, sequence_length], name="input_x")
        self.input_y = tf.placeholder(tf.float32, [None, num_classes], name="input_y")
        self.dropout_rate = dropout_keep_prob
        self.dropout_input = tf.placeholder(tf.float32, name="dropout_keep_prob")
        self.sess = sess
        self.max_sentence_len = sequence_length
        self.dp = dp
        # Keeping track of l2 regularization loss (optional)
        l2_loss = tf.constant(0.0)

        with tf.variable_scope('TextCNN'):
            # Embedding layer
            with tf.device('/cpu:0'), tf.name_scope("embedding"):
                self.W = tf.Variable(
                    tf.random_uniform([vocab_size, emd_dim], -1.0, 1.0),
                    name="W")
                self.embedded_chars = tf.nn.embedding_lookup(self.W, self.input_x)
                self.embedded_chars_expanded = tf.expand_dims(self.embedded_chars, -1)

            # Create a convolution + maxpool layer for each filter size
            pooled_outputs = []
            for filter_size, num_filter in zip(filter_sizes, num_filters):
                with tf.name_scope("conv-maxpool-%s" % filter_size):
                    # Convolution Layer
                    filter_shape = [filter_size, emd_dim, 1, num_filter]
                    W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W")
                    b = tf.Variable(tf.constant(0.1, shape=[num_filter]), name="b")
                    conv = tf.nn.conv2d(
                        self.embedded_chars_expanded,
                        W,
                        strides=[1, 1, 1, 1],
                        padding="VALID",
                        name="conv")
                    # Apply nonlinearity
                    h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")
                    # Maxpooling over the outputs
                    pooled = tf.nn.max_pool(
                        h,
                        ksize=[1, sequence_length - filter_size + 1, 1, 1],
                        strides=[1, 1, 1, 1],
                        padding='VALID',
                        name="pool")
                    pooled_outputs.append(pooled)

            # Combine all the pooled features
            num_filters_total = sum(num_filters)
            self.h_pool = tf.concat(pooled_outputs, 3)
            self.h_pool_flat = tf.reshape(self.h_pool, [-1, num_filters_total])

            # Add highway
            with tf.name_scope("highway"):
                self.h_highway = highway(self.h_pool_flat, self.h_pool_flat.get_shape()[1], 1, 0)

            # Add dropout
            with tf.name_scope("dropout"):
                self.h_drop = tf.nn.dropout(self.h_highway, self.dropout_input)

            # Final (unnormalized) scores and predictions
            with tf.name_scope("output"):
                W = tf.Variable(tf.truncated_normal([num_filters_total, num_classes], stddev=0.1), name="W")
                b = tf.Variable(tf.constant(0.1, shape=[num_classes]), name="b")
                l2_loss += tf.nn.l2_loss(W)
                l2_loss += tf.nn.l2_loss(b)
                self.scores = tf.nn.xw_plus_b(self.h_drop, W, b, name="scores")
                self.ypred_for_auc = tf.nn.softmax(self.scores)
                self.predictions = tf.argmax(self.scores, 1, name="predictions")

            # CalculateMean cross-entropy loss
            with tf.name_scope("loss"):
                losses = tf.nn.softmax_cross_entropy_with_logits(logits=self.scores, labels=self.input_y)
                self.loss = tf.reduce_mean(losses) + l2_reg_lambda * l2_loss
                self.d_loss = tf.reshape(tf.reduce_mean(self.loss), shape=[1])
                
            with tf.name_scope("accuracy"):
                correct_predictions = tf.equal(self.predictions, tf.argmax(self.input_y, 1))
                self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")
                
        self.params = tf.trainable_variables()
        d_optimizer = tf.train.AdamOptimizer(1e-4)
        grads_and_vars = d_optimizer.compute_gradients(self.loss, self.params, aggregation_method=2)
        self.train_op = d_optimizer.apply_gradients(grads_and_vars)
        #self.saver = tf.train.Saver([v for v in tf.trainable_variables() if 'summary_' not in v.name], max_to_keep = 5)
        self.saver = tf.train.Saver(tf.trainable_variables(), max_to_keep = 5)
        self.summary_placeholders, self.update_ops, self.summary_op = self.setup_summary()
        self.sess.run(tf.global_variables_initializer())
        
    def save(self, path, epoch):
        checkpoint_prefix = os.path.join(path, "model")
        self.saver.save(self.sess, checkpoint_prefix, global_step=epoch)
        print('save to %s success' % checkpoint_prefix)
        
    def restore(self, path):
        self.saver.restore(self.sess, path)
        print('restore %s success' % path)
        
    def setup_summary(self):
        train_loss_ = tf.Variable(0., name='summary_train_loss')
        tf.summary.scalar('Train_loss', train_loss_)
        train_acc_ = tf.Variable(0., name='summary_train_acc')
        tf.summary.scalar('Train_Acc', train_acc_)
        
        test_loss_ = tf.Variable(0., name='summary_train_loss')
        tf.summary.scalar('Train_loss', test_loss_)
        test_acc_ = tf.Variable(0., name='summary_test_acc')
        tf.summary.scalar('Test_Acc', test_acc_)
        
        summary_vars = [train_loss_, train_acc_, test_loss_, test_acc_]
        summary_placeholders = [tf.placeholder(tf.float32) for _ in range(len(summary_vars))]
        update_ops = [summary_vars[i].assign(summary_placeholders[i]) for i in range(len(summary_vars))]
        summary_op = tf.summary.merge_all()
        return summary_placeholders, update_ops, summary_op

    def infer(self, x_str):
        X_ind = [self.dp.w2id[w] for w in x_str.split()]
        X_pad_ind = [X_ind + [self.dp._x_pad] * (self.dp.max_length - len(X_ind))]
        #print(X_pad_ind)
        X_pad_ind = [X[:self.dp.max_length] for X in X_pad_ind]
        predict = self.sess.run(self.predictions, 
                {self.input_x: X_pad_ind,
                self.dropout_input: 1.0})[0]
        return predict
    
    def batch_infer(self, X_indices):
        X_pad_ind = [X_ind + [self.dp._x_pad,] * max(1, self.dp.max_length - len(X_ind)) for X_ind in X_indices]
        #print(X_pad_ind)
        X_pad_ind = [X[:self.dp.max_length] for X in X_pad_ind]
        predict = self.sess.run(self.predictions, 
                                {self.input_x: X_pad_ind,
                                self.dropout_input: 1.0})
        return predict
    
    def batch_infer_auc(self, X_indices):
        X_pad_ind = [X_ind + [self.dp._x_pad,] * max(1, self.dp.max_length - len(X_ind)) for X_ind in X_indices]
        #print(X_pad_ind)
        X_pad_ind = [X[:self.dp.max_length] for X in X_pad_ind]
        predict = self.sess.run(self.ypred_for_auc, 
                                {self.input_x: X_pad_ind,
                                self.dropout_input: 1.0})
        return predict
        
class TextCNN_DP:
    def __init__(self, X_indices, C_labels, w2id, batch_size, max_length, n_epoch, split_ratio=0.1, test_data=None):
        self.n_epoch = n_epoch
        if test_data == None:
            num_test = int(len(X_indices) * split_ratio)
            r = np.random.permutation(len(X_indices))
            X_indices = np.array(X_indices)[r].tolist()
            C_labels = np.array(C_labels)[r].tolist()
            self.C_train = np.array(C_labels[num_test:])
            self.X_train = np.array(X_indices[num_test:])
            self.C_test = np.array(C_labels[:num_test])
            self.X_test = np.array(X_indices[:num_test])
        else:
            self.X_train, self.C_train, self.X_test, self.C_test = test_data
            self.X_train = np.array(self.X_train)
            self.C_train = np.array(self.C_train)
            self.X_test = np.array(self.X_test)
            self.C_test = np.array(self.C_test)
        self.max_length = max_length
        self.num_batch = int(len(self.X_train) / batch_size)
        self.num_steps = self.num_batch * self.n_epoch
        self.batch_size = batch_size
        self.w2id = w2id
        self.id2w = dict(zip(w2id.values(), w2id.keys()))
        self._x_pad = w2id['<PAD>']
        print('Train_data: %d | Test_data: %d | Batch_size: %d | Num_batch: %d | vocab_size: %d' % (len(self.X_train), len(self.X_test), batch_size, self.num_batch, len(self.w2id)))
        
    def next_batch(self, X, C):
        r = np.random.permutation(len(X))
        X = X[r]
        C = C[r]
        for i in range(0, len(X) - len(X) % self.batch_size, self.batch_size):
            X_batch = X[i : i + self.batch_size]
            C_batch = C[i : i + self.batch_size]
            padded_X_batch = self.pad_sentence_batch(X_batch, self._x_pad)
            yield (np.array(padded_X_batch),
                   C_batch)
    
    def sample_test_batch(self):
        i = random.randint(0, int(len(self.C_test) / self.batch_size)-2)
        C = self.C_test[i*self.batch_size:(i+1)*self.batch_size]
        padded_X_batch = self.pad_sentence_batch(self.X_test[i*self.batch_size:(i+1)*self.batch_size], self._x_pad)
        return np.array(padded_X_batch), C
    
        
    def pad_sentence_batch(self, sentence_batch, pad_int):
        padded_seqs = []
        seq_lens = []
        sentence_batch = sentence_batch.tolist()
        max_sentence_len = self.max_length
        for sentence in sentence_batch:
            padded_seqs.append(sentence + [pad_int] * (max_sentence_len - len(sentence)))
            seq_lens.append(len(sentence))
        return padded_seqs