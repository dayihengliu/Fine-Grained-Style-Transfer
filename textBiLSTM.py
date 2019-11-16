import os
import tensorflow as tf
import numpy as np
import time
import random
import pickle



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


class BiLSTM:
    def __init__(self, dp, rnn_size, n_layers, num_classes, encoder_embedding_dim, 
                 sess, lr=0.001, grad_clip=5.0, force_teaching_ratio=1.0, l2_reg_lambda=0,
                residual=False, output_keep_prob=0.5, input_keep_prob=0.9, cell_type='lstm', reverse=False,
                decay_scheme='luong234'):
        
        self.rnn_size = rnn_size
        self.n_layers = n_layers
        self.grad_clip = grad_clip
        self.dp = dp
        self.num_classes = num_classes
        self.encoder_embedding_dim = encoder_embedding_dim
        self.residual = residual
        self.decay_scheme = decay_scheme
        self.l2_reg_lambda = l2_reg_lambda
        if self.residual:
            assert encoder_embedding_dim == rnn_size
        self.reverse = reverse
        self.cell_type = cell_type
        self.force_teaching_ratio = force_teaching_ratio
        self._output_keep_prob = output_keep_prob
        self._input_keep_prob = input_keep_prob
        self.sess = sess
        self.lr=lr
        self.build_graph()
        self.sess.run(tf.global_variables_initializer())
        self.saver = tf.train.Saver(tf.global_variables(), max_to_keep = 35)
        #self.summary_placeholders, self.update_ops, self.summary_op = self.setup_summary()
        
    # end constructor

    def build_graph(self):
        self.register_symbols()
        self.add_input_layer()
        self.add_encoder_layer()
        self.add_classifer()
        self.add_backward_path()
    # end method
    
    def _item_or_tuple(self, seq):
        """Returns `seq` as tuple or the singular element.
        Which is returned is determined by how the AttentionMechanism(s) were passed
        to the constructor.
        Args:
          seq: A non-empty sequence of items or generator.
        Returns:
           Either the values in the sequence as a tuple if AttentionMechanism(s)
           were passed to the constructor as a sequence or the singular element.
        """
        t = tuple(seq)
        if self._is_multi:
            return t
        else:
            return t[0]
        
    def add_input_layer(self):
        self.input_x = tf.placeholder(tf.int32, [None, None], name="X")
        self.X_seq_len = tf.placeholder(tf.int32, [None], name="X_seq_len")
        self.input_y = tf.placeholder(tf.int32, [None, self.num_classes], name='C')
        self.input_keep_prob = tf.placeholder(tf.float32,name="input_keep_prob")
        self.output_keep_prob = tf.placeholder(tf.float32,name="output_keep_prob")
        self.batch_size = tf.shape(self.input_x)[0]
        self.l2_loss = tf.constant(0.0)
        self.global_step = tf.Variable(0, name="global_step", trainable=False)
    # end method

    def single_cell(self, reuse=False):
        if self.cell_type == 'lstm':
             cell = tf.contrib.rnn.LayerNormBasicLSTMCell(self.rnn_size, reuse=reuse)
        else:
            cell = tf.contrib.rnn.GRUBlockCell(self.rnn_size)    
        cell = tf.contrib.rnn.DropoutWrapper(cell, self.output_keep_prob, self.input_keep_prob)
        if self.residual:
            cell = myResidualCell.ResidualWrapper(cell)
        return cell
    
    def add_encoder_layer(self):
        encoder_embedding = tf.get_variable('encoder_embedding', [len(self.dp.X_w2id), self.encoder_embedding_dim],
                                             tf.float32, tf.random_uniform_initializer(-1.0, 1.0))
        
        self.encoder_inputs = tf.nn.embedding_lookup(encoder_embedding, self.input_x)
        bi_encoder_output, bi_encoder_state = tf.nn.bidirectional_dynamic_rnn(
            cell_fw = tf.contrib.rnn.MultiRNNCell([self.single_cell() for _ in range(self.n_layers)]), 
            cell_bw = tf.contrib.rnn.MultiRNNCell([self.single_cell() for _ in range(self.n_layers)]),
            inputs = self.encoder_inputs,
            sequence_length = self.X_seq_len,
            dtype = tf.float32,
            scope = 'bidirectional_rnn')
        self.encoder_out = tf.reduce_mean(tf.concat(bi_encoder_output, 2), 1)
        print('encoder_out', self.encoder_out)
    
    def add_classifer(self):
        #print('self.encoder_out', self.encoder_out)
        with tf.name_scope("highway"):
            self.h_highway = highway(self.encoder_out, self.encoder_out.get_shape()[1], 1, 0)

        # Add dropout
        with tf.name_scope("dropout"):
            self.h_drop = tf.nn.dropout(self.h_highway, self.output_keep_prob)
        
        # Final (unnormalized) scores and predictions
        with tf.name_scope("output"):
            W = tf.Variable(tf.truncated_normal([self.rnn_size * 2 * self.n_layers, self.num_classes], stddev=0.1), name="W")
            b = tf.Variable(tf.constant(0.1, shape=[self.num_classes]), name="b")
            self.l2_loss += tf.nn.l2_loss(W)
            self.l2_loss += tf.nn.l2_loss(b)
            self.scores = tf.nn.xw_plus_b(self.h_drop, W, b, name="scores")
            self.ypred_for_auc = tf.nn.softmax(self.scores)
            self.predictions = tf.argmax(self.scores, 1, name="predictions")

       
    def add_backward_path(self):
        #print(self.logits, self.C)
        # CalculateMean cross-entropy loss
        with tf.name_scope("loss"):
            losses = tf.nn.softmax_cross_entropy_with_logits(logits=self.scores, labels=self.input_y)
            self.loss = tf.reduce_mean(losses) + self.l2_reg_lambda * self.l2_loss
            self.d_loss = tf.reshape(tf.reduce_mean(self.loss), shape=[1])

        with tf.name_scope("accuracy"):
            correct_predictions = tf.equal(self.predictions, tf.argmax(self.input_y, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")
        params = tf.trainable_variables()
        gradients = tf.gradients(self.d_loss, params)
        clipped_gradients, _ = tf.clip_by_global_norm(gradients, self.grad_clip)
        self.learning_rate = tf.constant(self.lr)
        self.learning_rate = self.get_learning_rate_decay(self.decay_scheme)  # decay
        self.train_op = tf.train.AdamOptimizer(self.learning_rate).apply_gradients(zip(clipped_gradients, params), global_step=self.global_step)
        
    def register_symbols(self):
        self._x_go = self.dp.X_w2id['<GO>']
        self._x_eos = self.dp.X_w2id['<EOS>']
        self._x_pad = self.dp.X_w2id['<PAD>']
        self._x_unk = self.dp.X_w2id['<UNK>']
        
        
        
    def infer(self, x_str):
        X_ind = [self.dp.X_w2id[w] for w in x_str.split()]
        X_pad_ind = [X_ind]
        #print(X_pad_ind)
        predict = self.sess.run(self.predictions, 
                {self.input_x: X_pad_ind,
                 self.X_seq_len:[len(X_ind)],
                self.output_keep_prob: 1.0,
                self.input_keep_prob:1.0})[0]
        return predict
    
    def restore(self, path):
        self.saver.restore(self.sess, path)
        print('restore %s success' % path)
        
    def get_learning_rate_decay(self, decay_scheme='luong234'):
        num_train_steps = self.dp.num_steps
        if decay_scheme == "luong10":
            start_decay_step = int(num_train_steps / 2)
            remain_steps = num_train_steps - start_decay_step
            decay_steps = int(remain_steps / 10)  # decay 10 times
            decay_factor = 0.5
        else:
            start_decay_step = int(num_train_steps * 2 / 3)
            remain_steps = num_train_steps - start_decay_step
            decay_steps = int(remain_steps / 4)  # decay 4 times
            decay_factor = 0.5
        return tf.cond(
            self.global_step < start_decay_step,
            lambda: self.learning_rate,
            lambda: tf.train.exponential_decay(
                self.learning_rate,
                (self.global_step - start_decay_step),
                decay_steps, decay_factor, staircase=True),
            name="learning_rate_decay_cond")
    
class BiLSTM_DP:
    def __init__(self, X_indices, C_labels, w2id, batch_size, n_epoch, split_ratio=0.1, test_data=None):
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
        #self.max_length = max_length
        self.num_batch = int(len(self.X_train) / batch_size)
        self.num_steps = self.num_batch * self.n_epoch
        self.batch_size = batch_size
        self.X_w2id = w2id
        self.X_id2w = dict(zip(w2id.values(), w2id.keys()))
        self._x_pad = w2id['<PAD>']
        print('Train_data: %d | Test_data: %d | Batch_size: %d | Num_batch: %d | vocab_size: %d' % (len(self.X_train), len(self.X_test), batch_size, self.num_batch, len(self.X_w2id)))
        
    def next_batch(self, X, C):
        r = np.random.permutation(len(X))
        X = X[r]
        C = C[r]
        for i in range(0, len(X) - len(X) % self.batch_size, self.batch_size):
            X_batch = X[i : i + self.batch_size]
            C_batch = C[i : i + self.batch_size]
            padded_X_batch, seq_lens = self.pad_sentence_batch(X_batch, self._x_pad)
            yield (np.array(padded_X_batch),
                   C_batch,
                  seq_lens)
    
    def sample_test_batch(self):
        i = random.randint(0, int(len(self.C_test) / self.batch_size)-2)
        C = self.C_test[i*self.batch_size:(i+1)*self.batch_size]
        padded_X_batch, seq_lens = self.pad_sentence_batch(self.X_test[i*self.batch_size:(i+1)*self.batch_size], self._x_pad)
        return np.array(padded_X_batch), C, seq_lens
    
        
    def pad_sentence_batch(self, sentence_batch, pad_int):
        padded_seqs = []
        seq_lens = []
        sentence_batch = sentence_batch.tolist()
        max_sentence_len = np.max([len(s) for s in sentence_batch])
        for sentence in sentence_batch:
            padded_seqs.append(sentence + [pad_int] * (max_sentence_len - len(sentence)))
            seq_lens.append(len(sentence))
        return padded_seqs, seq_lens  
    
class BiLSTM_Util:
    def __init__(self, dp, model, display_freq=3):
        self.display_freq = display_freq
        self.dp = dp
        self.D = model
        
    def train(self, epoch):
        avg_c_loss = 0.0
        avg_acc = 0.0
        tic = time.time()
        X_test_batch, C_test_batch, test_seq_lens  = self.dp.sample_test_batch()
        for local_step, (X_train_batch, C_train_batch, seq_lens) in enumerate(
            self.dp.next_batch(self.dp.X_train, self.dp.C_train)):
            #print(len(C_train_batch), len(X_train_batch))
            acc, loss, _ = self.D.sess.run([self.D.accuracy, self.D.d_loss, self.D.train_op], 
                {self.D.input_x: X_train_batch, 
                 self.D.input_y: C_train_batch, 
                 self.D.X_seq_len:seq_lens,
                 self.D.output_keep_prob:self.D._output_keep_prob,
                 self.D.input_keep_prob:self.D._input_keep_prob})
            avg_c_loss += loss
            avg_acc += acc
            if (local_step % int(self.dp.num_batch / self.display_freq)) == 0:
                val_acc, val_c_loss = self.D.sess.run([self.D.accuracy, self.D.d_loss], 
                                            {self.D.input_x: X_test_batch, 
                                            self.D.input_y: C_test_batch, 
                                            self.D.X_seq_len:test_seq_lens,
                                             self.D.output_keep_prob:1.0,
                                             self.D.input_keep_prob:1.0})
                print("Epoch %d/%d | Batch %d/%d | Train_loss: %.3f Acc %.3f | Test_loss: %.3f Acc %.3f | Time_cost:%.3f" % 
                      (epoch, self.n_epoch, local_step, self.dp.num_batch, avg_c_loss / (local_step + 1), avg_acc / (local_step + 1), val_c_loss, val_acc, time.time()-tic))
                self.cal()
                tic = time.time()
        return avg_c_loss / (local_step + 1), avg_acc / (local_step + 1)
    
    def test(self):
        avg_c_loss = 0.0
        avg_acc = 0.0
        tic = time.time()
        for local_step, (X_test_batch, C_test_batch, test_seq_lens) in enumerate(
            self.dp.next_batch(self.dp.X_test, self.dp.C_test)):
            acc, loss = self.D.sess.run([self.D.accuracy, self.D.d_loss], 
               {self.D.input_x: X_test_batch, 
                self.D.input_y: C_test_batch, 
                self.D.X_seq_len:test_seq_lens,
                 self.D.output_keep_prob:1.0,
                 self.D.input_keep_prob:1.0})
            avg_c_loss += loss
            avg_acc += acc
        return avg_c_loss / (local_step + 1), avg_acc / (local_step + 1)
    
    def fit(self, train_dir):
        self.n_epoch = self.dp.n_epoch
        out_dir = train_dir
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        print("Writing to %s" % out_dir)
        checkpoint_prefix = os.path.join(out_dir, "model")
        self.summary_writer = tf.summary.FileWriter(os.path.join(out_dir, 'Summary'), self.D.sess.graph)
        for epoch in range(1, self.n_epoch+1):
            tic = time.time()
            train_c_loss, train_acc = self.train(epoch)
            test_c_loss, test_acc = self.test()
            print("Epoch %d/%d | Train_loss: %.3f Acc %.3f | Test_loss: %.3f Acc %.3f" % 
                  (epoch, self.n_epoch, train_c_loss, train_acc, test_c_loss, test_acc))
            path = self.D.saver.save(self.D.sess, checkpoint_prefix, global_step=epoch)
            print("Saved model checkpoint to %s" % path)
    
    def show(self, sent, id2w):
        return " ".join([id2w.get(idx, u'&') for idx in sent])
    
    def cal(self, n_example=5):
        train_n_example = int(n_example / 2)
        test_n_example = n_example - train_n_example
        for _ in random.sample([t for t in range(len(self.dp.X_test))], test_n_example):
            example = self.show(self.dp.X_test[_], self.dp.X_id2w)
            o = self.D.infer(example)
            print('Test Input: %s | Output: %d | GroundTruth: %d' % (example, o, np.argmax(self.dp.C_test[_])))
        for _ in random.sample([t for t in range(len(self.dp.X_train))], train_n_example):
            example = self.show(self.dp.X_train[_], self.dp.X_id2w)
            o = self.D.infer(example)
            print('Train Input: %s | Output: %d | GroundTruth: %d' % (example, o, np.argmax(self.dp.C_train[_]))) 
        print("")