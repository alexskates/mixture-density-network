import tensorflow as tf
import numpy as np

class Model():
    def __init__(self, config, infer=False):
        self.config = config
        if infer:
            config.batch_size = 1
            config.seq_length = 1

        if config.cell == 'rnn':
            cell_fn = tf.nn.rnn_cell.BasicRNNCell
        elif config.cell == 'gru':
            cell_fn = tf.nn.rnn_cell.GRUCell
        elif config.cell == 'lstm':
            cell_fn = tf.nn.rnn_cell.BasicLSTMCell
        else:
            raise Exception("model type not supported: {}".format(config.cell))

        # Placeholders for input data:w
        self.input_data = tf.placeholder(tf.int32,
                            [config.batch_size, config.seq_length], "inputs")
        self.targets = tf.placeholder(tf.int32,
                            [config.batch_size, config.seq_length], "targets")

        # Create internal multi-layer cell for RNN
        cell = cell_fn(config.rnn_size)
        self.cell = cell = tf.nn.rnn_cell.MultiRNNCell([cell] * config.num_layers)

        # Initial state is all zeros
        self.initial_state = cell.zero_state(config.batch_size, tf.float32)

        with tf.variable_scope('rnnlm'):
            softmax_w = tf.get_variable("softmax_w",
                        [config.rnn_size, config.vocab_size])
            softmax_b = tf.get_variable("softmax_b",
                        [config.vocab_size])
            with tf.device("/cpu:0"):
                # Embeddings for inputs
                embedding = tf.get_variable("embedding",
                            [config.vocab_size, config.embedding_size])
                inputs = tf.split(1, config.seq_length,
                            tf.nn.embedding_lookup(embedding, self.input_data))
                inputs = [tf.squeeze(input_, [1]) for input_ in inputs]

        def loop(prev, _):
            prev = tf.matmul(prev, softmax_w) + softmax_b
            prev_symbol = tf.stop_gradient(tf.argmax(prev, 1))
            return tf.nn.embedding_lookup(embedding, prev_symbol)

        # Loop function defined above will be applied to the output of the
        # decoder and reused as input instead of using decoder_inputs
        outputs, last_state = tf.nn.seq2seq.rnn_decoder(inputs, self.initial_state, 
                                cell, loop_function=loop if infer else None,
                                scope='rnnlm')

        output = tf.reshape(tf.concat(1, outputs), [-1, config.rnn_size])

        self.logits = tf.matmul(output, softmax_w) + softmax_b
        self.probs = tf.nn.softmax(self.logits)
        loss = tf.nn.seq2seq.sequence_loss_by_example([self.logits],
                [tf.reshape(self.targets, [-1])],
                [tf.ones([config.batch_size * config.seq_length])],
                config.vocab_size)
        self.cost = tf.reduce_sum(loss) / config.batch_size / config.seq_length

        # Accuracy
        self.predictions = tf.argmax(self.probs, 1)
        correct_predictions = tf.equal(tf.to_int32(self.predictions), 
            tf.reshape(self.targets, [-1]))
        self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"))

        self.final_state = last_state

        # Process the gradients before applying them
        self.lr = tf.Variable(0.0, trainable=False)
        tvars = tf.trainable_variables()
        grads, _ = tf.clip_by_global_norm(tf.gradients(self.cost, tvars),
                    config.grad_clip)  

        # Training operations
        optimizer = tf.train.AdamOptimizer(self.lr)
        self.train_op = optimizer.apply_gradients(zip(grads, tvars))

    def sample(self, sess, chars, vocab, num=200, prime=1, 
            sampling_type=1):
        state = self.cell.zero_state(1, tf.float32).eval()
        for char in prime[:-1]:
            x = np.zeros((1, 1))
            x[0, 0] = vocab[char]
            feed = {self.input_data: x, self.initial_state:state}
            [state] = sess.run([self.final_state], feed)

        def weighted_pick(weights):
            t = np.cumsum(weights)
            s = np.sum(weights)
            return(int(np.searchsorted(t, np.random.rand(1)*s)))

        ret = prime
        char = prime[-1]
        for n in range(num):
            x = np.zeros((1, 1))
            x[0, 0] = vocab[char]
            feed = {self.input_data: x, self.initial_state:state}
            [probs, state] = sess.run([self.probs, self.final_state], feed)
            p = probs[0]

            if sampling_type == 0:
                sample = np.argmax(p)
            elif sampling_type == 2:
                if char == ' ':
                    sample = weighted_pick(p)
                else:
                    sample = np.argmax(p)
            else: # sampling_type == 1 default:
                sample = weighted_pick(p)

            pred = chars[sample]
            ret += pred
            char = pred
        return ret


