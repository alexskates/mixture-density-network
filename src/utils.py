import collections
import numpy as np

class Config():
    def __init__(self, raw_data, save_dir='save', embedding_size=6, 
            hidden_size=128, num_layers=2, cell='lstm', batch_size=50, 
            seq_length=50, num_epochs=50, save_every=1000, keep_prob=1.0,
            grad_clip=5.0, learning_rate=0.001, decay_rate=0.97,
            init_from=None):
        self.raw_data= raw_data # Raw state time courses, np array
        self.save_dir = save_dir # Where to store checkpoints
        self.embedding_size = embedding_size # Dimensions of embeddings
        self.rnn_size = hidden_size # Number of RNN hidden states
        self.num_layers = num_layers # Number of layers in RNN
        self.cell = cell # RNN, LSTM or GRU
        self.batch_size = batch_size # Mini-batch size
        self.seq_length = seq_length # RNN sequence length
        self.num_epochs = num_epochs # Number of training epochs
        self.save_every = save_every # Checkpointing frequency
        self.grad_clip = grad_clip # Clip the gradients at this value
        self.keep_prob = keep_prob # Keep prob for dropout
        self.learning_rate = learning_rate # Learning rate
        self.decay_rate = decay_rate # RMSProp decay rate
        self.init_from = init_from # Continut training from saved model at path
        self.vocab_size = None # Number of states. Loaded in training func
        

class TextLoader():
    def __init__(self, raw_data, batch_size, seq_length):
        """ Arguments:
        raw_data: a numpy array of the state sequences
        batch_size: the size of the mini-batch
        seq-length: the RNN sequence length """

        self.batch_size = batch_size
        self.seq_length = seq_length

        # Process the data
        self.process(raw_data)

        self.create_batches()
        self.reset_batch_pointer()

    # Creates self.tensor, a numpy array corresponding to the index of each
    # element in self.chars.
    def process(self, raw_data):
        counter = collections.Counter(raw_data)
        # Sort the items by count descending
        count_pairs = sorted(counter.items(), key=lambda x: -x[1])
        self.chars, _ = zip(*count_pairs) # List of different states
        self.vocab_size = len(self.chars) # Total number of states
        self.vocab = dict(zip(self.chars, range(len(self.chars))))
        self.tensor = np.array(list(map(self.vocab.get, raw_data)))

    def create_batches(self):
        self.num_batches = self.tensor.size // (
                            self.batch_size * self.seq_length)

        if self.num_batches==0:
            assert False, "Not enough data. Reduce seq_length/batch_size"

        self.tensor = self.tensor[:self.num_batches * 
                        self.batch_size * self.seq_length]
        xdata = self.tensor
        # ydata is xdata shifted one to the left (e.g. the output is the next
        # element of the sequence
        ydata = np.copy(self.tensor)
        ydata[:-1] = xdata[1:]
        ydata[-1] = xdata[0]
        self.x_batches = np.split(xdata.reshape(self.batch_size, -1),
                        self.num_batches, 1)
        self.y_batches = np.split(ydata.reshape(self.batch_size, -1), 
                        self.num_batches, 1)


    def next_batch(self):
        x, y = self.x_batches[self.pointer], self.y_batches[self.pointer]
        self.pointer += 1
        return x, y

    def reset_batch_pointer(self):
        self.pointer = 0
