from __future__ import print_function
import numpy as np
import tensorflow as tf
import time
import os

from utils import TextLoader
from model import Model

#if __name__ = "__main__":
#    main()

def train(config, data, model):
    losses = []

    with tf.Session() as sess:
        tf.initialize_all_variables().run()

        for e in range(config.num_epochs):
            sess.run(tf.assign(model.lr, config.learning_rate * (config.decay_rate ** e)))
            data_loader.reset_batch_pointer()
            state = model.initial_state.eval()
            
            for b in range(data_loader.num_batches): # Mini-batches
                start = time.time()

                # Load the next batch of data
                x, y = data_loader.next_batch() 
                feed = {model.input_data: x, model.targets: y, 
                        model.initial_state: state}
                
                # Training step        
                train_loss, state, _ = sess.run(
                        [model.cost, model.final_state, model.train_op], feed)

                end = time.time()

                print("{}/{} (epoch {}), train_loss = {:.3f}, time/batch = {:.3f}" \
                    .format(e * data_loader.num_batches + b,
                            config.num_epochs * data_loader.num_batches,
                            e, train_loss, end - start))

                losses.append(train_loss)

    return losses

#def main()
#    data_loader = TextLoader(config.raw_data, config.batch_size,
#                    config.seq_length)
#    config.vocab_size = data_loader.vocab_size
#    
#    model = Model(config)
