import numpy as np
import torch

class Trainer(object):
    def __init__(self, args, embedder, encoder, decoder, data):
        split = int(args.split_ratio * len(data))
        self.train_data = data[:split]
        self.val_data = data[split:-1]

        self.embedder = embedder
        self.encoder = encoder
        self.decoder = decoder
