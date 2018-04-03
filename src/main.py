#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Core modules
import os
import json
import time
import argparse

# 3rd party modules
import torch

# Custom modules
from models import *
from data_utils import *
from utils import *



parser = argparse.ArgumentParser(description='PyTorch QGen')
parser.add_argument('--num_epochs', default=100, type=int,
                    help='number of training epochs')
parser.add_argument('--seed', type=int, default=42,
                    help='Random Seed')
parser.add_argument('--batch_size', type=int, default=32,
                    help='Batch Size')
parser.add_argument('--gpu', action='store_true', default=False,
                    help='Use GPU')
# Dataset related
parser.add_argument('--train_data', default='../train-v1.1.json', type=str,
                    help='path to train data')
parser.add_argument('--words_to_take', type=int, default=2000,
                    help='Size of reduced Glove (use 0 for full)')
parser.add_argument('--load_data', default='', type=str,
                    help='Load pickled data')
parser.add_argument('--save_data', default='', type=str,
                    help='save pickled data')
parser.add_argument('--reduced_glove', action='store_true', default=False,
                    help='Use reduced_glove')
parser.add_argument('--example_to_train', type=int, default=1000,
                    help='example taken to train')

# Model
parser.add_argument('--hidden_size', type=int, default=300,
                    help='RNN hidden size')
parser.add_argument('--save', default='', type=str,
                    help='save the model after training')
parser.add_argument('--load', default='', type=str,
                    help='load the model')
parser.add_argument('--no_train', action='store_true', default=False,
                    help='dont start the training')

args = parser.parse_args()
print(args)

np.random.seed(args.seed)
torch.manual_seed(args.seed)

# word_to_take = 2000
# words = findKMostFrequentWords(3* word_to_take)

_word_to_idx = {}
_idx_to_word = []

if args.load_data != '':
    batches, num_batches, glove, _word_to_idx, _idx_to_word = load_obj(args.load_data)
else:
    batches, num_batches, glove, _word_to_idx, _idx_to_word = data_parse(args)
    if args.save_data != '':
        save_obj((batches, num_batches, glove, _word_to_idx, _idx_to_word), args.save_data)

print("Number of batches = ", num_batches)
def look_up_word(word):
    return _word_to_idx.get(word, UNKNOWN_TOKEN)

def look_up_token(token):
    return _idx_to_word[token]


# Input: word token -> embedding
embedder = Embedder(glove)
# embedding -> ans_pred, encoded_doc, encoded_doc_h (2 * given hidden due to biLSTM)
doc_encoder = DocumentEncoder(embedder.output_size, args.hidden_size, num_layers=1, bidirectional=True)
# encoded_doc, encoded_doc_h -> doubly encoded_doc, doubly_encoded_doc_h
q_encoder = BaseRNN(doc_encoder.output_size, 2*args.hidden_size)
# doubly_encoded_doc_h + ans_embedding -> qgen_dec_pred, qgen_dec_h
q_decoder = QuestionDecoder(doc_encoder.output_size, 2*args.hidden_size, embedder.input_size)


train_params = [ *list(doc_encoder.parameters()), *list(q_encoder.parameters()), *list(q_decoder.parameters()) ]
optimizer = torch.optim.Adam(train_params, lr=3e-4)
criterion = nn.NLLLoss()


def train(train_data, num_epochs):
    doc_encoder.train()
    q_encoder.train()
    q_decoder.train()

    for ep in range(num_epochs):
        epoch_begin_time = time.time()
        avg_loss= 0

        for i, batch in enumerate(train_data):
            # assert batch size
            if len(batch) != args.batch_size:
                continue

            optimizer.zero_grad()

            # make hidden zero for doc encoder
            dim1 = 2 if doc_encoder.bidirectional else 1
            doc_encoder_h = Variable(torch.zeros(dim1, batch_size, self.hidden_size))

            # TODO: repackage of hidden

            # TODO: Doc mask & Question Mask
            # t_document_mask = Variable(torch.from_numpy(mask)).float()
            # mask for extra length of doc
            # doc_mask = torch.from_numpy(batch['doc_mask'])


            # Supervised Learning of "part of Answer" prediction
            doc_token = Variable(torch.from_numpy(batch['document_tokens']).long())
            answer_target = Variable(torch.from_numpy(batch['answer_labels']))
            
            doc_embeddings = embedder(doc_token)
            answer_pred, doc_encoded, doc_encoder_h = doc_encoder(doc_embeddings, doc_encoder_h)
            a_loss = criterion(answer_pred.squeeze(), labels.squeeze())

            # TODO: choose context vectors
            # outputs = torch.mul(answer_tags.squeeze(-1),t_document_mask)
            answer_mask = answer_target.unsqueeze(-1)
            context_mask = 1 - answer_mask

            q_encoder_in = answer_mask * doc_encoded
            q_encoder_o, q_encoder_h = q_encoder(q_encoder_in, doc_encoder_h)

            q_embedded_in = embedder(torch.from_numpy(batch["question_input_tokens"]))
            q_target = Variable(torch.from_numpy(batch_input[batch_num]["question_output_tokens"]).long())

            # Pass encoder hidden
            q_decoder_h = q_encoder_h

            # Set q_loss = 0 for batch
            q_loss = 0
            # for q_len in range(q_embedded_in.shape):
            for q_len in range(batch["question_input_tokens"].shape[1]):
                q_decoder_out, q_decoder_h =  q_decoder(q_embedded_in[:,q_len:q_len+1,:], q_decoder_h)

                q_loss += criterion(q_decoder_out, q_target[:,q_len:q_len+1].squeeze())

            loss = q_loss + a_loss
            loss.backward()
            optimizer.step()

            avg_loss+= loss.data[0]
        
            print ('Batch: %d \t Epoch : %d\tNet Loss: %.4f \tAnswer Loss: %.4f \tQuestion Loss: %.4f'
                   %(i, ep, loss.data[0], a_loss.data[0], q_loss.data[0]))

        print('Average Loss after Epoch %d : %.4f' %(ep, avg_loss/num_batches))
        # TODO: Eval
        print("Epoch time: ".format(time.time() - epoch_begin_time))


def eval(data, generated=False):
    doc_encoder.eval()
    q_encoder.eval()
    q_decoder.eval()

    eval_begin_time = time.time()    
    for i, batch in enumerate(data):
        # assert batch size
        if len(batch) != args.batch_size:
            continue

        # make hidden zero for doc encoder
        dim1 = 2 if doc_encoder.bidirectional else 1
        doc_encoder_h = Variable(torch.zeros(dim1, batch_size, self.hidden_size))

        # Supervised Learning of "part of Answer" prediction
        doc_token = Variable(torch.from_numpy(batch['document_tokens']).long())
        answer_target = Variable(torch.from_numpy(batch['answer_labels']))
        
        doc_embeddings = embedder(doc_token)
        answer_pred, doc_encoded, doc_encoder_h = doc_encoder(doc_embeddings, doc_encoder_h)
        a_loss = criterion(answer_pred.squeeze(), labels.squeeze())

        # TODO: choose context vectors
        # outputs = torch.mul(answer_tags.squeeze(-1),t_document_mask)
        answer_mask = answer_target.unsqueeze(-1)
        context_mask = 1 - answer_mask

        q_encoder_in = answer_mask * doc_encoded
        q_encoder_o, q_encoder_h = q_encoder(q_encoder_in, doc_encoder_h)

        q_embedded_in = embedder(torch.from_numpy(batch["question_input_tokens"]))
        q_target = Variable(torch.from_numpy(batch_input[batch_num]["question_output_tokens"]).long())

        # Pass encoder hidden
        q_decoder_h = q_encoder_h

        # Set q_loss = 0 for batch
        q_loss = 0
        # for q_len in range(q_embedded_in.shape):
        for q_len in range(batch["question_input_tokens"].shape[1]):
            q_decoder_out, q_decoder_h =  q_decoder(q_embedded_in[:,q_len:q_len+1,:], q_decoder_h)

            q_loss += criterion(q_decoder_out, q_target[:,q_len:q_len+1].squeeze())

        loss = q_loss + a_loss

        print ('Batch: %d \t Epoch : %d\tNet Loss: %.4f \tAnswer Loss: %.4f \tQuestion Loss: %.4f'
                   %(i, ep, loss.data[0], a_loss.data[0], q_loss.data[0]))

    batch_loss+= loss.data[0]
        
    print('Average loss : %.4f' %(i, avg_loss/len(data)))
        # TODO: Eval
    print("Eval time: ".format(time.time() - eval_begin_time))



def save(path):
    d = dict()
    d['policy_net'] = policy_net.state_dict()
    d['log'] = log
    d['trainer'] = trainer.state_dict()
    torch.save(d, path)

def load(path):
    d = torch.load(path)
    log.clear()
    policy_net.load_state_dict(d['policy_net'])
    log.update(d['log'])
    trainer.load_state_dict(d['trainer'])


if args.load != '':
    load(args.load)

if not args.no_train:
    train(batches, args.num_epochs)

if args.save != '':
    save(args.save)

# TODO: Eval