#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Core modules
import os
import json
import time
import argparse
import random

# 3rd party modules
import torch

# Custom modules
from models import *
from data_utils import *
from utils import *


parser = argparse.ArgumentParser(description='PyTorch QGen')
parser.add_argument('--num_epochs', default=25, type=int,
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
                    help='whether to use reduced_glove')
parser.add_argument('--example_to_train', type=int, default=1000,
                    help='example taken to train')
parser.add_argument('--split_ratio', type=float, default=0.8,
                    help='ratio of training data')

# Hyperparameters
parser.add_argument('--hidden_size', type=int, default=300,
                    help='RNN hidden size')
parser.add_argument('--lr', type=float, default=3e-4,
                    help='Learning rate')
parser.add_argument('--tf_ratio', type=float, default=1.0,
                    help='Teacher Forcing Ratio')
# Model
parser.add_argument('--save', default='', type=str,
                    help='save the model after training')
parser.add_argument('--load', default='', type=str,
                    help='load the model')
parser.add_argument('--no_train', action='store_true', default=False,
                    help="don't start training")
parser.add_argument('--no_eval', action='store_true', default=False,
                    help="don't evaluate")
parser.add_argument('--gen', action='store_true', default=False,
                    help="Print Generated Questions")
parser.add_argument('--word_tf', action='store_true', default=False,
                    help="whether to use word or sentence based teacher forcing")
parser.add_argument('--unmasked_loss', action='store_true', default=False,
                    help="whether to use masked NLLLoss")


args = parser.parse_args()
print(args)

np.random.seed(args.seed)
torch.manual_seed(args.seed)
args.gpu = args.gpu and torch.cuda.is_available()

_word_to_idx = {}
_idx_to_word = []

if args.load_data != '':
    batches, num_batches, glove, _word_to_idx, _idx_to_word = load_obj(args.load_data)
else:
    batches, num_batches, glove, _word_to_idx, _idx_to_word = data_parse(args)
    if args.save_data != '':
        save_obj((batches, num_batches, glove, _word_to_idx, _idx_to_word), args.save_data)

print("Number of batches = ", num_batches)
max_doc_len = batches[0]['document_tokens'].shape[1]
max_q_len = batches[0]['question_input_tokens'].shape[1]


def look_up_word(word):
    return _word_to_idx.get(word, UNKNOWN_TOKEN)

def look_up_token(token):
    return _idx_to_word[token]


# Input: word token -> embedding
embedder = WordEmbedder(glove)
# embedding +ans -> ans_pred, encoded_doc, encoded_doc_h (2 * given hidden due to biLSTM)
doc_encoder = DocumentEncoder(embedder.output_size + 1, args.hidden_size, num_layers=1, bidirectional=True)
# encoded_doc, encoded_doc_h -> doubly encoded_doc, doubly_encoded_doc_h
q_encoder = BaseRNN(doc_encoder.output_size, 2*args.hidden_size)
# Most experimentation would be in input to DECODER
# doubly_encoded_doc_h + (context vec + encoder hidden), q_embedding -> qgen_dec_pred, qgen_dec_h
q_decoder = QuestionDecoder(embedder.output_size, 2*args.hidden_size, embedder.input_size)

if args.gpu:
    embedder.cuda()
    doc_encoder.cuda()
    q_encoder.cuda()
    q_decoder.cuda()


train_params = [ *list(doc_encoder.parameters()), *list(q_encoder.parameters()), *list(q_decoder.parameters()) ]
optimizer = torch.optim.Adam(train_params, lr=args.lr)
a_criterion = nn.BCELoss()
q_criterion =  nn.NLLLoss() if args.unmasked else MaskedNLLLoss()

def train_epoch(train_data, epoch):
    doc_encoder.train()
    q_encoder.train()
    q_decoder.train()

    print("No. of batches in training data: {}, with batch_size: {} ".format(len(train_data), len(train_data[0]['document_tokens'])))
    epoch_begin_time = time.time()
    avg_loss= 0

    for i, batch in enumerate(train_data):
        # assert batch size
        batch_size = len(batch['document_tokens'])
        if  batch_size != args.batch_size:
            print("Skipping batch {} due batch size mismatch.".format(i))
            continue

        optimizer.zero_grad()

        # make hidden zero for doc encoder
        dim1 = 2 if doc_encoder.bidirectional else 1
        doc_encoder_h = Variable(torch.zeros(dim1, batch_size, doc_encoder.hidden_size))

        # TODO: Doc mask & Question Mask
        # t_document_mask = Variable(torch.from_numpy(mask)).float()
        # mask for extra length of doc
        # doc_mask = torch.from_numpy(batch['doc_mask'])

        # Supervised Learning of "part of Answer" prediction
        doc_token = Variable(torch.from_numpy(batch['document_tokens']))
        answer_target = Variable(torch.from_numpy(batch['answer_labels']).float())

        if args.gpu:
            doc_encoder_h = doc_encoder_h.cuda()
            doc_token = doc_token.cuda()
            answer_target = answer_target.cuda()

        doc_embeddings = embedder(doc_token)
        # Adding additional dim. with answer tags
        doc_ans_embedding = torch.cat((doc_embeddings,answer_target.unsqueeze(-1)),dim=-1)

        answer_pred, doc_encoded, doc_encoder_h = doc_encoder(doc_ans_embedding, doc_encoder_h)
        a_loss = a_criterion(answer_pred.squeeze(), answer_target)

        # q_in for teacher forcing
        q_in_tf = torch.from_numpy(batch["question_input_tokens"]).long()
        # q_in for non-teacher forcing
        q_in = torch.from_numpy(np.full(q_in_tf[:,0].shape, look_up_word("<START>"))).long()
        q_in = q_in.unsqueeze(1)

        # q word targets for Sup. Learning
        q_target = Variable(torch.from_numpy(batch["question_output_tokens"]).long())
        if args.gpu:
            q_in_tf = q_in_tf.cuda()
            q_in = q_in.cuda()
            q_target = q_target.cuda()

        q_embedded_in = embedder(q_in)
        q_embedded_in_tf = embedder(q_in_tf)


        # Pass encoder hidden
        q_decoder_h = doc_encoder_h.view(1, batch_size, -1)

        # Set q_loss = 0 for batch
        q_loss = 0

        # Alternate between either word based or sentence based teacher forcing
        if args.word_tf:
            for q_len in range(max_q_len):
                use_teacher_forcing = True if random.random() < args.tf_ratio else False
                if use_teacher_forcing:
                    q_embedded_in = q_embedded_in_tf[:,q_len:q_len+1,:]
                    q_decoder_out, q_decoder_h = q_decoder(q_embedded_in, q_decoder_h)
                else:
                    q_decoder_out, q_decoder_h =  q_decoder(q_embedded_in, q_decoder_h)
                    _, q_in = q_decoder_out.max(2)
                    q_embedded_in = embedder(q_in)

                q_loss += q_criterion(q_decoder_out.squeeze(), q_target[:,q_len:q_len+1].squeeze())

        else:
            use_teacher_forcing = True if random.random() < args.tf_ratio else False

            if use_teacher_forcing:
                for q_len in range(max_q_len):
                    q_decoder_out, q_decoder_h =  q_decoder(q_embedded_in_tf[:,q_len:q_len+1,:], q_decoder_h)
                    q_loss += q_criterion(q_decoder_out.squeeze(), q_target[:,q_len:q_len+1].squeeze())
            else:
                for q_len in range(max_q_len):
                    q_decoder_out, q_decoder_h =  q_decoder(q_embedded_in, q_decoder_h)
                    q_loss += q_criterion(q_decoder_out.squeeze(), q_target[:,q_len:q_len+1].squeeze())
                    _, q_in = q_decoder_out.max(2)
                    q_embedded_in = embedder(q_in)

        # loss = q_loss + a_loss
        loss = q_loss
        loss.backward()
        optimizer.step()

        avg_loss+= loss.data[0]

        print ('Batch: %d \t Epoch : %d\tNet Loss: %.4f \tAnswer Loss: %.4f \tQuestion Loss: %.4f'
               %(i, epoch, loss.data[0], a_loss.data[0], q_loss.data[0]))

    print('Average Loss after Epoch %d : %.4f' %(epoch, avg_loss/num_batches))
    print("Epoch time: {:.2f}s".format(time.time() - epoch_begin_time))



def evaluate(data, generate=False):
    print("Evaluating:")
    doc_encoder.eval()
    q_encoder.eval()
    q_decoder.eval()

    if generate:
        max_q_len = data[0]["question_input_tokens"].shape[1]
        q_gen = {'gt':np.full((len(data),args.batch_size,max_q_len),'<END>',dtype=object),
                 'tf_gen':np.full((len(data),args.batch_size,max_q_len),'<END>',dtype=object),
                'full_gen':np.full((len(data),args.batch_size,max_q_len),'<END>',dtype=object),
                 }

    eval_begin_time = time.time()
    batch_loss = 0
    for i, batch in enumerate(data):
        # assert batch size
        batch_size = len(batch['document_tokens'])
        if batch_size != args.batch_size:
            continue

        # make hidden zero for doc encoder
        dim1 = 2 if doc_encoder.bidirectional else 1
        doc_encoder_h = Variable(torch.zeros(dim1, batch_size, doc_encoder.hidden_size))

        # Supervised Learning of "part of Answer" prediction
        doc_token = Variable(torch.from_numpy(batch['document_tokens']))
        answer_target = Variable(torch.from_numpy(batch['answer_labels']).float())

        if args.gpu:
            doc_encoder_h = doc_encoder_h.cuda()
            doc_token = doc_token.cuda()
            answer_target = answer_target.cuda()


        doc_embeddings = embedder(doc_token)
        # Adding additional dim. with answer tags
        doc_ans_embedding = torch.cat((doc_embeddings,answer_target.unsqueeze(-1)),dim=-1)

        answer_pred, doc_encoded, doc_encoder_h = doc_encoder(doc_ans_embedding, doc_encoder_h)
        a_loss = a_criterion(answer_pred.squeeze(), answer_target)

        # setting up decoder inputs and outputs
        q_in_tf = batch["question_input_tokens"]
        q_in_gen = np.full(q_in_tf[:,0].shape, look_up_word("<START>"))
        q_target = Variable(torch.from_numpy(batch["question_output_tokens"]).long())

        q_in_tf_tensor = torch.from_numpy(q_in_tf).long()
        q_in_gen_tensor = torch.from_numpy(q_in_gen).long()
        if args.gpu:
            q_in_tf_tensor = q_in_tf_tensor.cuda()
            q_in_gen_tensor = q_in_gen_tensor.cuda()
            q_target = q_target.cuda()

        q_embedded_in_tf = embedder(q_in_tf_tensor)
        q_embedded_in_gen = embedder(q_in_gen_tensor).unsqueeze(1)

        # Pass encoder hidden
        q_decoder_h = doc_encoder_h.view(1, batch_size, -1)
        # 2 hidden vectors for teacher forcing and fully generated
        q_decoder_h_tf = q_decoder_h.clone()
        q_decoder_h_gen = q_decoder_h.clone()

        # Set q_loss = 0 for batch
        q_loss_tf = 0
        q_loss_gen = 0
        for q_len in range(max_q_len):
            # teacher forcing
            q_decoder_out_tf, q_decoder_h_tf =  q_decoder(q_embedded_in_tf[:,q_len:q_len+1,:], q_decoder_h_tf)
            # full gen:
            q_decoder_out_gen, q_decoder_h_gen =  q_decoder(q_embedded_in_gen, q_decoder_h_gen)
            if args.gpu:
                q_out = np.argmax(q_decoder_out_gen.squeeze().cpu().data.numpy(), axis=1)
            else:
                q_out = np.argmax(q_decoder_out_gen.squeeze().data.numpy(), axis=1)
            q_in_gen = torch.from_numpy(q_out).long()
            if args.gpu:
                q_in_gen = q_in_gen.cuda()
            q_embedded_in_gen = embedder(q_in_gen).unsqueeze(1)

            # losses
            q_loss_tf += q_criterion(q_decoder_out_tf.squeeze(), q_target[:,q_len:q_len+1].squeeze())
            q_loss_gen += q_criterion(q_decoder_out_gen.squeeze(), q_target[:,q_len:q_len+1].squeeze())

            # storing for printing later
            if generate:
                q_gen['gt'][i,:,q_len] = np.array([look_up_token(j) for j in q_in_tf[:,q_len]])
                q_gen['full_gen'][i,:,q_len] = np.array([look_up_token(j) for j in q_out])

                if args.gpu:
                    q_out = np.argmax(q_decoder_out_tf.squeeze().cpu().data.numpy(), axis=1)
                else:
                    q_out = np.argmax(q_decoder_out_tf.squeeze().data.numpy(), axis=1)

                q_gen['tf_gen'][i,:,q_len] = np.array([look_up_token(j) for j in q_out])


        print ('Batch: %d\tQuestion Loss (teacher forcing): %.4f\tQuestion Loss (full generated): %.4f'
                   %(i, q_loss_tf.data[0], q_loss_gen.data[0]))

        batch_loss+= q_loss_gen.data[0]

    print('Average loss (full gen): %.4f' %( batch_loss/len(data)))
        # TODO: Eval Gen
    print("Eval time: {:.2f}s".format(time.time() - eval_begin_time))
    if generate:
        display_generated(q_gen)


def save(path):
    d = dict()
    # d['log'] = log
    d['doc_encoder'] = doc_encoder.state_dict()
    d['q_encoder'] = q_encoder.state_dict()
    d['q_decoder'] = q_decoder.state_dict()
    d['optimizer'] = optimizer.state_dict()
    torch.save(d, path)

def load(path):
    d = torch.load(path)
    # log.clear()
    doc_encoder.load_state_dict(d['doc_encoder'])
    q_encoder.load_state_dict(d['q_encoder'])
    q_decoder.load_state_dict(d['q_decoder'])
    optimizer.load_state_dict(d['optimizer'])
    # log.update(d['log'])
    # trainer.load_state_dict(d['trainer'])


if args.load != '':
    load(args.load)

split = int(args.split_ratio * len(batches))

if not args.no_train:
    for ep in range(args.num_epochs):
        train_epoch(batches[:split], ep)
        # Eval after each epoch from randomly chosen batch of val set
        b = [np.random.choice(batches[split:-1])]
        evaluate(b, generate=args.gen)

if args.save != '':
    save(args.save)

# Eval & generate questions
if not args.no_eval:
    evaluate(batches[split:], generate=args.gen)
