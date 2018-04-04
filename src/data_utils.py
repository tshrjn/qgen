#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import numpy as np


def data_parse(args):
    from constants import _add_word, UNKNOWN_WORD, START_WORD, END_WORD, UNKNOWN_TOKEN, START_TOKEN, END_TOKEN, _word_to_idx, _idx_to_word

    def look_up_word(word):
        return _word_to_idx.get(word, UNKNOWN_TOKEN)

    def look_up_token(token):
        return _idx_to_word[token]

# from nltk.tokenize import ToktokTokenizer
# toktok = ToktokTokenizer().tokenize
# if True:
#     word_tokenize = toktok
    embeddings_path = os.path.realpath(args.glove_path)
    with open(embeddings_path, encoding='utf-8') as f:
        line = f.readline()
        chunks = line.split(" ")
        dimensions = len(chunks) - 1
        f.seek(0)

        vocab_size = sum(1 for line in f)
        vocab_size += 3
        f.seek(0)

        glove = np.ndarray((vocab_size, dimensions), dtype=np.float32)
        glove[UNKNOWN_TOKEN] = np.zeros(dimensions)
        glove[START_TOKEN] = -np.ones(dimensions)
        glove[END_TOKEN] = np.ones(dimensions)

        for line in f:
            chunks = line.split(" ")
            idx = _add_word(chunks[0])
            if idx >= vocab_size:
                break
            glove[idx] = [float(chunk) for chunk in chunks[1:]]


    import json
    import pickle
    import operator
    from collections import Counter


    from nltk.corpus import stopwords
    from nltk.tokenize import word_tokenize
    nltkStopWords = stopwords.words('english')

    punctuations = [',', '?', '.', '-',]
    max_document_len = 0
    max_answer_len = 0
    max_question_len = 0


    data = json.load(open(args.train_data))
    np.random.seed(args.seed)


    # In[4]:


    nltkStopWords = stopwords.words('english')
    punctuations = [',', '?', '.', '-',]


    # In[5]:


    def extractor(data):
        contexts = []
        qas = []
        for i in range(len(data["data"])):
            for j in range(len(data["data"][i]["paragraphs"])):
                contexts.append(data["data"][i]["paragraphs"][j]["context"])
                qas.append(data["data"][i]["paragraphs"][j]["qas"])
        return (contexts,qas)


    # In[6]:


    CapPassage = False

    contexts,qas = extractor(data)

    def find_sub_list(sl,l):
        sll=len(sl)
        for ind in (i for i,e in enumerate(l) if e==sl[0]):
            if l[ind:ind+sll]==sl:
                return ind,ind+sll
        return (-1,-1)

    def capPassage(passage,answer,cap_length = 30):
        y = np.zeros(cap_length)
        left,right = find_sub_list(answer,passage)
        if(left==-1):
            return passage[0:cap_length]
        left = left - int((cap_length - len(answer))/2)
        right = right + int((cap_length + len(answer))/2)
        if(left < 0):
            left = 0
        if(right > len(passage)):
            right = len(passage)
        return passage[left:right]

    def findAnsVec(answer,passage):
        ans = np.zeros((len(passage)))
        start,end = find_sub_list(answer,passage)
        if(start==-1):
            start = passage.index(answer[0])
            end = start + len(answer)
        ans[start:end] = 1
        return ans


    # In[7]:


    X_train_comp_all = []
    X_train_comp_with_answer_marked_all = []
    X_train_ans_all = []
    X_train_comp_answer_label_all = []
    Y_train_ques_all = []
    invalid = 0
    for i,context in enumerate(contexts):
        passage = word_tokenize(context.lower())

        a_lab = np.zeros(len(passage))
        for j,_ in enumerate(qas[i]):
            answer = word_tokenize(qas[i][j]["answers"][0]['text'].lower())
            start,end = find_sub_list(answer,passage)
            if start == -1:
                invalid=invalid+1
                continue
            a_lab[start:end] = 1


        for j,_ in enumerate(qas[i]):
            try:
                question = word_tokenize(qas[i][j]['question'].lower())
                answer = word_tokenize(qas[i][j]["answers"][0]['text'].lower())
                start,end = find_sub_list(answer,passage)
                if start == -1:
                    invalid = invalid+1
                    continue
                marked_comp = np.zeros(len(passage))
                marked_comp[start:end] = 1

                if CapPassage:
                    cappedPassage = capPassage(passage,answer)
                else:
                    cappedPassage = passage

                X_train_comp_all.append(cappedPassage)
                X_train_comp_with_answer_marked_all.append(marked_comp)
                X_train_ans_all.append(answer)
                X_train_comp_answer_label_all.append(a_lab)
                Y_train_ques_all.append(question)
            except Exception as e:
                invalid = invalid+1


    # In[8]:


    def findKMostFrequentWords(k):
        ctr = Counter([item for sublist in X_train_comp_all for item in sublist] + [item for sublist in Y_train_ques_all for item in sublist])
        sorted_ctr = sorted(ctr.items(), key=operator.itemgetter(1), reverse=True)
        return [item[0] for item in sorted_ctr[0:k]]


    # In[189]:

    if args.words_to_take == 0:
        args.words_to_take  = len(glove)

    words = findKMostFrequentWords(min(len(glove), 3* args.words_to_take))


    # In[190]:


    _word_to_idx_reduced = {}
    _idx_to_word_reduced = []


    def _add_word_reduced(word):
        idx = len(_idx_to_word_reduced)
        _word_to_idx_reduced[word] = idx
        _idx_to_word_reduced.append(word)
        return idx


    UNKNOWN_TOKEN = _add_word_reduced(UNKNOWN_WORD)
    START_TOKEN = _add_word_reduced(START_WORD)
    END_TOKEN = _add_word_reduced(END_WORD)


    dimensions = glove.shape[1]
    reduced_glove = []
    reduced_glove.append(np.zeros(dimensions))
    reduced_glove.append(-np.ones(dimensions))
    reduced_glove.append(np.ones(dimensions))

    for i, word in enumerate(words):
        l = look_up_word(word)
        if(l != UNKNOWN_TOKEN):
            idx = _add_word_reduced(word)
            reduced_glove.append(glove[l])
            if(args.words_to_take!=0 and len(reduced_glove) == args.words_to_take):
                break

    print("No of words in (reduced) glove:", len(reduced_glove))

    def look_up_word_reduced(word):
        return _word_to_idx_reduced.get(word, UNKNOWN_TOKEN)


    def look_up_token_reduced(token):
        return _idx_to_word_reduced[token]

    reduced_glove = np.array(reduced_glove)
    reduced_glove.shape


    print("No. of invalid words/examples:", invalid)

    # In[12]:

    c = list(zip(X_train_comp_all,X_train_ans_all, Y_train_ques_all,X_train_comp_with_answer_marked_all, X_train_comp_answer_label_all))
    np.random.shuffle(c)
    X_train_comp_all_shuffled,X_train_ans_all_shuffled, Y_train_ques_all_shuffled, X_train_comp_with_answer_marked_all_shuffled, X_train_comp_answer_label_shuffled = zip(*c)


    # In[13]:


    examples_to_take_train = args.example_to_train if args.example_to_train !=0 else len(X_train_comp_all_shuffled)

    X_train_comp = X_train_comp_all_shuffled[0:examples_to_take_train]
    X_train_ans = X_train_ans_all_shuffled[0:examples_to_take_train]
    Y_train_ques = Y_train_ques_all_shuffled[0:examples_to_take_train]
    X_train_comp_with_answer_marked = X_train_comp_with_answer_marked_all_shuffled[0:examples_to_take_train]
    X_train_comp_answer_label = X_train_comp_answer_label_shuffled[0:examples_to_take_train]


    # In[14]:


    max_document_len = len(max(X_train_comp,key=len))
    max_answer_len = len(max(X_train_ans,key=len))
    max_question_len = len(max(Y_train_ques,key=len)) + 1


    # In[15]:


    X_train_comp_with_answer_marked[0], max_document_len


    # In[16]:


    document_tokens = np.full((examples_to_take_train, max_document_len), END_TOKEN,dtype=np.int32)
    document_lengths = np.zeros(examples_to_take_train, dtype=np.int32)

    answer_labels_all = np.zeros((examples_to_take_train, max_document_len), dtype=np.int32)
    answer_labels = np.zeros((examples_to_take_train, max_document_len), dtype=np.int32)
    answer_lengths = np.zeros(examples_to_take_train, dtype=np.int32)

    question_input_tokens = np.full((examples_to_take_train, max_question_len), END_TOKEN, dtype=np.int32)
    question_output_tokens = np.full((examples_to_take_train, max_question_len), END_TOKEN, dtype=np.int32)
    question_lengths = np.zeros(examples_to_take_train, dtype=np.int32)


    for i in range(examples_to_take_train):
        answer_labels_all[i,0:len(X_train_comp_answer_label[i])] = X_train_comp_answer_label[i]
        answer_labels[i, 0:len(X_train_comp_with_answer_marked[i])] = X_train_comp_with_answer_marked[i]
        for j, word in enumerate(X_train_comp[i]):
            document_tokens[i, j] = look_up_word_reduced(word)
        document_lengths[i] = len(X_train_comp[i])

        answer_lengths[i] = len(X_train_ans[i])

        question_input_words = ([START_WORD] + Y_train_ques[i])
        question_output_words = (Y_train_ques[i] + [END_WORD])

        for j, word in enumerate(question_input_words):
                question_input_tokens[i, j] = look_up_word_reduced(word)
        for j, word in enumerate(question_output_words):
            question_output_tokens[i, j] = look_up_word_reduced(word)
        question_lengths[i] = len(question_input_words)


    def create_vocabulary(data):
        flat_list = [item for sublist in data for item in sublist]
        vocabulary = sorted(set(flat_list))
        vocabulary.append("<UNK>")
        vocabulary.append("unk")
        vocabulary.append("eos")
        vocabulary = ["<EOS>"] + vocabulary
        word_to_index = { word:i for i,word in enumerate(vocabulary) }
        index_to_word = { i:word for i,word in enumerate(vocabulary) }
        return (vocabulary,word_to_index,index_to_word)


    vocabulary_comp,word_to_index_comp,index_to_word_comp = create_vocabulary(X_train_comp + Y_train_ques)


    def create_one_hot_vector(data,vocabulary,word_to_index,index_to_word, maxLen):
        one_hot = np.zeros([maxLen,len(vocabulary)])
        for i,word in enumerate(data):
            if i >= maxLen:
                break
            if(word not in word_to_index):
                word = "<UNK>"
            one_hot[i][word_to_index[word]] = 1
        return one_hot

    def create_one_hot_vector_from_indices(data,maxLen,vocabulary):
        one_hot = np.zeros([maxLen,len(vocabulary)])
        for i,indice in enumerate(data):
            if i >= maxLen:
                break
            one_hot[i][int(indice)] = 1
        return one_hot


    def create_one_hot_training_Set(data,maxLen,vocabulary):
        one_hot_data = np.zeros([data.shape[0],maxLen,len(vocabulary)])
        for i in range(data.shape[0]):
            one_hot_data[i] = create_one_hot_vector_from_indices(data[i],maxLen,vocabulary)
        return one_hot_data

    def context_to_indices_glove(X,max_len):

        m = len(X)

        X_indices = np.full([m,max_len],look_up_word_reduced(END_WORD))

        for i in range(m):
            j = 0
            for w in X[i]:
                if(j>=max_len):
                    break;

                X_indices[i, j] = look_up_word_reduced(w)
                j = j+1
        return X_indices


    document_tokens = context_to_indices_glove(X_train_comp, max_document_len)


    def createBatch(inputs,batch_size,shuffle=False):
        outputs = []

        num_batches = len(inputs[0]) // batch_size + 1

        for i in range(num_batches-1):
            start = 0

            output = {'document_tokens':[],
                        'document_lengths':[],
                        'answer_labels_all':[],
                        'answer_labels':[],
                        'answer_lengths': [],
                        'question_input_tokens':[],
                        'question_output_tokens':[],
                        'question_lengths':[]}

            for index,inp in enumerate(inputs):
                #maxD = max(inputs[1][start:start+batch_size])
                maxD = max_document_len
                maxA = max(inputs[4][start:start+batch_size])
                maxQ = max_question_len

                if index == 0:
                    output['document_tokens'].append(inp[start:start+batch_size,0:maxD])
                elif index==1:
                    output['document_lengths'].append(inp[start:start+batch_size])
                elif index==2:
                    output['answer_labels_all'].append(inp[start:start+batch_size,0:maxD])
                elif index==3:
                    output['answer_labels'].append(inp[start:start+batch_size,0:maxD])
                elif index==4:
                    output['answer_lengths'].append(inp[start:start+batch_size])
                elif index==5:
                    output['question_input_tokens'].append(inp[start:start+batch_size, 0:maxQ])
                elif index==6:
                    output['question_output_tokens'].append(inp[start:start+batch_size, 0:maxQ])
                elif index==7:
                    output['question_lengths'].append(inp[start:start+batch_size])

            output["document_tokens"] = np.array(output["document_tokens"])
            output["document_lengths"] = np.array(output["document_lengths"])
            output["answer_labels_all"] = np.array(output["answer_labels_all"])
            output["answer_labels"] = np.array(output["answer_labels"])
            output["answer_lengths"] = np.array(output["question_lengths"])
            output["question_input_tokens"] = np.array(output["question_input_tokens"])
            output["question_output_tokens"] = np.array(output["question_output_tokens"])
            output["question_lengths"] = np.array(output["question_lengths"])
            outputs.append(output)
            start = start + batch_size

        # Remaining training sample i.e. training mod batch_size
        #maxD = max(inputs[1][start:])
        maxD = max_document_len
        maxA = max(inputs[4][start:])
        maxQ = max_question_len
        output = {'document_tokens':[],
                    'document_lengths':[],
                    'answer_labels_all':[],
                    'answer_labels':[],
                    'answer_lengths': [],
                    'question_input_tokens':[],
                    'question_output_tokens':[],
                    'question_lengths':[]}
        if index == 0:
            output['document_tokens'].append(inp[start:,0:maxD])
        elif index==1:
            output['document_lengths'].append(inp[start:])
        elif index==2:
            output['answer_labels_all'].append(inp[start:,0:maxD])
        elif index==3:
            output['answer_labels'].append(inp[start:,0:maxD])
        elif index==4:
            output['answer_lengths'].append(inp[start:])
        elif index==5:
            output['question_input_tokens'].append(inp[start:, 0:maxQ])
        elif index==6:
            output['question_output_tokens'].append(inp[start:, 0:maxQ])
        elif index==7:
            output['question_lengths'].append(inp[start:])

        output["document_tokens"] = np.array(output["document_tokens"])
        output["document_lengths"] = np.array(output["document_lengths"])
        output["answer_labels"] = np.array(output["answer_labels"])
        output["answer_labels_all"] = np.array(output["answer_labels_all"])
        output["answer_lengths"] = np.array(output["question_lengths"])
        output["question_input_tokens"] = np.array(output["question_input_tokens"])
        output["question_output_tokens"] = np.array(output["question_output_tokens"])
        output["question_lengths"] = np.array(output["question_lengths"])

        outputs.append(output)

        return outputs


    # In[191]:


    batch_input = createBatch([document_tokens,document_lengths,answer_labels_all,answer_labels,answer_lengths,question_input_tokens,question_output_tokens,question_lengths]
                        ,args.batch_size)
    for b in batch_input:
        for k, v in b.items():
            b[k] = v.squeeze()
    number_of_batches = len(batch_input)


    return batch_input, number_of_batches, reduced_glove, _word_to_idx_reduced, _idx_to_word_reduced

'''
X_train_comp_all=[]
Y_train_ques_all=[]

def create_vocabulary(data):
    flat_list = [item for sublist in data for item in sublist]
    vocabulary = sorted(set(flat_list))
    vocabulary.append("<UNK>")
    vocabulary.append("unk")
    vocabulary.append("eos")
    vocabulary = ["<EOS>"] + vocabulary
    word_to_index = { word:i for i,word in enumerate(vocabulary) }
    index_to_word = { i:word for i,word in enumerate(vocabulary) }
    return (vocabulary,word_to_index,index_to_word)


def capPassage(passage,answer,cap_length = 30):
    y = np.zeros(cap_length)
    left,right = find_sub_list(answer,passage)
    if(left==-1):
        return passage[0:cap_length]
    left = left - int((cap_length - len(answer))/2)
    right = right + int((cap_length + len(answer))/2)
    if(left < 0):
        left = 0
    if(right > len(passage)):
        right = len(passage)
    return passage[left:right]

def find_sub_list(sl,l):
    sll=len(sl)
    for ind in (i for i,e in enumerate(l) if e==sl[0]):
        if l[ind:ind+sll]==sl:
            return ind,ind+sll
    return (-1,-1)

def findAnsVec(answer,passage):
    ans = np.zeros((len(passage)))
    start,end = find_sub_list(answer,passage)
    if(start==-1):
        start = passage.index(answer[0])
        end = start + len(answer)
    ans[start:end] = 1
    return ans

def context_separator(data):
   returns tuple of (contexts,qas)
   context: list of context (i.e. str)
   qa: list of dict each dict has keys: 'answers, id, question'
       'answer' values dict of keys 'answer_start', 'text'
    contexts = []
    qas = []
    for i in range(len(data["data"])):
        for j in range(len(data["data"][i]["paragraphs"])):
            contexts.append(data["data"][i]["paragraphs"][j]["context"])
            qas.append(data["data"][i]["paragraphs"][j]["qas"])
    return (contexts,qas)


def squad_parser(contexts, qas, examples_to_take_train=None, CapPassage=False):
    global X_train_comp_all
    global Y_train_ques_all

    X_train_comp_all = []
    X_train_comp_ans_all = []
    X_train_ans_all = []
    Y_train_ques_all = []
    invalid = 0
    X_train_ans_label_all = []

    #
    for i,context in enumerate(contexts):
        passage = word_tokenize(context.lower())

        a_lab = np.zeros(len(passage))
        for j,_ in enumerate(qas[i]):
            answer = word_tokenize(qas[i][j]["answers"][0]['text'].lower())
            start,end = find_sub_list(answer,passage)
            if start == -1:
                continue
            a_lab[start:end+1] = 1


        for j,_ in enumerate(qas[i]):
            try:
                question = word_tokenize(qas[i][j]['question'].lower())
                answer = word_tokenize(qas[i][j]["answers"][0]['text'].lower())

                if CapPassage:
                    cappedPassage = capPassage(passage,answer)
                else:
                    cappedPassage = passage

                X_train_comp_ans_all.append(findAnsVec(answer,passage))
                X_train_ans_label_all.append(a_lab)
                X_train_comp_all.append(cappedPassage)
                X_train_ans_all.append(answer)
                Y_train_ques_all.append(question)
            except Exception as e:
                # TODO: using a better tokenizer - 1500 q-ans pairs affected
                print("Exception: ",e)
                invalid = invalid+1

    print("No. of Invalid:", invalid)
    c = list(zip(X_train_comp_all,X_train_comp_ans_all, X_train_ans_all, X_train_ans_label_all,Y_train_ques_all))
    np.random.shuffle(c)
    X_train_comp_all_shuffled,X_train_comp_ans_all_shuffled, X_train_ans_shuffled, X_train_ans_label_shuffled,Y_train_ques_all_shuffled = zip(*c)

    print("examples_to_take_train", examples_to_take_train)

    X_train_comp = X_train_comp_all_shuffled[0:examples_to_take_train]
    X_train_comp_ans = X_train_comp_ans_all_shuffled[0:examples_to_take_train]
    X_train_ans = X_train_ans_shuffled[0:examples_to_take_train]
    X_train_ans_label = X_train_ans_label_shuffled[0:examples_to_take_train]
    Y_train_ques = Y_train_ques_all_shuffled[0:examples_to_take_train]
    answer_indices = [np.where(x==1)[0].tolist() for x in X_train_comp_ans]

    return (X_train_comp, X_train_comp_ans, X_train_ans, X_train_ans_label,Y_train_ques,answer_indices)

def builder(X_train_comp, X_train_comp_ans, X_train_ans, X_train_ans_label,Y_train_ques,answer_indices, examples_to_take_train, glove):
    global max_document_len
    global max_answer_len
    global max_question_len

    max_document_len = len(max(X_train_comp,key=len))
    max_answer_len = len(max(X_train_ans,key=len))
    max_question_len = len(max(Y_train_ques,key=len)) + 1

    # initialise all
    document_tokens = np.zeros((examples_to_take_train, max_document_len), dtype=np.int32)
    document_lengths = np.zeros(examples_to_take_train, dtype=np.int32)
    answer_labels = np.zeros((examples_to_take_train, max_document_len), dtype=np.int32)
    answer_masks = np.zeros((examples_to_take_train, max_answer_len, max_document_len), dtype=np.int32)
    answer_lengths = np.zeros(examples_to_take_train, dtype=np.int32)
    question_input_tokens = np.zeros((examples_to_take_train, max_question_len), dtype=np.int32)
    question_output_tokens = np.zeros((examples_to_take_train, max_question_len), dtype=np.int32)
    question_lengths = np.zeros(examples_to_take_train, dtype=np.int32)
    suppression_answer = np.zeros((examples_to_take_train, glove.shape[0], 1),dtype=np.int32)
    expression_contexts = np.zeros((examples_to_take_train, max_question_len,glove.shape[0]),dtype=np.int32)
    expression_probabilities = np.zeros((examples_to_take_train, max_question_len,glove.shape[0]),dtype=np.float32)

    # Build Vocab
    vocabulary_comp,word_to_index_comp,index_to_word_comp = create_vocabulary(X_train_comp + Y_train_ques)

    words_to_take = len(glove)
    # Build others
    for i in range(examples_to_take_train):
        answer_labels[i,0:len(X_train_ans_label[i])] = X_train_ans_label[i]
        for j, word in enumerate(X_train_comp[i]):
            document_tokens[i, j] = look_up_word(word)
        document_lengths[i] = len(X_train_comp[i])

        for j, index in enumerate(answer_indices[i]):
            answer_masks[i, j, index] = 1
        answer_lengths[i] = len(answer_indices[i])

        #print(Y_train_ques[i])
        question_input_words = ([START_WORD] + Y_train_ques[i])
        question_output_words = (Y_train_ques[i] + [END_WORD])

        for j, word in enumerate(question_input_words):
                question_input_tokens[i, j] = look_up_word(word)
        for j, word in enumerate(question_output_words):
            question_output_tokens[i, j] = look_up_word(word)
        question_lengths[i] = len(question_input_words)

        for j, word in enumerate(X_train_ans[i]):
            if(word not in Y_train_ques[i]):
                suppression_answer[i, look_up_word(word),:] = 1

        words_to_consider_expression = set(X_train_comp[i] + nltkStopWords + punctuations)

        for j,word in enumerate(words_to_consider_expression):
            expression_contexts[i,:,look_up_word(word)] = 1

        for j,word in enumerate(words_to_consider_expression):
            expression_probabilities[i,:,look_up_word(word)] = len(np.where(expression_contexts[i][0] == 1)[0]) / float(words_to_take)
        expression_probabilities[i,:,np.where(expression_probabilities[i][0] == 0)[0]] = len(np.where(expression_contexts[i][0] == 0)[0]) / float(words_to_take)


    document_tokens = sentences_to_indices_glove(X_train_comp, max_document_len)

    return [document_tokens,document_lengths,answer_labels,answer_masks,answer_lengths,question_input_tokens,question_output_tokens,question_lengths,suppression_answer,expression_contexts,expression_probabilities]


def findKMostFrequentWords(k):
    ctr = Counter([item for sublist in X_train_comp_all for item in sublist] + [item for sublist in Y_train_ques_all for item in sublist])
    sorted_ctr = sorted(ctr.items(), key=operator.itemgetter(1), reverse=True)
    return [item[0] for item in sorted_ctr[0:k]]





def create_one_hot_vector(data,vocabulary,word_to_index,index_to_word, maxLen):
    one_hot = np.zeros([maxLen,len(vocabulary)])
    for i,word in enumerate(data):
        if i >= maxLen:
            break
        if(word not in word_to_index):
            word = "<UNK>"
        one_hot[i][word_to_index[word]] = 1
    return one_hot

def create_one_hot_vector_from_indices(data,maxLen,vocabulary):
    one_hot = np.zeros([maxLen,len(vocabulary)])
    for i,indice in enumerate(data):
        if i >= maxLen:
            break
        one_hot[i][int(indice)] = 1
    return one_hot


def create_one_hot_training_Set(data,maxLen,vocabulary):
    one_hot_data = np.zeros([data.shape[0],maxLen,len(vocabulary)])
    for i in range(data.shape[0]):
        one_hot_data[i] = create_one_hot_vector_from_indices(data[i],maxLen,vocabulary)
    return one_hot_data


def sentences_to_indices_glove(X,max_len):

    m = len(X)

    X_indices = np.full([m,max_len],look_up_word(END_WORD))

    for i in range(m):
        j = 0
        for w in X[i]:
            if(j>=max_len):
                break;

            X_indices[i, j] = look_up_word(w)
            j = j+1
    return X_indices

def create_batch(inputs,batch_size,shuffle=False):
    num_batches = len(inputs[0]) // batch_size + 1

    outputs = []

    for index,inp in enumerate(inputs):

        output = {'document_tokens':[],
                    'document_lengths':[],
                    'answer_labels':[],
                    'answer_masks': [],
                    'answer_lengths': [],
                    'question_input_tokens':[],
                    'question_output_tokens':[],
                    'question_lengths':[],
                    'suppression_answer':[],
                    'expression_contexts': [],
                    'expression_probabilities':[]}

        start = 0
        for i in range(num_batches):
            if i == num_batches - 1:
                end = None
            else:
                end = start+batch_size
            maxD = max(inputs[1][start:end])
            maxA = max(inputs[4][start:end])
            maxQ = max(inputs[7][start:end])

            if index == 0:
                output['document_tokens'] = inp[start:end,:maxD]
            elif index==1:
                output['document_lengths'] = inp[start:end]
            elif index == 2:
                output['answer_labels']=inp[start:end,:maxD]
            elif index==3:
                output['answer_masks']=inp[start:end,:maxA,:maxD]
            elif index==4:
                output['answer_lengths']=inp[start:end]
            elif index==5:
                output['question_input_tokens']=inp[start:end,:maxQ]
            elif index==6:
                output['question_output_tokens']=inp[start:end,:maxQ]
            elif index==7:
                output['question_lengths'] = inp[start:end]
            elif index==8:
                output['suppression_answer'] = inp[start:end]
            elif index==9:
                output['expression_contexts'] = inp[start:end,0:maxQ,:]
            elif index==10:
                output['expression_probabilities'] = inp[start:end,0:maxQ,:]
            start = start + batch_size

            # output['doc_mask'] = np.full((batch_size, max_document_len))
            # for j, m in enumerate(output['doc_mask']):
            #     output['doc_mask'][i, output['document_lengths']:] =

        outputs.append(output)

    return outputs, num_batches

# def create_complete_dict(inputs,shuffle=False):
#     data = {}
#     for index,inp in enumerate(inputs):
#         if index == 0:
#             output['document_tokens'] = inp
#         elif index==1:
#             output['document_lengths'] = inp
#         elif index == 2:
#             output['answer_labels']=inp
#         elif index==3:
#             output['answer_masks']=inp
#         elif index==4:
#             output['answer_lengths']=inp
#         elif index==5:
#             output['question_input_tokens']=inp
#         elif index==6:
#             output['question_output_tokens']=inp
#         elif index==7:
#             output['question_lengths'] = inp
#         elif index==8:
#             output['suppression_answer'] = inp
#         elif index==9:
#             output['expression_contexts'] = inp
#         elif index==10:
#             output['expression_probabilities'] = inp


def save_obj(obj, path ):
    with open(path , 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(path):
    with open(path, 'rb') as f:
        return pickle.load(f)

def get_reduced_glove(reduce_glove):
    if reduce_glove:
    # if True:
        global look_up_word
        global look_up_token
        global _add_word

        dimensions = glove.shape[1]
        reduced_glove = []
        reduced_glove.append(np.zeros(dimensions))
        reduced_glove.append(-np.ones(dimensions))
        reduced_glove.append(np.ones(dimensions))

        words = findKMostFrequentWords(100000)

        print("WORDS:", len(words))
        for word in words:
            l = look_up_word(word)
            if(l != UNKNOWN_TOKEN):
                print("idx", word, l)
                idx = _add_word_reduced(word)
                reduced_glove.append(glove[l])
                if(len(reduced_glove) == wordToTake):
                    break

        reduced_glove = np.array(reduced_glove)

        look_up_word = look_up_word_reduced
        look_up_token = look_up_token_reduced
        _add_word = _add_word_reduced

        return reduced_glove
    else:
        return glove


def _add_word_reduced(word):
    idx = len(_idx_to_word_reduced)
    _word_to_idx_reduced[word] = idx
    _idx_to_word_reduced.append(word)
    return idx


def look_up_word_reduced(word):
    return _word_to_idx_reduced.get(word, UNKNOWN_TOKEN)


def look_up_token_reduced(token):
    return _idx_to_word_reduced[token]


wordToTake = 10000

_word_to_idx_reduced = {}
_idx_to_word_reduced = []


UNKNOWN_TOKEN = _add_word_reduced(UNKNOWN_WORD)
START_TOKEN = _add_word_reduced(START_WORD)
END_TOKEN = _add_word_reduced(END_WORD)
'''
