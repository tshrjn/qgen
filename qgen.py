
# coding: utf-8


from embedding import *


# In[12]:


import json
from pprint import pprint
import re
import numpy as np

import torch
import nltk
nltk.download('punkt')
nltk.download('stopwords')


# In[13]:


data = json.load(open('train-v1.1.json'))


# In[14]:


from nltk.corpus import stopwords
nltk.download('stopwords')
nltk.download('punkt')
nltkStopWords = stopwords.words('english')
punctuations = [',', '?', '.', '-',]


# In[15]:


def extractor(data):
    contexts = []
    qas = []
    for i in range(len(data["data"])):
        for j in range(len(data["data"][i]["paragraphs"])):
            contexts.append(data["data"][i]["paragraphs"][j]["context"])
            qas.append(data["data"][i]["paragraphs"][j]["qas"])
    return (contexts,qas)


# In[16]:


CapPassage = False

from nltk.tokenize import word_tokenize
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


# In[17]:


X_train_comp_all = []
X_train_comp_ans_all = []
X_train_ans_all = []
Y_train_ques_all = []
invalid = 0
X_train_ans_label_all = []
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
            invalid = invalid+1
    


# In[18]:


from collections import Counter
import operator
def findKMostFrequentWords(k):
    ctr = Counter([item for sublist in X_train_comp_all for item in sublist] + [item for sublist in Y_train_ques_all for item in sublist])
    sorted_ctr = sorted(ctr.items(), key=operator.itemgetter(1), reverse=True)
    return [item[0] for item in sorted_ctr[0:k]]


# In[19]:


wordToTake = 2000
words = findKMostFrequentWords(100000)


# In[20]:


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

for word in words:
    l = look_up_word(word)
    if(l != UNKNOWN_TOKEN):
        idx = _add_word_reduced(word)
        reduced_glove.append(glove[l])
        if(len(reduced_glove) == wordToTake):
            break
        
def look_up_word_reduced(word):
    return _word_to_idx_reduced.get(word, UNKNOWN_TOKEN)


def look_up_token_reduced(token):
    return _idx_to_word_reduced[token]

reduced_glove = np.array(reduced_glove)


# In[21]:

print(invalid)
for i in np.where(X_train_ans_label_all[110] == 1)[0]:
    print(X_train_comp_all[110][i])


# In[22]:


print(X_train_comp_all[0])
print(X_train_ans_all[0])


# In[23]:


find_sub_list(X_train_ans_all[0] , X_train_comp_all[0])


# In[24]:


print(invalid)
print(X_train_comp_all[101])
print(X_train_ans_all[101])
print(Y_train_ques_all[101])

c = list(zip(X_train_comp_all,X_train_comp_ans_all, X_train_ans_all, X_train_ans_label_all,Y_train_ques_all))
np.random.shuffle(c)
X_train_comp_all_shuffled,X_train_comp_ans_all_shuffled, X_train_ans_shuffled, X_train_ans_label_shuffled,Y_train_ques_all_shuffled = zip(*c)

print(X_train_comp_all_shuffled[101])
print(X_train_comp_ans_all_shuffled[101])
print(X_train_ans_shuffled[101])
print(X_train_ans_label_shuffled[101])
print(Y_train_ques_all_shuffled[101])


# In[25]:


#examples_to_take_train = len(X_train_comp_all_shuffled)
examples_to_take_train = 40000

X_train_comp = X_train_comp_all_shuffled[0:examples_to_take_train]
X_train_comp_ans = X_train_comp_ans_all_shuffled[0:examples_to_take_train]
X_train_ans = X_train_ans_shuffled[0:examples_to_take_train]
X_train_ans_label = X_train_ans_label_shuffled[0:examples_to_take_train]
Y_train_ques = Y_train_ques_all_shuffled[0:examples_to_take_train]
answer_indices = [np.where(x==1)[0].tolist() for x in X_train_comp_ans]


# In[26]:


max_document_len = len(max(X_train_comp,key=len))
max_answer_len = len(max(X_train_ans,key=len))
max_question_len = len(max(Y_train_ques,key=len)) + 1


# In[27]:


document_tokens = np.zeros((examples_to_take_train, max_document_len), dtype=np.int32)
document_lengths = np.zeros(examples_to_take_train, dtype=np.int32)
answer_labels = np.zeros((examples_to_take_train, max_document_len), dtype=np.int32)
answer_masks = np.zeros((examples_to_take_train, max_answer_len, max_document_len), dtype=np.int32)
answer_lengths = np.zeros(examples_to_take_train, dtype=np.int32)
question_input_tokens = np.zeros((examples_to_take_train, max_question_len), dtype=np.int32)
question_output_tokens = np.zeros((examples_to_take_train, max_question_len), dtype=np.int32)
question_lengths = np.zeros(examples_to_take_train, dtype=np.int32)
suppression_answer = np.zeros((examples_to_take_train, reduced_glove.shape[0], 1),dtype=np.int32)
expression_contexts = np.zeros((examples_to_take_train, max_question_len,reduced_glove.shape[0]),dtype=np.int32)
expression_probabilities = np.zeros((examples_to_take_train, max_question_len,reduced_glove.shape[0]),dtype=np.float32)


# In[28]:


answer_labels[0]


# In[29]:


print(answer_labels.shape)
for i in range(examples_to_take_train):
    answer_labels[i,0:len(X_train_ans_label[i])] = X_train_ans_label[i]
    for j, word in enumerate(X_train_comp[i]):
        document_tokens[i, j] = look_up_word_reduced(word)
    document_lengths[i] = len(X_train_comp[i])

    for j, index in enumerate(answer_indices[i]):
        answer_masks[i, j, index] = 1
    answer_lengths[i] = len(answer_indices[i])
    
    #print(Y_train_ques[i])
    question_input_words = ([START_WORD] + Y_train_ques[i])
    question_output_words = (Y_train_ques[i] + [END_WORD])

    for j, word in enumerate(question_input_words):
            question_input_tokens[i, j] = look_up_word_reduced(word)
    for j, word in enumerate(question_output_words):
        question_output_tokens[i, j] = look_up_word_reduced(word)
    question_lengths[i] = len(question_input_words)
    
    for j, word in enumerate(X_train_ans[i]):
        if(word not in Y_train_ques[i]):
            suppression_answer[i, look_up_word_reduced(word),:] = 1
            
    words_to_consider_expression = set(X_train_comp[i] + nltkStopWords + punctuations)

    for j,word in enumerate(words_to_consider_expression):
        expression_contexts[i,:,look_up_word_reduced(word)] = 1
        
    for j,word in enumerate(words_to_consider_expression):
        expression_probabilities[i,:,look_up_word_reduced(word)] = len(np.where(expression_contexts[i][0] == 1)[0]) / float(wordToTake)
    expression_probabilities[i,:,np.where(expression_probabilities[i][0] == 0)[0]] = len(np.where(expression_contexts[i][0] == 0)[0]) / float(wordToTake)
    
        


# In[30]:


# In[31]:


print(len(np.where(expression_contexts[10][0] == 0)[0]))
print(len(np.where(expression_contexts[10][0] == 1)[0]))

print(len(np.where(expression_probabilities[10][0] > 0.5)[0]))
print(len(np.where(expression_probabilities[10][0] < 0.5)[0]))


# In[32]:


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


# In[33]:


print(reduced_glove.shape)
vocabulary_comp,word_to_index_comp,index_to_word_comp = create_vocabulary(X_train_comp + Y_train_ques)
print(len(vocabulary_comp))
print(word_to_index_comp["?"])
print(word_to_index_comp["what"])


# In[34]:


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




# In[35]:


def sentences_to_indices_glove(X,max_len):
    
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


# In[36]:


document_tokens = sentences_to_indices_glove(X_train_comp, max_document_len)


# In[37]:


document_tokens[0]


# In[38]:


answer_labels.shape


# ## Batch Data Preparation

# In[39]:


#document_tokens = np.zeros((examples_to_take_train, max_document_len), dtype=np.int32)
#document_lengths = np.zeros(examples_to_take_train, dtype=np.int32)
#answer_labels = np.zeros((examples_to_take_train, max_document_len), dtype=np.int32)
#answer_masks = np.zeros((examples_to_take_train, max_answer_len, max_document_len), dtype=np.int32)
#answer_lengths = np.zeros(examples_to_take_train, dtype=np.int32)
#question_input_tokens = np.zeros((examples_to_take_train, max_question_len), dtype=np.int32)
#question_output_tokens = np.zeros((examples_to_take_train, max_question_len), dtype=np.int32)
#question_lengths = np.zeros(examples_to_take_train, dtype=np.int32)
#suppression_answer = np.zeros((examples_to_take_train,max_answer_len),dtype=np.int32)
#expression_contexts = np.zeros((examples_to_take_train, max_question_len,reduced_glove.shape[0]),dtype=np.int32)
#expression_probabilities = np.zeros((examples_to_take_train, max_question_len,reduced_glove.shape[0]),dtype=np.float32)


# In[40]:


import math
def createBatch(inputs,batch_size,shuffle=False):
    outputs = []
    num_batches = math.ceil(len(inputs[0])/batch_size)
    
    for index,inp in enumerate(inputs):
        start = 0
        output = []
        for i in range(num_batches-1):
            maxD = max(inputs[1][start:start+batch_size])
            maxA = max(inputs[4][start:start+batch_size])
            maxQ = max(inputs[7][start:start+batch_size])
            if index == 0 or index == 2:
                output.append(inp[start:start+batch_size,0:maxD]) 
            elif index==3:
                output.append(inp[start:start+batch_size,0:maxA,0:maxD]) 
            elif index==5 or index==6:
                output.append(inp[start:start+batch_size,0:maxQ])
            elif index==9 or index==10:
                output.append(inp[start:start+batch_size,0:maxQ,:])
            else: 
                output.append(inp[start:start+batch_size])
            start = start + batch_size
        
        # Remaining training sample i.e. training mod batch_size
        maxD = max(inputs[1][start:])
        maxA = max(inputs[4][start:])
        maxQ = max(inputs[7][start:])
        if index == 0 or index == 2:
            output.append(inp[start:,0:maxD]) 
        elif index==3:
            output.append(inp[start:,0:maxA,0:maxD]) 
        elif index==5 or index==6:
            output.append(inp[start:,0:maxQ])
        elif index==9 or index==10:
            output.append(inp[start:,0:maxQ,:]) 
        else: 
            output.append(inp[start:])
        outputs.append(output)
    
    return outputs


# In[41]:


batch_size = 64
batch_input = createBatch([document_tokens,document_lengths,answer_labels,answer_masks,answer_lengths,question_input_tokens,question_output_tokens,question_lengths,suppression_answer,expression_contexts,expression_probabilities]
                    ,batch_size)
number_of_batches = len(batch_input[0])


# ## Model

# In[197]:


import torch
import torch.nn as nn
from torch.autograd import Variable
from torch import optim
import torch.nn.functional as F

use_cuda = torch.cuda.is_available()


# In[198]:


use_cuda


# ### Document Embedding

# In[199]:


class Embedder(nn.Module):
    def __init__(self, input_size, output_size):
        super(Embedder, self).__init__()
        self.embedding = nn.Embedding(input_size, output_size)
        
        self.embedding.weight = nn.Parameter(torch.from_numpy(reduced_glove).float())
        self.embedding.weight.requires_grad = False
    def forward(self, x):
        return self.embedding(x)


# In[200]:


class AnswerEncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1):
        super(AnswerEncoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.gru = nn.GRU(input_size, hidden_size, batch_first= True, bidirectional=True).cuda() #Input_size = Hidden_Size
        self.fc = nn.Linear(hidden_size*2, 1).cuda()

    def forward(self, x, hidden):
        output, hidden = self.gru(x, hidden)
        final_output = self.fc(output)
        final_output = F.sigmoid(final_output)
        self.hiddenState = hidden
        return final_output, output, hidden
    
    def initHidden(self):
        result = Variable(torch.zeros(2, batch_size, self.hidden_size)) #2 for BiDirectional
        if use_cuda:
            result = result.cuda()
        return result


# In[201]:


class QuestionEncoderRNN(nn.Module):
    
    def __init__(self,input_size, hidden_size):
        super(QuestionEncoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.gru = nn.GRU(input_size, hidden_size, batch_first= True).cuda()
    
    def forward(self, x, hidden):
        output, hidden = self.gru(x, hidden)
        self.hiddenState = hidden
        return output, hidden
    
    def initHidden(self):
        result = Variable(torch.zeros(1, 1, self.hidden_size))
        if use_cuda:
            result = result.cuda()
        return result


# In[202]:


class QuestionDecoderRNN(nn.Module):
    
    def __init__(self,input_size, hidden_size):
        super(QuestionDecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.gru = nn.GRU(input_size, hidden_size, batch_first= True).cuda()
        
    def forward(self, x, hidden):
        output, hidden = self.gru(x, hidden)
        self.hiddenState = hidden
        return output, hidden
    
    def initHidden(self):
        result = Variable(torch(1, 1, self.hidden_size))
        if use_cuda:
            result = result.cuda()
        return result

class FCLayer(nn.Module):
    def __init__(self,input_size, output_size):
        super(FCLayer, self).__init__()
        self.fc = nn.Linear(input_size, output_size).cuda()
        
    def forward(self, x):
        return self.fc(x)
    
    
class QuestionGenerationFC(nn.Module):
    def __init__(self,input_size, output_size):
        super(QuestionGenerationFC, self).__init__()
        self.hidden_size = hidden_size
        self.fc = nn.Linear(hidden_size, output_size).cuda()

    def forward(self, x):
        output = self.fc(x)
        output = F.log_softmax(output.view(-1,1,1)).view(1,1,-1)
        return output
         


# In[203]:


hidden_size = reduced_glove.shape[1]

embedder = Embedder(input_size = reduced_glove.shape[0], output_size = reduced_glove.shape[1])
fcLayer = FCLayer(hidden_size*2, hidden_size)
answerEncoder = AnswerEncoderRNN(input_size = hidden_size, hidden_size=int(hidden_size/2))
questionEncoder = QuestionEncoderRNN(input_size=hidden_size, hidden_size=hidden_size)
questionDecoder = QuestionDecoderRNN(input_size=hidden_size, hidden_size=hidden_size)
questionGenerator = QuestionGenerationFC(input_size = hidden_size, output_size=reduced_glove.shape[0])

train_param = []

for model in [embedder, answerEncoder, questionEncoder, questionDecoder, questionGenerator]:
    train_param += [p for p in model.parameters() if p.requires_grad]

print("Number of trainable parameters = ", len(train_param))

optimizer = torch.optim.Adam(train_param, 0.0001)
criterion1 = nn.BCELoss()
#criterion2 = nn.CrossEntropyLoss()
criterion2 = nn.NLLLoss()

def save():
    torch.save({'answerEncoder': answerEncoder.state_dict(),
                'questionEncoder': questionEncoder.state_dict(),
                'questionDecoder': questionDecoder.state_dict(),
                'questionGenerator': questionGenerator.state_dict(),
                'optimizer': optimizer.state_dict(),
                }, 'qgen_base.pt')

# In[204]:

def repackage_hidden(h):
    """Wraps hidden states in new Variables, to detach them from their history."""
    if type(h) == Variable:
        return Variable(h.data)
    else:
        return tuple(repackage_hidden(v) for v in h)

verboseBatchPrinting = True
averageBatchLossPrinting = True

num_epochs = 100
answer_encoder_hidden = answerEncoder.initHidden()
question_encoder_hidden = questionEncoder.initHidden()
question_decoder_hidden = None
for epoch in range(1, num_epochs+1):
    avg_loss = 0
    for batch_num in range(len(batch_input[0])):
        
        current_batch_size = len(batch_input[0][batch_num])
        if current_batch_size != batch_size:
            continue
        
        
        maxDocLenForBatch = int(max(batch_input[1][batch_num]))
        mask = np.zeros((current_batch_size, maxDocLenForBatch))
        for i in range(current_batch_size):
            mask[i][0:batch_input[1][0][i]] = 1
            
        inp = Variable(torch.from_numpy(batch_input[0][batch_num]).long())

        labels = Variable(torch.from_numpy(batch_input[2][batch_num])).long()
        if use_cuda:
            labels = labels.cuda()

        optimizer.zero_grad()
        embedded_inp = embedder(inp).cuda()
        answer_encoder_hidden = repackage_hidden(answer_encoder_hidden)
        answer_tags, answer_outputs, answer_encoder_hidden = answerEncoder(embedded_inp, answer_encoder_hidden)
        
        
        if use_cuda:
            answer_outputs = answer_outputs.cuda()
            answer_tags = answer_tags.cuda()
        
        
        t_document_mask = Variable(torch.from_numpy(mask)).float()
        if use_cuda:
            t_document_mask = t_document_mask.cuda()
        outputs = torch.mul(answer_tags.squeeze(-1),t_document_mask)
        
        
        answer_loss = criterion1(outputs, labels.float())
            
        
        t_answer_mask = Variable(torch.from_numpy(batch_input[3][batch_num])).float()
        if use_cuda:
            t_answer_mask = t_answer_mask.cuda()
        
        question_encoder_input = torch.matmul(t_answer_mask, answer_outputs.float())
        question_encoder_hidden_batch = Variable(torch.zeros(1,current_batch_size,questionEncoder.hidden_size))
        if use_cuda:
            question_encoder_hidden_batch = question_encoder_hidden_batch.cuda()
        
        question_encoder_hidden = repackage_hidden(question_encoder_hidden)
        for i in range(current_batch_size):
            _ , question_encoder_hidden = questionEncoder(question_encoder_input[i:i+1,0:batch_input[4][batch_num][i],:], question_encoder_hidden)
            question_encoder_hidden_batch[:,i:i+1,:] = question_encoder_hidden
            
        #question_encoder_hidden_batch = fcLayer(question_encoder_hidden_batch)
            
        if type(question_decoder_hidden) == Variable:
            question_decoder_hidden = repackage_hidden(question_decoder_hidden)
        question_loss = 0
        for i in range(current_batch_size):
            question_decoder_hidden = question_encoder_hidden_batch[:,i:i+1,:].clone()
            embedded_inputs = embedder(torch.from_numpy(batch_input[5][batch_num][i]).long()).cuda()
            output_labels = Variable(torch.from_numpy(batch_input[5][batch_num][i]).long())
            if use_cuda:
                output_labels = output_labels.cuda()
                
            for quesL in range(batch_input[7][batch_num][i]):
                decoder_output, question_decoder_hidden = questionDecoder(
                    embedded_inputs[quesL:quesL+1].unsqueeze(1),
                    question_decoder_hidden)
                
                final_output = questionGenerator(decoder_output)
                output_label = Variable(torch.zeros(1,2000))
                if use_cuda:
                    output_label = output_label.cuda()
                output_label[:,batch_input[5][batch_num][i][quesL]] = 1
                question_loss += criterion2(final_output.squeeze(0), 
                                           output_labels[quesL:quesL+1])
                ##question_loss += criterion2(final_output.squeeze(0), output_label)
        

        #net_loss = answer_loss + question_loss
        net_loss = question_loss
        net_loss.backward(retain_graph=True)
        optimizer.step()


        avg_loss+= net_loss.data[0]
        if verboseBatchPrinting:
            print ('Batch: %d \t Epoch : %d\tNet Loss: %.4f \tAnswer Loss: %.4f \tQuestion Loss: %.4f' 
                   %(batch_num, epoch, net_loss.data[0], answer_loss.data[0], question_loss.data[0]))

        
        
    if averageBatchLossPrinting:
        print('Average Loss after Epoch %d : %.4f'
                   %(epoch, avg_loss/number_of_batches))

    if epoch % 2==0:
        save()
