import pickle
import random
def display_generated(q_gen, num_examples):
    num_batches, batch_size, max_q_len = q_gen['gt'].shape
    for i in range(num_batches):
        for j in random.sample(range(batch_size), num_examples):
            print("T.F. Generated:", ' '.join(q_gen['tf_gen'][i,j]).replace('<END>',''))
            print("Full Generated:", ' '.join(q_gen['full_gen'][i,j]).replace('<END>',''))
            print("Ground Truth..:", ' '.join(q_gen['gt'][i,j][1:]).replace('<END>',''))
            print("\n")

def save_obj(obj, path ):
    with open(path , 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(path):
    with open(path, 'rb') as f:
        return pickle.load(f)
