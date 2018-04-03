def display_generated(q_gen):
    num_batches, batch_size, max_q_len = q_gen['gt'].shape
    for i in range(num_batches):
        for j in range(batch_size):
            print("Generated:", ' '.join(q_gen['gen'][i,j]))
            print("Gr. Truth:", ' '.join(q_gen['gt'][i,j]))
            print("\n")
        