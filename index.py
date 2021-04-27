import faiss

import numpy as np


def index_search(xb,xq):
    ngpus = faiss.get_num_gpus()


    print("number of GPUs:", ngpus)
    d = xb.shape()[-1]
    cpu_index = faiss.IndexFlatL2(d)

    gpu_index = faiss.index_cpu_to_all_gpus(  # build the index
        cpu_index
    )

    gpu_index.add(xb)              # add vectors to the index
    print(gpu_index.ntotal)

    k = 10                          # we want to see 4 nearest neighbors
    D, I = gpu_index.search(xq, k) # actual search
    
    return D, I
