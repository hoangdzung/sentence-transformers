from sentence_transformers import SentenceTransformer, LoggingHandler
import numpy as np
import logging
import sys
import time
import pickle
from scipy.spatial.distance import cdist
from tabulate import tabulate
#### Just some code to print debug information to stdout
np.set_printoptions(threshold=100)

logging.basicConfig(format='%(asctime)s - %(message)s',
                            datefmt='%Y-%m-%d %H:%M:%S',
                                                level=logging.INFO,
                                                                    handlers=[LoggingHandler()])

# Load Sentence model (based on BERT) from URL
model = SentenceTransformer('bert-large-nli-mean-tokens')

#sentences = [line.strip() for line in open(sys.argv[1])][:5000]
messages = pickle.load(open(sys.argv[1],'rb'))
embeddings = np.load(sys.argv[2])
while(True):
    message = input("Enter:")
    if len(message) ==0: break
    message_embeddings = model.encode([message])
    dist = cdist(message_embeddings, embeddings,"cosine")
    dist=np.squeeze(dist)
    sorted_idx = np.argsort(dist)[:10]
    print(tabulate([[messages[idx], dist[idx]] for idx in sorted_idx]))
