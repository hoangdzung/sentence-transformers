from sentence_transformers import SentenceTransformer, LoggingHandler
import numpy as np
import logging
import sys
import time
import pickle
from scipy.spatial.distance import cosine, pdist, squareform

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
dist = squareform(pdist(embeddings, 'cosine'))
dist = np.where(dist<0.01,2,dist)
col,row=np.where(dist<0.1)
def f(i):
    print(messages[col[i]])
    print(messages[row[i]])
import pdb;pdb.set_trace()

while(True):
    message = input("Enter:")
    if len(message) ==0: break
    idx = int(input('Id:'))
    message_embeddings = model.encode([message])
    message_embedding = message_embeddings[0]
    print(messages[idx])
    print(cosine(message_embedding, embeddings[idx]))
