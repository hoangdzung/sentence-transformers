"""
This basic example loads a pre-trained model from the web and uses it to
generate sentence embeddings for a given list of sentences.
"""

from sentence_transformers import SentenceTransformer, LoggingHandler
import numpy as np
import logging
import sys
import time
import pickle

#### Just some code to print debug information to stdout
np.set_printoptions(threshold=100)

logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO,
                    handlers=[LoggingHandler()])

# Load Sentence model (based on BERT) from URL
model = SentenceTransformer('bert-large-nli-mean-tokens')

if sys.argv[1].endswith('.txt'):
    sentences = [line.strip() for line in open(sys.argv[1])]
else:
    sentences = pickle.load(open(sys.argv[1], 'rb'))

stime = time.time()
sentence_embeddings = model.encode(sentences)
sentence_embeddings = np.array(sentence_embeddings)

np.save(sys.argv[2], sentence_embeddings)
# The result is a list of sentence embeddings as numpy arrays
#for sentence, embedding in zip(sentences, sentence_embeddings):
#    print("Sentence:", sentence)
#    print("Embedding:", embedding)
#    print("")
