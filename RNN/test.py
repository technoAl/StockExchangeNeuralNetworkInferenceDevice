from gensim.models.keyedvectors import KeyedVectors
import numpy as np

glove = KeyedVectors.load_word2vec_format("glove.6B.50d.txt.w2v", binary=False)

MAXLEN = 100

def to_glove(sentence):
    out = []
    for word in sentence.split():
        word = word.lower()
        try:
            out.append(glove[word])
        except:
            continue
    if len(out) > MAXLEN:
        out = out[:50]
    elif len(out) < MAXLEN:
        for _ in range(len(out), MAXLEN):
            out.append(np.zeros(50))
    return out


print(to_glove("Hello My World"))

# Passed the check
