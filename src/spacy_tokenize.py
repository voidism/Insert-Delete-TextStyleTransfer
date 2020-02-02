import spacy
spacy_nlp = spacy.load('en_core_web_sm', disable=['parser', 'tagger', 'ner', 'entity_ruler'])

import sys
f = open(sys.argv[1], 'r')
fw = open(sys.argv[2], 'w')

for line in f:
    line = line.strip()
    doc = spacy_nlp(line)
    tokens = [token.text for token in doc]
    fw.write(' '.join(tokens)+'\n')
