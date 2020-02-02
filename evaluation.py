import sys, os
import numpy as np 
import fasttext
import gleu

from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

def read_data(file_name):
    data = [] 
    with open(file_name, 'r') as fin:
        for line in fin:
            line = line.strip()
            data.append(line)
    return data

def read_ref_data(file_name, out_ref):
    data = [] 
    with open(file_name, 'r') as fin:
        for line in fin:
            line = line.strip().split('\t')[1]
            data.append(line)

    with open(out_ref, 'w') as fout:
        for line in data:
            fout.write(line+'\n')

if __name__ == '__main__':
    data_path = sys.argv[1]
    output_path = sys.argv[2]
    data_type = sys.argv[3]
    pos_name = sys.argv[4]
    neg_name = sys.argv[5]

    neg_src = data_path + '/sentiment.test.0'
    neg_ref = data_path + '/reference.0'
    neg_hum = data_path + '/human.0'
    pos_src = data_path + '/sentiment.test.1'
    pos_ref = data_path + '/reference.1'
    pos_hum = data_path + '/human.1'
    
    read_ref_data(neg_ref, neg_hum)
    read_ref_data(pos_ref, pos_hum)
    neg_file = output_path + f'/{neg_name}'
    pos_file = output_path + f'/{pos_name}'
    # neg_file = neg_src
    # pos_file = pos_src
    print ('file name: ', neg_name, pos_name)
    # ACC
    neg_data = read_data(neg_file)
    pos_data = read_data(pos_file)
    model = fasttext.load_model(data_type+'/model.bin')

    acc = 0
    for sentence in neg_data:
        result = model.predict(sentence)
        if result[0][0] == "__label__1":
            acc += 1
    for sentence in pos_data:
        result = model.predict(sentence)
        if result[0][0] == "__label__0":
            acc += 1
    print (f'ACC: {acc/1000}')

    # BLEU 

    source = [[sen.strip().split()] for sen in open(neg_src, 'r')] + [[sen.strip().split()] for sen in open(pos_src, 'r')]
    output = [sen.strip().split() for sen in open(neg_file, 'r')] + [sen.strip().split() for sen in open(pos_file, 'r')]
    cc = SmoothingFunction()
    score = 0
    for s, o in zip(source, output):
        score += sentence_bleu(s, o, smoothing_function=cc.method4)
    print (f'bleu (src): {score/1000}')

    source = [[sen.strip().split()] for sen in open(neg_hum, 'r')] + [[sen.strip().split()] for sen in open(pos_hum, 'r')]
    output = [sen.strip().split() for sen in open(neg_file, 'r')] + [sen.strip().split() for sen in open(pos_file, 'r')]
    cc = SmoothingFunction()
    score = 0
    for s, o in zip(source, output):
        score += sentence_bleu(s, o, smoothing_function=cc.method4)
    print (f'bleu (ref): {score/1000}')

    # GLEU

    iter = 400
    sent = False
    n = 4
    gleu_calculator = gleu.GLEU(n)
    gleu_calculator.load_sources(neg_src)
    gleu_calculator.load_references([neg_hum])
    neg_score = [g for g in gleu_calculator.run_iterations(num_iterations=iter, source=neg_src, hypothesis=neg_file, per_sent=sent)]
    gleu_calculator = gleu.GLEU(n)
    gleu_calculator.load_sources(pos_src)
    gleu_calculator.load_references([pos_hum])
    pos_score = [g for g in gleu_calculator.run_iterations(num_iterations=iter, source=pos_src, hypothesis=pos_file, per_sent=sent)]
    print (f'gleu: {(float(neg_score[0][0])+float(pos_score[0][0])) / 2}')

