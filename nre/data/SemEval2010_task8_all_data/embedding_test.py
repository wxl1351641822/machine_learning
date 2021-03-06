from tqdm import tqdm
from collections import Counter
import random
import numpy as np
import constant
random.seed(1234)
np.random.seed(1234)
def load_token(path):
    tokens=[]
    with open(path,'r',encoding='utf-8') as f:
        data=eval(f.read())
        for d in tqdm(data):
            print(d)
            t = d['token']
            # if(d['relation']!='other'):
            #     e1s,e1e,e2s,e2e=d['relation_subj_start'],d['relation_subj_end'],d['relation_obj_start'],d['relation_obj_end']
            #     t[e1s:e1e+1]=[constant.PAD]*(e1e-e1s+1)
            #     t[e2s:e2e + 1] = [constant.PAD] * (e2e - e2s + 1)
            # print(t)
            tokens+=t
    return tokens

def load_glove(glove_path,wv_dim):
    glove_vocab=set()
    with open(glove_path,'r',encoding='utf-8') as f:
        for line in tqdm(f.readlines()):
            elem=line.split()
            glove_vocab.add((''.join(elem[:-wv_dim])).lower())
            # print(elem)
    return glove_vocab



def build_vocab(train_token,glove_vocab):
    counter=Counter(train_token)
    v=sorted([t for t in counter],key=counter.get,reverse=True)
    print('vocab built with %d/%d words.'%(len(v),len(counter)))
    v=constant.VOCAB_PREFIX+v
    return v

def count_oov(data,v):
    c=Counter(data)
    # print(c.values())
    total=sum(c.values())
    matched=sum(c[t] for t in v)
    return total,total-matched

def build_embedding(v,glove_path,glove_dim):
    vocab_size=len(v)
    emb=np.random.randn(vocab_size, glove_dim).astype(np.float32) / np.sqrt(vocab_size)
    w2id={w:id for id,w in enumerate(v)}
    emb[constant.PAD_ID]=0
    # print(emb)
    with open(glove_path,'r',encoding='utf-8') as f:
        for line in tqdm(f):
            elem=line.split()
            word=''.join(elem[:-glove_dim])
            word=word.lower()
            if word in v:
                emb[w2id[word]]=elem[-glove_dim:]
        # print(emb)
    return emb

def get_labellist(path,out_path):
    labellist=set()
    with open(path,'r',encoding='utf-8') as f:
        data=eval(f.read())
        for d in tqdm(data):
            labellist.add(d['relation'])
    labellist=sorted(list(labellist))
    with open(out_path,'w',encoding='utf-8') as f:
        f.write(str(list(labellist)))
    # print(labellist)
    return labellist

def get_poslist(path,out_path):
    poslist = set()
    with open(path, 'r', encoding='utf-8') as f:
        data = eval(f.read())
        for d in tqdm(data):
            for k in d['stanford_pos']:
                poslist.add(k)
    poslist = constant.VOCAB_PREFIX+sorted(list(poslist))
    with open(out_path, 'w', encoding='utf-8') as f:
        f.write(str(poslist))
    # print(labellist)
    return poslist

def get_dependencylist(path,out_path):
    dependencylist = set()
    with open(path, 'r', encoding='utf-8') as f:
        data = eval(f.read())
        for d in tqdm(data):
            for k in d['stanford_dependency']:
                dependencylist.add(k)
    with open(out_path, 'w', encoding='utf-8') as f:
        f.write(str(constant.VOCAB_PREFIX+list(dependencylist)))
    # print(labellist)
    return dependencylist

print('加载train_token...')
train_token=load_token(constant.train_path)
print('加载test_token...')
test_token=load_token(constant.test_path)

print("glove->glove_vocab...")
glove_vocab=load_glove(constant.glove_path,constant.glove_dim)
print('glove_vocab的长度：',len(glove_vocab))

print('建立vocab（train中在glove的部分）...')
v=build_vocab(train_token,glove_vocab)

print('未录入词的占比：')
dataset={'train':train_token,'test':test_token}
for dname,data in dataset.items():
    total,oov=count_oov(data,v)
    print('%s 的未录入词占比为%d/%d'%(dname,oov,total))

print('建立embedding...')
embedding=build_embedding(v,constant.glove_path,constant.glove_dim)

print('保存vocab和embedding')
with open(constant.vocab_path,'w',encoding='utf-8') as f:
    f.write(str(v))
np.save(constant.emb_path,embedding)

print('建立labellist...')
get_labellist(constant.train_path, constant.label_path)
print('建立poslist')
get_poslist(constant.train_path, constant.pos_path)
print('建立dependencylist')
get_dependencylist(constant.train_path, constant.dependency_path)