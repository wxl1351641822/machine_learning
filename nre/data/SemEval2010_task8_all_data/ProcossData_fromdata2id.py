import constant
from tqdm import tqdm
import random
def fromData2id(path):
    vocab=[]
    with open(constant.vocab_path,'r',encoding='utf-8') as f:
        vocab=eval(f.read())
    w2id={w:i for i,w in enumerate(vocab)}
    print(len(vocab))
    # print(w2id)
    with open(constant.label_path,'r',encoding='utf-8') as f:
        labellist=eval(f.read())
    label2id={w:i for i,w in enumerate(labellist)}


    with open(constant.pos_path,'r',encoding='utf-8') as f:
        poslist=eval(f.read())
    pos2id={w:i for i,w in enumerate(poslist)}
    with open(constant.dependency_path, 'r', encoding='utf-8') as f:
        dependencylist = eval(f.read())
    dependency2id = {w: i for i, w in enumerate(dependencylist)}
    data = []
    idlist = {'id':[],'relation_id': [], 'token_id': [], 'stanford_pos_id': [], 'stanford_dependency_id': [], 'position_1': [],
              'position_2': [],'relation_subj_start':[],'relation_subj_end':[],'relation_obj_start':[],'relation_obj_end':[]}
    with open(path,'r',encoding='utf-8') as f:
        data=eval(f.read())
        random.shuffle(data)
        # print(data)
        for d in tqdm(data):
            idlist['id'].append(d['id'])
            idlist['relation_id'].append(label2id[d['relation']])
            l=len(d['token'])
            e1s, e1e, e2s, e2e = d['relation_subj_start'], d['relation_subj_end'], d['relation_obj_start'], d[
                'relation_obj_end']
            # print(e1s)
            idlist['relation_subj_start'].append(e1s)
            idlist['relation_subj_end'].append(e1e)
            idlist['relation_obj_start'].append(e2s)
            idlist['relation_obj_end'].append(e2e)
            if(l<=constant.max_length):
                idlist['token_id'].append([w2id[t] if t in vocab else constant.UNK_ID for t in d['token']]+[constant.PAD_ID]*(constant.max_length-l))
                idlist['stanford_pos_id'].append([pos2id[t] for t in d['stanford_pos']]+[constant.PAD_ID]*(constant.max_length-l))
                idlist['stanford_dependency_id'].append([dependency2id[t] for t in d['stanford_dependency']]+[constant.PAD_ID]*(constant.max_length-l))
                # e11,e12,e21,e22=d['relation_subj_start'], d['relation_subj_end'], d['relation_obj_start'], d['relation_obj_end']

                idlist['position_1'].append([i - e1s+constant.max_length if i - e1s < 0 else i - e1e+constant.max_length if i - e1e > 0 else constant.max_length for i in range(len(d['token']))]+[constant.PAD_ID]*(constant.max_length-l))
                idlist['position_2'].append([i - e2s+constant.max_length if i - e2s < 0 else i - e2e+constant.max_length if i - e2e > 0 else constant.max_length for i in range(len(d['token']))]+[constant.PAD_ID]*(constant.max_length-l))
            else:
                idlist['token_id'].append([w2id[t] if t in vocab else constant.UNK_ID for t in d['token'][:constant.max_length]] )
                idlist['stanford_pos_id'].append([pos2id[t] for t in d['stanford_pos'][:constant.max_length]] )
                idlist['stanford_dependency_id'].append(
                    [dependency2id[t] for t in d['stanford_dependency'][:constant.max_length]] )
                # e11,e12,e21,e22=d['relation_subj_start'], d['relation_subj_end'], d['relation_obj_start'], d['relation_obj_end']

                idlist['position_1'].append([i - e1s+constant.max_length if i - e1s < 0 else i - e1e+constant.max_length if i - e1e > 0 else constant.max_length for i in range(constant.max_length)])
                idlist['position_2'].append([i - e2s+constant.max_length if i - e2s < 0 else i - e2e+constant.max_length if i - e2e > 0 else constant.max_length for i in range(constant.max_length)])
                # print(e1s,e1e,e2s,e2e)
                # print(idlist['position_2'][-1])
                # print(idlist['position_1'][-1])
            for t in d['token']:
                if t in vocab:
                    if(w2id[t]>19197):
                        print(t)


    return idlist

def fromData2id_slice(path,times,data):
    idlist={}
    length=len(data['token_id'])//times
    for i in range(times):
        # print(length)
        # print(data['token_id'])
        # print(i*length,(i+1)*length)
        idlist['id']=data['id'][i*length:(i+1)*length]
        idlist['token_id']=data['token_id'][i*length:(i+1)*length]
        idlist['relation_id'] = data['relation_id'][i * length:(i + 1) * length]
        idlist['relation_subj_start']=data['relation_subj_start'][i * length:(i + 1) * length]
        idlist['relation_subj_end']=data['relation_subj_end'][i * length:(i + 1) * length]
        idlist['relation_obj_start']=data['relation_obj_start'][i * length:(i + 1) * length]
        idlist['relation_obj_end']=data['relation_obj_end'][i * length:(i + 1) * length]
        idlist['stanford_pos_id'] = data['stanford_pos_id'][i * length:(i + 1) * length]
        idlist['stanford_dependency_id'] = data['stanford_dependency_id'][i * length:(i + 1) * length]
        idlist['position_1'] = data['position_1'][i * length:(i + 1) * length]
        idlist['position_2'] = data['position_2'][i * length:(i + 1) * length]
        with open(path+str(i), 'w', encoding='utf-8') as f:
            f.write(str(idlist))
    return idlist

def get_distrib(path):
    distrib = {}

    for i in range(19):
        distrib[i] = 0
    for j in range(9):
        with open(path+str(j),'r',encoding='utf-8') as f:
            data=eval(f.read())
            y=data['relation_id']
            for i in y:
                # print(i)
                distrib[i] += 1
    print(distrib)
    for i in range(19):
        distrib[i] = 0
    with open(path+str(9),'r',encoding='utf-8') as f:
        data=eval(f.read())
        print(data)
        y=data['relation_id']
        for i in y:
            distrib[i] += 1
    print(distrib)
    return distrib

print('建立idlist...')
train_idlist=fromData2id(constant.train_path)
print(train_idlist)
with open(constant.att_lstm_traindata_path, 'w', encoding='utf-8') as f:
    f.write(str(train_idlist))
test_idlist=fromData2id(constant.test_path)
with open(constant.att_lstm_testdata_path, 'w', encoding='utf-8') as f:
    f.write(str(test_idlist))
# test_idlist=fromData2id(constant.test_path)
l=len(train_idlist)//10
k=len(test_idlist)//10

# print(train_idlist)

test_idlist=fromData2id_slice(constant.att_lstm_traindata_path,10,train_idlist)
train_idlist=fromData2id_slice(constant.att_lstm_testdata_path,10,test_idlist)
get_distrib(constant.att_lstm_traindata_path)