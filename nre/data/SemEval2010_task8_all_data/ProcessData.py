from stanfordcorenlp import StanfordCoreNLP
from tqdm import tqdm

def read_semEval(path):
    datalist=[]
    nlp = StanfordCoreNLP(r'D:/ruanjian/stanford_coreNLP/stanford-corenlp-full-2018-10-05')
    with open(path,'r',encoding='utf-8') as f:
        datalist=f.readlines();
    result=[]
    row={}
    e1_start = -1
    e1_end = -1
    e2_start = -1
    e2_end = -1
    for i,data in tqdm(enumerate(datalist)):
        data=data.lower()
        if(i%4==0):
            e1_start = -1
            e1_end = -1
            e2_start = -1
            e2_end = -1
            id,sentence=data.split("\t")
            # print(id)
            # print(sentence[1:-2])
            token=sentence[1:-3].split()
            for j,t in enumerate(token):
                if('<e1>' in t or '<e1>' in t):
                    e1_start=j
                    token[j]=token[j].replace('<e1>','')
                if ('</e1>' in t):
                    e1_end = j
                    token[j] = token[j].replace('</e1>', '')
                if ('<e2>' in t or '<e2>' in t):
                    e2_start = j
                    token[j] = token[j].replace('<e2>', '')
                if ('</e2>' in t):
                    e2_end = j
                    token[j] = token[j].replace('</e2>', '')
            row={}
            sentence=sentence.replace('<e1>','')
            sentence = sentence.replace('</e1>', '')
            sentence = sentence.replace('<e2>', '')
            sentence = sentence.replace('</e2>', '')
            row['id']=int(id)
            row['sentence']=sentence[1:-2]
            row['token']=token
            pos,dependency,dependency_head=stanford_pos(nlp,sentence[1:-2])
            row['stanford_pos']=pos
            row['stanford_dependency']=dependency
            row['stanford_dependency_head'] = dependency_head
        elif(i%4==1):
            # print(data[:-8])
            if(data[:-1]=='Other'):
                row['relation']=data[:-1]
            else:
                row['relation']=data[:-8]
                head=data[-7:-5]
                tail=data[-4:-2]
                # print(head,tail)
                if(head=='e1'):#e1->e2:subj->obj
                    row['relation_subj_start']=e1_start
                    row['relation_subj_end'] = e1_end
                    row['relation_obj_start'] = e2_start
                    row['relation_obj_end'] = e2_end
                else:
                    row['relation_subj_start'] = e2_start
                    row['relation_subj_end'] = e2_end
                    row['relation_obj_start'] = e1_start
                    row['relation_obj_end'] = e1_end
        elif(i%4==2):
            row['Comment']=data[8:-1]
            result.append(row)
    # print(result)
    nlp.close()

    return result

def stanford_pos(nlp,sentence):
    pos=nlp.pos_tag(sentence)
    depend=nlp.dependency_parse(sentence)
    # print(pos)
    # print(depend)
    p=[]
    d=[0]*len(depend)
    d_head=[0]*len(depend)
    # print(len(pos), len(depend))
    # print(d)
    for i in range(len(pos)):
        p.append(pos[i][1])
        d[depend[i][2]-1]=depend[i][0]
        d_head[depend[i][2]-1]=depend[i][1]
    # print(d)
    return p,d,d_head




# train=read_semEval('./TRAIN_FILE.TXT')
# with open('./mytrain.json', 'w', encoding='utf-8') as f:
#     f.write(str(train))
test=read_semEval('./TEST_FILE_FULL.TXT')
with open('./mytest.json', 'w', encoding='utf-8') as f:
    f.write(str(test))