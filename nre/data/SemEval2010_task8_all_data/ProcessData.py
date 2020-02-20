from stanfordcorenlp import StanfordCoreNLP
from tqdm import tqdm
import constant
import re

def clean_str(text):
    text = text.lower()
    # Clean the text
    text = re.sub(r"[^A-Za-z0-9^,!.\/'+-=]", " ", text)
    text = re.sub(r"what's", "what is ", text)
    text = re.sub(r"that's", "that is ", text)
    text = re.sub(r"there's", "there is ", text)
    text = re.sub(r"it's", "it is ", text)
    text = re.sub(r"\'s", " ", text)
    text = re.sub(r"\'ve", " have ", text)
    text = re.sub(r"can't", "can not ", text)
    text = re.sub(r"n't", " not ", text)
    text = re.sub(r"i'm", "i am ", text)
    text = re.sub(r"\'re", " are ", text)
    text = re.sub(r"\'d", " would ", text)
    text = re.sub(r"\'ll", " will ", text)
    text = re.sub(r",", " ", text)
    text = re.sub(r"\.", " ", text)
    text = re.sub(r"!", " ! ", text)
    text = re.sub(r"\/", " ", text)
    text = re.sub(r"\^", " ^ ", text)
    text = re.sub(r"\+", " + ", text)
    text = re.sub(r"\-", " - ", text)
    text = re.sub(r"\=", " = ", text)
    text = re.sub(r"'", " ", text)
    text = re.sub(r"(\d+)(k)", r"\g<1>000", text)
    text = re.sub(r":", " : ", text)
    text = re.sub(r" e g ", " eg ", text)
    text = re.sub(r" b g ", " bg ", text)
    text = re.sub(r" u s ", " american ", text)
    text = re.sub(r"\0s", "0", text)
    text = re.sub(r" 9 11 ", "911", text)
    text = re.sub(r"e - mail", "email", text)
    text = re.sub(r"j k", "jk", text)
    text = re.sub(r"\s{2,}", " ", text)
    return text.strip()

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
            sentence = sentence.replace('<e1>', ' _'+constant.SUBJ_START+'_ ')
            sentence = sentence.replace('</e1>',  ' _'+constant.SUBJ_END+'_ ')
            sentence = sentence.replace('<e2>',  ' _'+constant.OBJ_START+'_ ')
            sentence = sentence.replace('</e2>',  ' _'+constant.OBJ_END+'_ ')
            sentence=clean_str(sentence)
            # print(sentence)
            # print(id)
            # print(sentence[1:-2])
            # sentence = sentence.replace(constant.SUBJ_START,' '+constant.SUBJ_START+' ')
            # sentence = sentence.replace(constant.SUBJ_END, ' '+constant.SUBJ_END+' ')
            # sentence = sentence.replace(constant.OBJ_START, ' '+constant.OBJ_START+' ')
            # sentence = sentence.replace(constant.OBJ_END, ' '+constant.OBJ_END+' ')
            token=sentence.split()
            for j,t in enumerate(token):
                if(constant.SUBJ_START in t):
                    e1_start=j
                if (constant.SUBJ_END in t):
                    e1_end = j
                if (constant.OBJ_START in t):
                    e2_start = j
                if (constant.OBJ_END in t):
                    e2_end = j
            row={}
            # print(token)
            # print(e1_start,e1_end,e2_start,e2_end)

            row['id']=int(id)
            row['sentence']=sentence
            row['token']=token
            pos,dependency,dependency_head=stanford_pos(nlp,sentence)
            row['stanford_pos']=pos
            row['stanford_dependency']=dependency
            row['stanford_dependency_head'] = dependency_head
            row['relation_subj_start'] = e1_start
            row['relation_subj_end'] = e1_end
            row['relation_obj_start'] = e2_start
            row['relation_obj_end'] = e2_end
            # print(row)
        elif(i%4==1):
            # print(data[:-1])
            if(data[:-1]=='other'):
                # print(data['relation'])
                row['relation']='other'
            else:
                row['relation']=data[:-1]

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






train=read_semEval('./TRAIN_FILE.TXT')
with open('./mytrain.json', 'w', encoding='utf-8') as f:
    f.write(str(train))
test=read_semEval('./TEST_FILE_FULL.TXT')
with open('./mytest.json', 'w', encoding='utf-8') as f:
    f.write(str(test))


# get_labellist(constant.train_path,constant.label_path)