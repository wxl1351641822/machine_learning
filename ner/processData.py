import torch
def getDict(list):
    dict={}
    for index,item in enumerate(list):
        dict[item]=index
    return dict

def getData(rowData,dic_label,dic_word):
    data_x =[]
    data_y = []
    # print(dic_word)
    for row in rowData:
        x_row=[]
        y_row=[]
        # print(row)
        for x in row:
            # print(x)
            x_row.append(dic_word[x[0]])
            y_row.append(dic_label[x[1]])
        data_x.append(x_row)
        data_y.append(y_row)
    return [data_x,data_y]

def getword_set(len1,x):
    word_set=[]
    for i in range(len1):
        word_set.append([])
    for i in range(len(x[0])):
        for j in range(len(x[0][i])):
            word_set[x[1][i][j]].append(x[0][i][j])
    return word_set


def readFile(filepath):
    with open(filepath,'r',encoding='utf-8') as f:
        article=f.readlines();
        article_split=[x.split() for x in article]

        result=[]
        dic_word_list=set()
        dic_label_list=set()
        row=[]
        for x in article_split:
            if(len(x)==0):
                if(len(row)>0):
                    # print(row)

                    result.append(row)
                    row=[]
            else:
                row.append([x[0],x[len(x)-1]])
                dic_word_list.add(x[0])
                dic_label_list.add(x[len(x)-1])

        if (len(row) > 0):
            # print(row)
            result.append(row)
        # print(result)

        dic_word_list.add("<UNK>")
        dic_word_list = list(dic_word_list)

        dic_label_list = list(dic_label_list)
        dic_label_list.insert(0,'<BEG>')
        dic_label_list.append('<END>')
        with open('./dict/dic_word_list', 'w', encoding='utf-8') as f:
            f.write(str(dic_word_list))
        with open('./dict/dic_label_list', 'w', encoding='utf-8') as f:
            f.write(str(dic_label_list))
        with open('./dict/train', 'w', encoding='utf-8') as f:
            f.write(str(result))
        dic_word = getDict(dic_word_list)
        dic_label = getDict(dic_label_list)
        with open('./dict/dic_word', 'w', encoding='utf-8') as f:
            f.write(str(dic_word))
        with open('./dict/dic_label', 'w', encoding='utf-8') as f:
            f.write(str(dic_label))
        traindata = getData(result, dic_label, dic_word)
        with open('./dict/traindata', 'w', encoding='utf-8') as f:
            f.write(str(traindata))
        word_set=getword_set(len(dic_label),traindata)
        with open('./dict/word_set', 'w', encoding='utf-8') as f:
            f.write(str(word_set))
        return traindata,dic_word_list,dic_label_list,dic_word,dic_label

def quickReadFile_eval(path):
    s=''
    with open(path,'r',encoding='utf-8') as f:
        s=f.read()
    return eval(s)

def getAllTrain():
    traindata=quickReadFile_eval('./dict/traindata')
    dic_word_list=quickReadFile_eval('./dict/dic_word_list')
    dic_label_list=quickReadFile_eval('./dict/dic_label_list')
    dic_word=quickReadFile_eval('./dict/dic_word')
    dic_label=quickReadFile_eval('./dict/dic_label')
    return traindata,dic_word_list,dic_label_list,dic_word,dic_label


#
#
# def getTransitionFromData_(x,len_label):
#     transition=torch.ones(len_label,len_label)
#     for i in range(len(x) - 1):
#         # print(x[i,1],,x[i+1,1])
#         transition[x[i][1],x[i+1][1]]+=1
#     # print(transition)
#     # print(torch.sum(transition,axis=1))
#     # print()
#     # print(1/45,15/45)
#     return transition/torch.sum(transition,axis=1).reshape(len_label,1)
#     # torch.sum(transition[0])
#
#
# def getb_(x,len_label,len_word):
#     b= torch.ones(len_label, len_word)
#     for i in range(len(x)):
#             b[x[i][1],x[i][0]]+=1
#     # print(b)
#     # print(torch.sum(b,axis=1))
#     # print(b/torch.sum(b,axis=1).reshape(len_label,1))
#     return b/torch.sum(b,axis=1).reshape(len_label,1)

# readFile("./CoNLL-2003/eng.train")
# readFile("./corpus/dh_msra.txt")
# traindata,dic_word_list,dic_label_list,dic_word,dic_label=getAllTrain();
# print(traindata)
# getTransitionFromData(traindata,len(dic_label_list))
# getb(traindata,len(dic_label_list),len(dic_word_list))

