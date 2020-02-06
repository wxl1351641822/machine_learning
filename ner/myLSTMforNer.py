from myLSTM import *
from processData import *

traindata,dic_word_list,dic_label_list,dic_word,dic_label=getAllTrain()
# print(traindata)
EMBEDDING_DIM=300
HIDDEN_DIM=10
model = LSTMTag(EMBEDDING_DIM, HIDDEN_DIM, len(dic_word_list), len(dic_label_list),1)

for sentence,tag in zip(traindata[0],traindata[1]):
    x=torch.tensor(sentence)
    y=torch.tensor(tag)
    tag_score=model.forward(x)
    model.BP(y)
    # print(torch.max(tag_score,axis=1))

##会出现nan？怎么办。