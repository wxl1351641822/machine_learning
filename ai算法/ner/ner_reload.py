import torch
from ner_model import *
print("0.获得词典")
dic_label2ids,dic_word2ids,train,ids2label=getData('CoNLL-2003/eng.train')
print("1.导入模型")
# model = torch.load('model/net.pkl')
transition=tran(dic_label2ids,train)
model=BiLSTM_CRF(len(dic_word2ids), dic_label2ids, INPUT_SIZE, HIDDEN_SIZE,dic_label2ids,transition)
print("2.导入模型参数")
model.load_state_dict(torch.load('model/net_0_80_params.pkl'))
print("3.输入要预测的句子")
sentence="I am in China"
def predict(sentence):
    print("4.转换成词向量列表")
    sentence_in = prepare_sequence(sentence, dic_word2ids)#从词的列表变成了id的列表
    print("5.预测：")
    prediction=model(sentence_in)
    predict_label=[ids2label[i] for i in prediction[1]]
    print("预测结果:",predict_label)
    return prediction

_,_,valid,_=getData('CoNLL-2003/eng.testa')
for sentence,l in valid:

    print(sentence,l)
    prediction=predict(sentence)
    l=torch.tensor([dic_label2ids[i] for i in l])

    # l = torch.tensor([dic_label2ids[i] for i in prediction])
    acc= (l == prediction[1]).sum().float() / len(sentence)
    print(acc,l,prediction[1])
# while(True):
#     sentence=input("输入要预测的句子：")
#     predict(sentence)
