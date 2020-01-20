from ner_model import *

dic_label2ids,dic_word2ids,train,ids2label=getData('CoNLL-2003/eng.train')




# import torch
# t=torch.FloatTensor([[1,2,3],[1,2,3],[1,2,3]])
# a=torch.FloatTensor([2,3,4]).view(3,1)
# print(t,a)
# print(t/a)