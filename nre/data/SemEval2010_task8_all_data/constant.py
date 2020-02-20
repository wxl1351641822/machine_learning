num_epoch=200
lr_decay=0.9
weight_decay=1e-5
lr=0.01
checkpoints=5
checkpoint_path='./model/ATTCNN_nonline_lextual_1'

glove_dim=100
lstm_hidden_dim=100
vocab_size=19197
label_size=19
max_length=90
batch_size=16
pos_embedding_dim=5
dropout=0.5
rnn_dropout=0.7
emb_dropout=0.7




filter_list=[3,4,5]
filter_num=100
cnn_hidden_dim=100
cnn_window=3
"""
lstm的最好参数：
acc=0.7
batch_size=50
无pos
max_length=90
lstm_hidden_dim=128/180

cnn1:acc=0.68
glove_dim=100
lstm_hidden_dim=100
vocab_size=19197
label_size=19
max_length=90
batch_size=16
pos_embedding_dim=10
dropout=0.5
filter_list=[3,4,5]
filter_num=60
cnn_window=3
cnn2:
ATTCNN
glove_dim=100
lstm_hidden_dim=100
vocab_size=19197
label_size=19
max_length=90
batch_size=50
pos_embedding_dim=5
dropout=0.5
filter_list=[3,4,5]
filter_num=60
cnn_window=3

cnn-nonline1:0.6925-93
lstm_hidden_dim=100
vocab_size=19197
label_size=19
max_length=90
batch_size=50
pos_embedding_dim=5
dropout=0.5
filter_list=[3,4,5]
filter_num=60
cnn_hidden_dim=100

cnn_window=3

cnn-concate4:0.697
glove_dim=100
lstm_hidden_dim=100
vocab_size=19197
label_size=19
max_length=90
batch_size=50
pos_embedding_dim=5
dropout=0.5
filter_list=[3,4,5]
filter_num=60
cnn_hidden_dim=100

cnn_window=3


"""

SUBJ_START='e11'
SUBJ_END='e12'
OBJ_START='e21'
OBJ_END='e22'
PAD='<PAD>'
PAD_ID=0
UNK='<UNK>'
UNK_ID=1
VOCAB_PREFIX=[PAD,UNK]


train_path='./mytrain.json'
test_path='./mytest.json'
glove_path='../glove/glove.6B.100d.txt'
emb_path='vocab/embedding.npy'
vocab_path='vocab/vocab.list'
label_path='vocab/label.list'
pos_path='vocab/pos.list'
dependency_path='vocab/dependency.list'

att_lstm_traindata_path='feature_idlist/train.idlist'
att_lstm_testdata_path='feature_idlist/test.idlist'
position_feature_path='feature_idlist/position_feature'
