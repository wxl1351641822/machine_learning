import torch
import torch.nn as nn



torch.manual_seed(1)

# Hyper Parameters
EPOCH = 100               # train the training data n times, to save time, we just train 1 epoch
# BATCH_SIZE = 64
# TIME_STEP = max_sentence_len          # rnn time step / image height
START_TAG = "<START>"
STOP_TAG = "<STOP>"
NOWORD="<NOWORD>"
HIDDEN_SIZE=50
INPUT_SIZE = 10000         # embedding_size
LR = 0.01               # learning rate
BATCH_size=100


def argmax(vec):
    # return the argmax as a python int
    _, idx = torch.max(vec, 1)
    return idx.item()


def prepare_sequence(seq, to_ix):
    idxs=[]
    for w in seq:
        try:
            idxs.append(to_ix[w])
        except KeyError:
            idxs.append(len(to_ix)-1)
    return torch.tensor(idxs, dtype=torch.long)


# Compute log sum exp in a numerically stable way for the forward algorithm
def log_sum_exp(vec):
    max_score = vec[0, argmax(vec)]
    max_score_broadcast = max_score.view(1, -1).expand(1, vec.size()[1])
    return max_score + \
        torch.log(torch.sum(torch.exp(vec - max_score_broadcast)))

def tran(dic_label2ids,train):
    trainsition = torch.zeros(len(dic_label2ids), len(dic_label2ids))

    # print(trainsition)
    # print(trainsition.shape)

    for sentence, labels in train:
        for i in range(len(sentence) - 1):
            trainsition[dic_label2ids[labels[i]]][dic_label2ids[labels[i + 1]]] += 1
    trainsition = trainsition.float() + 0.00001
    trainsition = trainsition / torch.sum(trainsition, dim=1).view(10, 1)
    return trainsition


class BiLSTM_CRF(nn.Module):

    def __init__(self, vocab_size, tag_to_ix, embedding_dim, hidden_dim,dic_label2ids,trans):
        super(BiLSTM_CRF, self).__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.tag_to_ix = tag_to_ix
        self.tagset_size = len(tag_to_ix)

        self.word_embeds = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim // 2,
                            num_layers=4, bidirectional=True)

        # Maps the output of the LSTM into tag space.
        self.hidden2tag = nn.Linear(hidden_dim, self.tagset_size)

        # Matrix of transition parameters.  Entry i,j is the score of
        # transitioning *to* i *from* j.
        # self.transitions = nn.Parameter(
        #     torch.randn(self.tagset_size, self.tagset_size))a
        self.transitions=nn.Parameter(trans)
        print(trans.shape)
        print(dic_label2ids)
        # These two statements enforce the constraint that we never transfer
        # to the start tag and we never transfer from the stop tag
        self.transitions.data[dic_label2ids[START_TAG], :] = -10000
        self.transitions.data[:, dic_label2ids[STOP_TAG]] = -10000

        self.hidden = self.init_hidden()


    def init_hidden(self):
        return (torch.randn(8, 1, self.hidden_dim // 2),
                torch.randn(8, 1, self.hidden_dim // 2))

    def _forward_alg(self, feats):#前向算法--

        init_alphas = torch.full((1, self.tagset_size), -10000.)#5个-10000

        # 初始时,start位置为0，其他位置为-10000
        init_alphas[0][self.tag_to_ix[START_TAG]] = 0.

        # 赋给变量方便后面反向传播
        forward_var = init_alphas

        for feat in feats:
            alphas_t = []
            for next_tag in range(self.tagset_size):#每个时间点的feat--也就是crf中的状态，next_tag是时间步
                # 状态特征函数的得分
                emit_score = feat[next_tag].view(1, -1).expand(1, self.tagset_size)
                # print(emit_score)

                # 状态转移函数的得分
                trans_score = self.transitions[next_tag].view(1, -1)#这个是随机初始化的
                # print(trans_score)
                # 从上一个单词的每个状态转移到next_tag状态的得分
                # 所以next_tag_var是一个大小为tag_size的数组
                next_tag_var = forward_var + trans_score + emit_score#因为是log的。

                # 对next_tag_var进行log_sum_exp操作
                alphas_t.append(log_sum_exp(next_tag_var).view(1))

            # print(alphas_t)
            forward_var = torch.cat(alphas_t).view(1, -1)
            # print(forward_var)
        terminal_var = forward_var + self.transitions[self.tag_to_ix[STOP_TAG]]
        alpha = log_sum_exp(terminal_var)
        return alpha

    def _get_lstm_features(self, sentence):
        self.hidden = self.init_hidden()
        # print(sentence)
        embeds = self.word_embeds(sentence).view(len(sentence), 1, -1)#变成了词向量
        lstm_out, self.hidden = self.lstm(embeds, self.hidden)
        lstm_out = lstm_out.view(len(sentence), self.hidden_dim)
        lstm_feats = self.hidden2tag(lstm_out)
        # print(lstm_feats)
        return lstm_feats

    def _score_sentence(self, feats, tags):
        # Gives the score of a provided tag sequence
        score = torch.zeros(1)
        tags = torch.cat([torch.tensor([self.tag_to_ix[START_TAG]], dtype=torch.long), tags])
        # print("tags",tags)
        for i, feat in enumerate(feats):
            score = score + \
                self.transitions[tags[i + 1], tags[i]] + feat[tags[i + 1]]
        score = score + self.transitions[self.tag_to_ix[STOP_TAG], tags[-1]]
        return score

    def _viterbi_decode(self, feats):
        backpointers = []
        # 初始化
        init_vvars = torch.full((1, self.tagset_size), -10000.)
        init_vvars[0][self.tag_to_ix[START_TAG]] = 0

        forward_var = init_vvars
        for feat in feats:
            # 保持路径节点,用于重构最优路径
            bptrs_t = []
            # 保持路径变量概率
            viterbivars_t = []

            for next_tag in range(self.tagset_size):
                next_tag_var = forward_var + self.transitions[next_tag]
                best_tag_id = argmax(next_tag_var)
                bptrs_t.append(best_tag_id)
                viterbivars_t.append(next_tag_var[0][best_tag_id].view(1))

            forward_var = (torch.cat(viterbivars_t) + feat).view(1, -1)
            backpointers.append(bptrs_t)

        # 转移到STOP_TAG
        terminal_var = forward_var + self.transitions[self.tag_to_ix[STOP_TAG]]
        best_tag_id = argmax(terminal_var)
        path_score = terminal_var[0][best_tag_id]

        # 反向迭代求最优路径
        best_path = [best_tag_id]
        for bptrs_t in reversed(backpointers):
            best_tag_id = bptrs_t[best_tag_id]
            best_path.append(best_tag_id)

        # 把start_tag pop出来，最终的结果不需要
        start = best_path.pop()
        assert start == self.tag_to_ix[START_TAG]
        best_path.reverse()
        return path_score, torch.tensor(best_path)

    def neg_log_likelihood(self, sentence, tags):
        feats = self._get_lstm_features(sentence)
        forward_score = self._forward_alg(feats)
        gold_score = self._score_sentence(feats, tags)

        return forward_score - gold_score

    def forward(self, sentence):  # dont confuse this with _forward_alg above.
        # Get the emission scores from the BiLSTM
        lstm_feats = self._get_lstm_features(sentence)

        # Find the best path, given the features.
        score, tag_seq = self._viterbi_decode(lstm_feats)
        return score, tag_seq



def getData(filepath):
    with open(filepath,'r',encoding='utf-8') as f:
        train_data=f.readlines()
    row_w=[]
    row_l=[]
    train=[]
    label_set=set()
    word_set=set()
    for x in train_data:
        r=x[:-1].split(' ')

        if(len(r)<=1):
            train.append((row_w,row_l))
            row_w = []
            row_l = []
            # row=[]
        else:
            row_w.append(r[0].lower())
            row_l.append(r[3])
            word_set.add(r[0].lower())
            label_set.add(r[3])

    ids2word=list(word_set)
    ids2label=list(label_set)
    dic_word2ids={ids2word[i]:i for i in range(len(ids2word))}
    dic_label2ids={ids2label[i]:i for i in range(len(ids2label))}
    dic_label2ids[START_TAG]=len(dic_label2ids)
    dic_label2ids[STOP_TAG]=len(dic_label2ids)
    dic_word2ids[NOWORD]=len(dic_word2ids)
    return dic_label2ids,dic_word2ids,train,ids2label




