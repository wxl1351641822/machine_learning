import torch
import torch.nn as nn
import torch.optim as optim
from processData import *
from tqdm import tqdm


torch.manual_seed(1)

def prepare_sequence(seq, to_ix):
    idxs = [to_ix[w] for w in seq]
    return torch.tensor(idxs, dtype=torch.long)

class BiLSTM_CRF(nn.Module):

    def __init__(self, vocab_size, tag_to_ix, embedding_dim, hidden_dim):
        super(BiLSTM_CRF, self).__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.tag_to_ix = tag_to_ix
        self.tagset_size = len(tag_to_ix)

        self.word_embeds = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim // 2,
                            num_layers=1, bidirectional=True)

        # Maps the output of the LSTM into tag space.
        self.hidden2tag = nn.Linear(hidden_dim, self.tagset_size)

        # Matrix of transition parameters.  Entry i,j is the score of
        # transitioning *to* i *from* j.
        self.transitions = nn.Parameter(
            torch.randn(self.tagset_size, self.tagset_size))

        # These two statements enforce the constraint that we never transfer
        # to the start tag and we never transfer from the stop tag
        self.transitions.data[tag_to_ix[START_TAG], :] = -10000
        self.transitions.data[:, tag_to_ix[STOP_TAG]] = -10000

        self.hidden = self.init_hidden()

    def init_hidden(self):
        return (torch.randn(2, 1, self.hidden_dim // 2),
                torch.randn(2, 1, self.hidden_dim // 2))

    def _forward_alg(self, feats):
        # Do the forward algorithm to compute the partition function
        init_alphas = torch.full([self.tagset_size], -10000.)
        # START_TAG has all of the score.
        init_alphas[self.tag_to_ix[START_TAG]] = 0.

        # Wrap in a variable so that we will get automatic backprop
        # Iterate through the sentence
        forward_var_list=[]
        forward_var_list.append(init_alphas)
        for feat_index in range(feats.shape[0]):
            gamar_r_l = torch.stack([forward_var_list[feat_index]] * feats.shape[1])
            t_r1_k = torch.unsqueeze(feats[feat_index],0).transpose(0,1)
            aa = gamar_r_l + t_r1_k + self.transitions
            forward_var_list.append(torch.logsumexp(aa,dim=1))
        terminal_var = forward_var_list[-1] + self.transitions[self.tag_to_ix[STOP_TAG]]
        terminal_var = torch.unsqueeze(terminal_var,0)
        alpha = torch.logsumexp(terminal_var, dim=1)[0]
        return alpha

    def _get_lstm_features(self, sentence):
        self.hidden = self.init_hidden()
        embeds = self.word_embeds(sentence).view(len(sentence), 1, -1)
        lstm_out, self.hidden = self.lstm(embeds, self.hidden)
        lstm_out = lstm_out.view(len(sentence), self.hidden_dim)
        lstm_feats = torch.tanh(self.hidden2tag(lstm_out))
        return lstm_feats

    def _score_sentence(self, feats, tags):
        # Gives the score of a provided tag sequence
        score = torch.zeros(1)
        tags = torch.cat([torch.tensor([self.tag_to_ix[START_TAG]], dtype=torch.long), tags])
        for i, feat in enumerate(feats):
            score = score + \
                self.transitions[tags[i + 1], tags[i]] + feat[tags[i + 1]]
        score = score + self.transitions[self.tag_to_ix[STOP_TAG], tags[-1]]
        return score

    def _viterbi_decode(self, feats):
        backpointers = []
        # Initialize the viterbi variables in log space
        init_vvars = torch.full((1, self.tagset_size), -10000.)
        init_vvars[0][self.tag_to_ix[START_TAG]] = 0

        # forward_var at step i holds the viterbi variables for step i-1
        forward_var_list = []
        forward_var_list.append(init_vvars)

        for feat_index in range(feats.shape[0]):
            gamar_r_l = torch.stack([forward_var_list[feat_index]] * feats.shape[1])
            gamar_r_l = torch.squeeze(gamar_r_l)
            next_tag_var = gamar_r_l + self.transitions
            viterbivars_t,bptrs_t = torch.max(next_tag_var,dim=1)

            t_r1_k = torch.unsqueeze(feats[feat_index], 0)
            forward_var_new = torch.unsqueeze(viterbivars_t,0) + t_r1_k

            forward_var_list.append(forward_var_new)
            backpointers.append(bptrs_t.tolist())

        # Transition to STOP_TAG
        terminal_var = forward_var_list[-1] + self.transitions[self.tag_to_ix[STOP_TAG]]

        best_tag_id = torch.argmax(terminal_var).tolist()
        # print(best_tag_id)
        path_score = terminal_var[0][best_tag_id]

        # Follow the back pointers to decode the best path.
        best_path = [best_tag_id]
        for bptrs_t in reversed(backpointers):
            best_tag_id = bptrs_t[best_tag_id]
            best_path.append(best_tag_id)
        # Pop off the start tag (we dont want to return that to the caller)
        start = best_path.pop()
        assert start == self.tag_to_ix[START_TAG]  # Sanity check
        best_path.reverse()
        return path_score, best_path


    def neg_log_likelihood(self, sentence, tags):
        feats = self._get_lstm_features(sentence)
        forward_score = self._forward_alg(feats)
        gold_score = self._score_sentence(feats, tags)
        return forward_score - gold_score

    def forward(self, sentence):  # dont confuse this with _forward_alg above.
        # Get the emission scores from the BiLSTM
        lstm_feats = self._get_lstm_features(sentence)
        # print(lstm_feats.shape)
        # Find the best path, given the features.
        score, tag_seq = self._viterbi_decode(lstm_feats)
        return score, tag_seq

def measure(predict,y):
    acc = (torch.sum(torch.eq(predict, y))).type(torch.FloatTensor) / float(len(y))
    TP=torch.zeros(7,dtype=float)
    FP=torch.zeros(7,dtype=float)
    FN=torch.zeros(7,dtype=float)
    for i in range(len(y)):
        if(y[i]==predict[i]):
            TP[y[i]-1]+=1
        else:
            FP[predict[i]-1]+=1
            FN[y[i]-1]+=1
    # micro:算总的
    # print(torch.sum(TP))
    print(TP)
    print(FP)
    print(FN)
    micro_precision=torch.sum(TP)/(torch.sum(TP)+torch.sum(FP))
    micro_recall=torch.sum(TP)/(torch.sum(TP)+torch.sum(FN))
    micro_F1=2*(micro_precision*micro_recall)/(micro_precision+micro_recall)
    # macro ：算每一类的然后平均
    # TP[TP==0]=1e-8
    # FP[FP==0]=1e-8
    # FN[FN==0]=1e-8
    macro_precision=TP/(TP+FP)
    macro_recall=TP/(TP+FN)

    macro_F1=2*(macro_recall*macro_precision)/(macro_recall+macro_precision)
    print(macro_F1)
    macro_F1=torch.mean(macro_F1)
    print(acc,micro_F1,macro_F1)
    return acc,micro_F1,macro_F1
if __name__== '__main__':
    START_TAG = "<BEG>"
    STOP_TAG = "<END>"
    EMBEDDING_DIM = 300
    HIDDEN_DIM = 256
    training_data, dic_word_list, dic_label_list, word_to_ix, tag_to_ix = getAllTrain()

    # Make up some training data
    # training_data = [(
    #     "the wall street journal reported today that apple corporation made money".split(),
    #     "B I I I O O O B I O O".split()
    # ), (
    #     "georgia tech is a university in georgia".split(),
    #     "B I O O O O B".split()
    # )]
    #
    # word_to_ix = {}
    # for sentence, tags in training_data:
    #     for word in sentence:
    #         if word not in word_to_ix:
    #             word_to_ix[word] = len(word_to_ix)
    #
    # tag_to_ix = {"B": 0, "I": 1, "O": 2, START_TAG: 3, STOP_TAG: 4}
    #
    model = BiLSTM_CRF(len(dic_word_list), tag_to_ix, EMBEDDING_DIM, HIDDEN_DIM)
    # print(list(model.named_parameters()))
    # print(list(model.parameters()))
    optimizer = optim.SGD(model.parameters(), lr=0.01, weight_decay=1e-4)
    torch.save(model, 'model/net0.pkl')
    # Check predictions before training
    with torch.no_grad():
        precheck_sent = torch.tensor(training_data[0][0])
        # print(precheck_sent)
        precheck_tags = torch.tensor(training_data[1][0])
        print(model(precheck_sent))

    # Make sure prepare_sequence from earlier in the LSTM section is loaded
    for epoch in range(1):  # again, normally you would NOT do 300 epochs, it is toy data
        for sentence, tags in tqdm(zip(training_data[0][:-2000],training_data[1][:-2000])):
            # print(sentence,tags)
            # Step 1. Remmber that Pytorch accumulates gradients.
            # We need to clear them out before each instance
            model.zero_grad()
            sentence_in=torch.tensor(sentence)
            targets=torch.tensor(tags)
            # Step 2. Get our inputs ready for the network, that is,
            # turn them into Tensors of word indices.
            # sentence_in = prepare_sequence(sentence, word_to_ix)
            # targets = torch.tensor([tag_to_ix[t] for t in tags], dtype=torch.long)
            score,predict=model(sentence_in)
            # Step 3. Run our forward pass.
            loss = model.neg_log_likelihood(sentence_in, targets)

            # Step 4. Compute the loss, gradients, and update the parameters by
            # calling optimizer.step()
            loss.backward()
            optimizer.step()
        torch.save(model.state_dict(), 'model/net0_%d_params.pkl' % epoch)

    # Check predictions after training
    with torch.no_grad():
        # precheck_sent = torch.tensor(training_data[0][0])
        # print(model(precheck_sent))
        y = torch.tensor([training_data[1][-2000]])
        sentence_in = torch.tensor(training_data[0][-2000])
        tag_scores,predict1 = model(sentence_in)

        predict = torch.tensor([predict1])

        for sentence, tags in tqdm(zip(training_data[0][-2001:], training_data[1][-2001:])):
            # 准备网络输入, 将其变为词索引的 Tensor 类型数据
            sentence_in = torch.tensor(sentence)
            # targets = torch.tensor(tags)
            tag_scores,predict1 = model(sentence_in)

            predict = torch.cat((predict, torch.tensor([predict1])), axis=1)
            y = torch.cat((y, torch.tensor([tags])), axis=1)

            x0 = [dic_word_list[s] for s in sentence]
            y0 = [dic_label_list[t] for t in tags]
            predict0 = [dic_label_list[t] for t in predict1]
            print(x0)
            print(y0)
            print(predict0)
        # print(predict.shape)
        # print(y.shape)
        measure(predict.reshape(y.shape[1]), y.reshape(y.shape[1]))

