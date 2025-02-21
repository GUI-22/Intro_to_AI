import torch
from gensim.models import KeyedVectors
from torch import nn
from torch import optim
from torch.nn import functional
import utils


# 设置超参数

batch_size = 128
seq_len = 100

#预处理训练、测评数据
word_vectors = KeyedVectors.load_word2vec_format("./Dataset/wiki_word2vec_50.bin", binary=True)
#对于未知的词语，用随机一个vector表示
unk_vector = torch.randn(50, dtype=torch.float32) * 0.01
train_iter = utils.data_loader("./Dataset/train.txt", word_vectors, unk_vector, batch_size, True, seq_len)
test_iter = utils.data_loader("./Dataset/test.txt", word_vectors, unk_vector, batch_size, False, seq_len)
validation_iter = utils.data_loader("./Dataset/validation.txt", word_vectors, unk_vector, batch_size, False, seq_len)

class GlobalMaxPool_1d(nn.Module):
    def __init__(self):
        super(GlobalMaxPool_1d, self).__init__()
    def forward(self, X):
        # X.shape == (batch_size, channels, seq_len)
        # Pool之后X.shape == (batch_size, channels, 1)，即每个channels内部取max
        return functional.max_pool1d(X, kernel_size=X.shape[2])
    
class text_CNN(nn.Module):
    def __init__(self, kernel_sizes, channels, embed_size=50, drop_out=0.5):
        super(text_CNN, self).__init__()
        self.convs = nn.ModuleList()
        for channel, kernel_size in zip(channels, kernel_sizes):
            self.convs.append(nn.Conv1d(in_channels=embed_size, out_channels=channel, kernel_size=kernel_size))
        self.pool = GlobalMaxPool_1d()
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(drop_out)
        self.linear = nn.Linear(sum(channels), 2)
    def forward(self, X):
        # X.shape == (batch_size, seq_len, embed_size)
        # 输入conv时最后一维应该是“代表一个句子的一部分”，因此要permute
        X = X.permute(0, 2, 1)
        # conv之后 shape == (batch_size, out_channel, seq_len)
        # max_pool之后 shape == (batch_size, out_channel, 1)
        # squeeze之后 shape == (batch_size, out_channel)
        outputs = [self.pool(self.relu(conv(X))) for conv in self.convs]
        outputs = [output.squeeze(-1) for output in outputs]
        encoded = torch.cat(outputs, dim=-1)
        return self.linear(self.dropout(encoded))
    

class BiRNN_LSTM(nn.Module):
    def __init__(self, num_hiddens, num_layers, drop_out=0, embed_size=50):
        super(BiRNN_LSTM, self).__init__()
        # 以下运用了双向LSTM
        self.encoder = nn.LSTM(input_size=embed_size, 
                                hidden_size=num_hiddens, 
                                num_layers=num_layers,
                                bidirectional=True)
        self.dropout = nn.Dropout(drop_out)
        self.decoder = nn.Linear(4*num_hiddens, 2)

    def forward(self, inputs):
        outputs, _ = self.encoder(inputs.permute(1, 0, 2)) # output, (h, c)
        encoding = torch.cat((outputs[0], outputs[-1]), -1)
        return self.decoder(self.dropout(encoding))


class BiRNN_GRU(nn.Module):
    def __init__(self, num_hiddens, num_layers, drop_out=0, embed_size=50):
        super(BiRNN_GRU, self).__init__()
        self.encoder = nn.GRU(input_size=embed_size, 
                                hidden_size=num_hiddens, 
                                num_layers=num_layers,
                                bidirectional=True)
        self.dropout = nn.Dropout(drop_out)
        self.decoder = nn.Linear(4*num_hiddens, 2)

    def forward(self, inputs):
        outputs, _ = self.encoder(inputs.permute(1, 0, 2)) # output, (h, c)
        encoding = torch.cat((outputs[0], outputs[-1]), -1)
        return self.decoder(self.dropout(encoding))
    

class MLP(nn.Module):
    def __init__(self, num_hiddens, num_layers, drop_out=0, embed_size=50):
        super(MLP, self).__init__()
        self.num_layers = num_layers
        self.fc_layers = nn.ModuleList([nn.Linear(embed_size * seq_len if i == 0 else num_hiddens, num_hiddens) for i in range(num_layers)])
        self.output_layer = nn.Linear(num_hiddens, 2)
        self.dropout = nn.Dropout(drop_out)

    def forward(self, x):
        batch_size, seq_len, embed_size = x.size()
        x = x.view(batch_size, -1)
        for i in range(self.num_layers):
            x = torch.relu(self.fc_layers[i](x))
            x = self.dropout(x)
        x = self.output_layer(x)
        return x

# net = text_CNN(kernel_sizes=[3, 4], channels=[50, 50], drop_out=0)
# net = BiRNN_LSTM(num_hiddens=48, num_layers=1, drop_out=0.4)
# net = BiRNN_GRU(num_hiddens=48, num_layers=1, drop_out=0.4)
net = MLP(num_hiddens=48, num_layers=3, drop_out=0.4)

net = utils.initialize_model(net, None)

# model_path = "./saved_path/saved_CNN.pth"
# model_path = "./saved_path/saved_LSTM.pth"
# model_path = "./saved_path/saved_GRU.pth"
model_path = "./saved_path/saved_MLP.pth"


lr = 1e-2
num_epochs = 20
optimizer = optim.Adam(net.parameters(), lr)
loss = nn.CrossEntropyLoss()
device = torch.device("cuda")
utils.train(train_iter, validation_iter, test_iter, net, loss, optimizer, device, num_epochs, model_path)

net.load_state_dict(torch.load(model_path))
test_acc, test_f_score = utils.eval_acc(net, test_iter, device)
print('total_epoch %d, test_acc %.3f, test_f_score, %.3f' % (num_epochs, test_acc, test_f_score))

