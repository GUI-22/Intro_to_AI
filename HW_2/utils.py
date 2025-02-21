import torch
import time
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.init as init

def read_file(path):
    items = []
    with open(path, "r", encoding='utf-8') as in_file:
        for line in in_file:
            items.append(line.split())
    # random.shuffle(items)
    labels = []
    sentences = []
    for item in items:
        labels.append(int(item[0]))
        sentences.append(item[1:])
    return torch.tensor(labels, dtype=torch.long), sentences

def trunc_pad(sentences, word_vectors, unk_vector, target_len=80):
    # new_sentences.shape (num_sentence, num_target_len, 50)
    new_sentences = []
    for sentence in sentences:
        new_sentence = []
        # trunc & pad
        if len(sentence) < target_len:
            sentence = sentence + ["<pad>"] * (target_len - len(sentence))
        else:
            sentence = sentence[:target_len]
        # convert to word_vector
        for word in sentence:
            if word in word_vectors:
                new_sentence.append(torch.tensor(word_vectors[word], dtype=torch.float32))
            elif word == "<pad>":
                new_sentence.append(torch.zeros(50, dtype=torch.float32))
            else:
                new_sentence.append(unk_vector)
        new_sentences.append(torch.stack(new_sentence))
    return torch.stack(new_sentences)

def data_loader(path, word_vectors, unk_vector, batch_size, is_train=True, target_len=80):
    # train有19998条example
    labels, sentences = read_file(path)
    sentences = trunc_pad(sentences, word_vectors, unk_vector, target_len=target_len)
    # 注意：进入DataLoader之前，先把标签、数据合并成TensorDataset
    dataset = TensorDataset(sentences, labels)
    data_iter = DataLoader(dataset, batch_size=batch_size, shuffle=is_train)
    return data_iter


# 以下代码仿照dive_2_deepLearning逻辑实现的
# 参考https://zh.d2l.ai/chapter_natural-language-processing-applications/sentiment-analysis-rnn.html
def eval_acc(net, data_iter, device=None):
    if device is None:
        device = list(net.parameters())[0].device
    num_right = 0.0
    n = 0
    TP = 0.0
    FP = 0.0
    FN = 0.0
    for X, y in data_iter:
        net.eval()
        y_hat = net(X.to(device))
        y = y.to(device)
        num_right += (y_hat.argmax(dim=1) == y).float().sum().cpu().item()
        n += y_hat.shape[0]
        for y_hat_row, y_row in zip(y_hat, y):
            if y_hat_row[1] > y_hat_row[0]:
                if y_row == 1:
                    TP += 1
                else:
                    FP += 1
            else:
                if y_row == 1:
                    FN += 1
        net.train()
    acc = num_right / n
    f_score = 2.0 / (2 + FP/TP + FN/TP)
    return acc, f_score

    
def train(train_iter, validation_iter, test_iter, net, loss_func, optimizer, device, num_epochs, model_path):
    net = net.to(device)
    num_batch = 0
    best_validation_acc = 0.0
    for epoch in range(num_epochs):
        train_loss, train_acc = 0.0, 0.0
        n = 0
        start_time = time.time()
        for X, y in train_iter:
            y_hat = net(X.to(device)).to(device)
            loss = loss_func(y_hat, y.to(device))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_acc += (y_hat.argmax(dim=1) == y.to(device)).float().sum().cpu().item()
            train_loss += loss.cpu().item()
            n += y.shape[0]
            num_batch += 1
        validation_acc, validation_f_score = eval_acc(net, validation_iter)
        print('epoch %d, loss %.4f, train_acc %.3f, validation_acc %.3f, validation_f_score, %.3f, time %.1f sec' % (epoch + 1, train_loss / num_batch, train_acc / n, validation_acc, validation_f_score, time.time() - start_time))
        if validation_acc > best_validation_acc:
            best_validation_acc = validation_acc
            torch.save(net.state_dict(), model_path)

        # temp_net = net
        # temp_net.load_state_dict(torch.load(model_path))
        # test_acc, test_f_score = eval_acc(temp_net, test_iter)
        # print('epoch %d, test_acc %.3f, test_f_score, %.3f' % (epoch + 1, test_acc, test_f_score))


def initialize_model(model, init_method=None):
    # 传入None表示使用默认初始化
    if init_method is None:
        return model
    if init_method == 'kaiming':
        init_func = init.kaiming_normal_
    elif init_method == 'xavier':
        init_func = init.xavier_normal_
    elif init_method == 'zeros':
        init_func = lambda x: init.zeros_(x)
    elif init_method == 'gaussian':
        init_func = lambda x: init.normal_(x, mean=0, std=0.01)
    elif init_method == 'orthogonal':
        init_func = init.orthogonal_
    else:
        raise ValueError("Invalid initialization method!")

    for name, param in model.named_parameters():
        if 'weight' in name:
            init_func(param)
        elif 'bias' in name:
            init.zeros_(param)

    return model