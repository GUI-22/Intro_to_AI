from transformers import BertTokenizer, BertForSequenceClassification
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch.optim import AdamW
import torch
from torch import nn
from transformers import AutoTokenizer, AutoModelForMaskedLM
import time
import random


# 设置超参数
max_len = 128       #句子truncate pad后长度
batch_size = 64     #batch_size
num_epochs = 10     #总共训练轮数

# Step1: 处理数据
def read_file(path, is_train=False):
    items = []
    with open(path, "r", encoding='utf-8') as in_file:
        for line in in_file:
            items.append(line)
    if is_train:
      random.shuffle(items)
    labels = []
    sentences = []
    for item in items:
        labels.append(int(item[0]))
        sentences.append(item[1:])
    return torch.tensor(labels, dtype=torch.long), sentences

train_labels, train_sentences = read_file("./Dataset/train.txt", is_train=True)
test_labels, test_sentences = read_file("./Dataset/test.txt")
tokenizer = AutoTokenizer.from_pretrained("./bert_base_Chinese")

def data_loader(labels, sentences, tokenizer=tokenizer, max_len=max_len, batch_size=batch_size):
    tokenized_sentences = tokenizer(sentences, padding=True, truncation=True, max_length=max_len, return_tensors='pt')
    train_data = list(zip(tokenized_sentences['input_ids'], tokenized_sentences['attention_mask'], labels))
    return DataLoader(train_data, batch_size=batch_size)

train_iter = data_loader(train_labels, train_sentences)
test_iter = data_loader(test_labels, test_sentences)

# Step2: 加载、训练model

def eval_acc(net, data_iter, device=None):
    if device is None:
        device = list(net.parameters())[0].device
    num_right = 0.0
    n = 0
    TP = 0.0
    FP = 0.0
    FN = 0.0
    for input_ids, attention_mask, y in data_iter:
        net.eval()
        attention_mask = attention_mask.to(device)
        y = y.to(device)
        y_hat = net(input_ids.to(device), attention_mask=attention_mask).logits
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

    
def train(train_iter, validation_iter, net, loss_func, optimizer, device, num_epochs, model_path):
    net = net.to(device)
    num_batch = 0
    best_validation_acc = 0.0
    for epoch in range(num_epochs):
        train_loss, train_acc = 0.0, 0.0
        n = 0
        start_time = time.time()
        total_batches = len(train_iter)
        for batch_index, (input_ids, attention_mask, y) in enumerate(train_iter):
            attention_mask = attention_mask.to(device) 
            y = y.to(device)            
            y_hat = net(input_ids.to(device), attention_mask=attention_mask).logits.to(device)
            loss = loss_func(y_hat, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_acc += (y_hat.argmax(dim=1) == y.to(device)).float().sum().cpu().item()
            train_loss += loss.cpu().item()
            n += y.shape[0]
            num_batch += 1
            if num_batch % 20 == 0:
                progress = (batch_index + 1) / total_batches * 100
                print(f"Progress: {progress:.2f}%")
        validation_acc, f_score = eval_acc(net, validation_iter)
        print('epoch %d, loss %.4f, train_acc %.3f, validation_acc %.3f, f_score, %.3f, time %.1f sec'
              % (epoch + 1, train_loss / num_batch, train_acc / n, validation_acc, f_score, time.time() - start_time))
        if validation_acc > best_validation_acc:
            best_validation_acc = validation_acc
            torch.save(net.state_dict(), model_path)

       
model = BertForSequenceClassification.from_pretrained(
    "./bert_base_Chinese", # 使用12层的BERT模型
    num_labels = 2, # 二分类任务（比如情感分析）
    output_attentions = False, # 模型是否返回注意力权重
    output_hidden_states = False, # 模型是否返回所有隐藏状态
)

model_path = "./saved_path/saved_bert.pth"

optimizer = AdamW(model.parameters(), lr=3e-5)
loss = nn.CrossEntropyLoss()
device = torch.device("cuda")
train(train_iter, test_iter, model, loss, optimizer, device, num_epochs, model_path)

model.load_state_dict(torch.load(model_path))
test_acc, test_f_score = eval_acc(model, test_iter, device)
print(f"in test dataset, acc = {test_acc}, f_score = {test_f_score}")