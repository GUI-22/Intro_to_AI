from collections import defaultdict
import re
from tqdm import tqdm
import json
import time
import argparse

# 加载拼音汉字表 和 常用汉字表
def load_vocab_and_common_chinese(pinyin_chinese_char_table, common_chinese_table):
    common_chinese = ""
    vocab = {}
    with open(common_chinese_table, 'r', encoding='gbk') as file:
        common_chinese = file.read().strip()
    with open(pinyin_chinese_char_table, 'r', encoding='gbk') as file:
        for line in file:
            line = line.strip().split(' ')
            pinyin = line[0]
            characters = [char for char in line[1:] if char in common_chinese]
            vocab[pinyin] = characters
    return (vocab, common_chinese)

# 计数，构造1、2、3、4元词典
def count_4gram(gram_1, gram_2, gram_3, gram_4, total_gram_1, common_chinese, sentence):
    # gram_1
    for char in sentence:
        if char in common_chinese:
            if char in gram_1:
                gram_1[char] += 1
            else:
                gram_1[char] = 1
            total_gram_1 += 1
    # gram_2
    for idx in range(len(sentence) - 1):
        if sentence[idx] in common_chinese and sentence[idx + 1] in common_chinese:
            phase = sentence[idx] + sentence[idx + 1]
            if phase in gram_2:
                gram_2[phase] += 1
            else:
                gram_2[phase] = 1
    # gram_3
    for idx in range(len(sentence) - 2):
        if sentence[idx] in common_chinese and sentence[idx + 1] in common_chinese and sentence[idx + 2] in common_chinese:
            phase = sentence[idx] + sentence[idx + 1] + sentence[idx + 2]
            if phase in gram_3:
                gram_3[phase] += 1
            else:
                gram_3[phase] = 1
    # gram_4
    for idx in range(len(sentence) - 3):
        if sentence[idx] in common_chinese and sentence[idx + 1] in common_chinese and sentence[idx + 2] in common_chinese and sentence[idx + 3] in common_chinese:
            phase = sentence[idx] + sentence[idx + 1] + sentence[idx + 2] + sentence[idx + 3]
            if phase in gram_4:
                gram_4[phase] += 1
            else:
                gram_4[phase] = 1
    return (gram_1, gram_2, gram_3, gram_4, total_gram_1)


# 计数，构造1、2、3元词典
def count_3gram(gram_1, gram_2, gram_3, total_gram_1, common_chinese, sentence):
    # gram_1
    for char in sentence:
        if char in common_chinese:
            if char in gram_1:
                gram_1[char] += 1
            else:
                gram_1[char] = 1
            total_gram_1 += 1
    # gram_2
    for idx in range(len(sentence) - 1):
        if sentence[idx] in common_chinese and sentence[idx + 1] in common_chinese:
            phase = sentence[idx] + sentence[idx + 1]
            if phase in gram_2:
                gram_2[phase] += 1
            else:
                gram_2[phase] = 1
    # gram_3
    for idx in range(len(sentence) - 2):
        if sentence[idx] in common_chinese and sentence[idx + 1] in common_chinese and sentence[idx + 2] in common_chinese:
            phase = sentence[idx] + sentence[idx + 1] + sentence[idx + 2]
            if phase in gram_3:
                gram_3[phase] += 1
            else:
                gram_3[phase] = 1
    return (gram_1, gram_2, gram_3, total_gram_1)


# 计数，构造1、2元词典
def count_2gram(gram_1, gram_2, total_gram_1, common_chinese, sentence):
    # gram_1
    for char in sentence:
        if char in common_chinese:
            if char in gram_1:
                gram_1[char] += 1
            else:
                gram_1[char] = 1
            total_gram_1 += 1
    # gram_2
    for idx in range(len(sentence) - 1):
        if sentence[idx] in common_chinese and sentence[idx + 1] in common_chinese:
            phase = sentence[idx] + sentence[idx + 1]
            if phase in gram_2:
                gram_2[phase] += 1
            else:
                gram_2[phase] = 1
    return (gram_1, gram_2, total_gram_1)

# 读取sina数据集
def load_corpus_sina_4gram(corpus_file, gram_1, gram_2, gram_3, gram_4, total_gram_1, common_chinese):
    # delete
    with open(corpus_file, 'r', encoding='gbk') as file:
        total_lines = sum(1 for _ in file)
    #delete end
    with open(corpus_file, 'r', encoding='gbk') as file:
        chinese_pattern = re.compile('[\u4e00-\u9fa5]+')
        # 注意 tqdm总是加载最外面
        for idx, line in tqdm(enumerate(file), total=total_lines):
            if not line:
                continue
            try:
                item = json.loads(line)
                sentence_list = chinese_pattern.findall(item["html"]) + chinese_pattern.findall(item["title"])
                # 防止sentence_list空
                if sentence_list:
                    for sentence in sentence_list:
                        gram_1, gram_2, gram_3, gram_4, total_gram_1 = count_4gram(gram_1, gram_2, gram_3, gram_4, total_gram_1, common_chinese, sentence)
            except:
                print(f"in file {corpus_file} in line {idx} goes wrong\n")
    return (gram_1, gram_2, gram_3, gram_4, total_gram_1)


def load_corpus_sina_3gram(corpus_file, gram_1, gram_2, gram_3, total_gram_1, common_chinese):
    # delete
    with open(corpus_file, 'r', encoding='gbk') as file:
        total_lines = sum(1 for _ in file)
    #delete end
    with open(corpus_file, 'r', encoding='gbk') as file:
        chinese_pattern = re.compile('[\u4e00-\u9fa5]+')
        # 注意 tqdm总是加载最外面
        for idx, line in tqdm(enumerate(file), total=total_lines):
            if not line:
                continue
            try:
                item = json.loads(line)
                sentence_list = chinese_pattern.findall(item["html"]) + chinese_pattern.findall(item["title"])
                # 防止sentence_list空
                if sentence_list:
                    for sentence in sentence_list:
                        gram_1, gram_2, gram_3, total_gram_1 = count_3gram(gram_1, gram_2, gram_3, total_gram_1, common_chinese, sentence)
            except:
                print(f"in file {corpus_file} in line {idx} goes wrong\n")
    return (gram_1, gram_2, gram_3, total_gram_1)

def load_corpus_sina_2gram(corpus_file, gram_1, gram_2, total_gram_1, common_chinese):
    # delete
    with open(corpus_file, 'r', encoding='gbk') as file:
        total_lines = sum(1 for _ in file)
    #delete end
    with open(corpus_file, 'r', encoding='gbk') as file:
        chinese_pattern = re.compile('[\u4e00-\u9fa5]+')
        # 注意 tqdm总是加载最外面
        for idx, line in tqdm(enumerate(file), total=total_lines):
            if not line:
                continue
            try:
                item = json.loads(line)
                sentence_list = chinese_pattern.findall(item["html"]) + chinese_pattern.findall(item["title"])
                # 防止sentence_list空
                if sentence_list:
                    for sentence in sentence_list:
                        gram_1, gram_2, total_gram_1 = count_2gram(gram_1, gram_2, total_gram_1, common_chinese, sentence)
            except:
                print(f"in file {corpus_file} in line {idx} goes wrong\n")
    return (gram_1, gram_2, total_gram_1)

#读取SMP数据集
def load_corpus_SMP_4gram(corpus_file, gram_1, gram_2, gram_3, gram_4, total_gram_1, common_chinese):
    # delete
    with open(corpus_file, 'r', encoding='gbk') as file:
        total_lines = sum(1 for _ in file)
    #delete end
    with open(corpus_file, 'r', encoding='gbk') as file:
        chinese_pattern = re.compile('[\u4e00-\u9fa5]+')
        # 注意 tqdm总是加载最外面
        for idx, line in tqdm(enumerate(file), total=total_lines):
            if not line:
                continue
            try:
                item = line
                sentence_list = chinese_pattern.findall(item)
                # 防止sentence_list空
                if sentence_list:
                    for sentence in sentence_list:
                        gram_1, gram_2, gram_3, gram_4, total_gram_1 = count_4gram(gram_1, gram_2, gram_3, gram_4, total_gram_1, common_chinese, sentence)
            except:
                print(f"in file {corpus_file} in line {idx} goes wrong\n")
    return (gram_1, gram_2, gram_3, gram_4, total_gram_1)


def load_corpus_SMP_3gram(corpus_file, gram_1, gram_2, gram_3, total_gram_1, common_chinese):
    # delete
    with open(corpus_file, 'r', encoding='gbk') as file:
        total_lines = sum(1 for _ in file)
    #delete end
    with open(corpus_file, 'r', encoding='gbk') as file:
        chinese_pattern = re.compile('[\u4e00-\u9fa5]+')
        # 注意 tqdm总是加载最外面
        for idx, line in tqdm(enumerate(file), total=total_lines):
            if not line:
                continue
            try:
                item = line
                sentence_list = chinese_pattern.findall(item)
                # 防止sentence_list空
                if sentence_list:
                    for sentence in sentence_list:
                        gram_1, gram_2, gram_3, total_gram_1 = count_3gram(gram_1, gram_2, gram_3, total_gram_1, common_chinese, sentence)
            except:
                print(f"in file {corpus_file} in line {idx} goes wrong\n")
    return (gram_1, gram_2, gram_3, total_gram_1)


def load_corpus_SMP_2gram(corpus_file, gram_1, gram_2, total_gram_1, common_chinese):
    # delete
    with open(corpus_file, 'r', encoding='gbk') as file:
        total_lines = sum(1 for _ in file)
    #delete end
    with open(corpus_file, 'r', encoding='gbk') as file:
        chinese_pattern = re.compile('[\u4e00-\u9fa5]+')
        # 注意 tqdm总是加载最外面
        for idx, line in tqdm(enumerate(file), total=total_lines):
            if not line:
                continue
            try:
                item = line
                sentence_list = chinese_pattern.findall(item)
                # 防止sentence_list空
                if sentence_list:
                    for sentence in sentence_list:
                        gram_1, gram_2, total_gram_1 = count_2gram(gram_1, gram_2, total_gram_1, common_chinese, sentence)
            except:
                print(f"in file {corpus_file} in line {idx} goes wrong\n")
    return (gram_1, gram_2, total_gram_1)

# 读取百度百科数据集
def load_corpus_baidu_bike_3gram(corpus_file, gram_1, gram_2, gram_3, total_gram_1, common_chinese):
    # delete
    with open(corpus_file, 'r', encoding='utf-8') as file:
        total_lines = sum(1 for _ in file)
    #delete end
    with open(corpus_file, 'r', encoding='utf-8') as file:
        chinese_pattern = re.compile('[\u4e00-\u9fa5]+')
        # 注意 tqdm总是加载最外面
        for idx, line in tqdm(enumerate(file), total=total_lines):
            if not line:
                continue
            try:
                item = line
                sentence_list = chinese_pattern.findall(item)
                # 防止sentence_list空
                if sentence_list:
                    for sentence in sentence_list:
                        gram_1, gram_2, gram_3, total_gram_1 = count_3gram(gram_1, gram_2, gram_3, total_gram_1, common_chinese, sentence)
            except:
                print(f"in file {corpus_file} in line {idx} goes wrong\n")
    return (gram_1, gram_2, gram_3, total_gram_1)

# 读取百度问答数据集
def load_corpus_wenda_3gram(corpus_file, gram_1, gram_2, gram_3, total_gram_1, common_chinese):
    # delete
    with open(corpus_file, 'r') as file:
        total_lines = sum(1 for _ in file)
    #delete end
    with open(corpus_file, 'r') as file:
        chinese_pattern = re.compile('[\u4e00-\u9fa5]+')
        # 注意 tqdm总是加载最外面
        for idx, line in tqdm(enumerate(file), total=total_lines):
            if not line:
                continue
            try:
                item = json.loads(line)
                sentence_list = chinese_pattern.findall(item["title"]) + chinese_pattern.findall(item["answer"])
                # 防止sentence_list空
                if sentence_list:
                    for sentence in sentence_list:
                        gram_1, gram_2, gram_3, total_gram_1 = count_3gram(gram_1, gram_2, gram_3, total_gram_1, common_chinese, sentence)
            except:
                print(f"in file {corpus_file} in line {idx} goes wrong\n")
    return (gram_1, gram_2, gram_3, total_gram_1)

def load_corpus_wenda_4gram(corpus_file, gram_1, gram_2, gram_3, gram_4, total_gram_1, common_chinese):
    # delete
    with open(corpus_file, 'r') as file:
        total_lines = sum(1 for _ in file)
    #delete end
    with open(corpus_file, 'r') as file:
        chinese_pattern = re.compile('[\u4e00-\u9fa5]+')
        # 注意 tqdm总是加载最外面
        for idx, line in tqdm(enumerate(file), total=total_lines):
            if not line:
                continue
            try:
                item = json.loads(line)
                sentence_list = chinese_pattern.findall(item["title"]) + chinese_pattern.findall(item["answer"])
                # 防止sentence_list空
                if sentence_list:
                    for sentence in sentence_list:
                        gram_1, gram_2, gram_3, gram_4, total_gram_1 = count_4gram(gram_1, gram_2, gram_3, gram_4, total_gram_1, common_chinese, sentence)
            except:
                print(f"in file {corpus_file} in line {idx} goes wrong\n")
    return (gram_1, gram_2, gram_3, gram_4, total_gram_1)

def load_corpus_wenda_2gram(corpus_file, gram_1, gram_2, total_gram_1, common_chinese):
    # delete
    with open(corpus_file, 'r') as file:
        total_lines = sum(1 for _ in file)
    #delete end
    with open(corpus_file, 'r') as file:
        chinese_pattern = re.compile('[\u4e00-\u9fa5]+')
        # 注意 tqdm总是加载最外面
        for idx, line in tqdm(enumerate(file), total=total_lines):
            if not line:
                continue
            try:
                item = json.loads(line)
                sentence_list = chinese_pattern.findall(item["title"]) + chinese_pattern.findall(item["answer"])
                # 防止sentence_list空
                if sentence_list:
                    for sentence in sentence_list:
                        gram_1, gram_2, total_gram_1 = count_2gram(gram_1, gram_2, total_gram_1, common_chinese, sentence)
            except:
                print(f"in file {corpus_file} in line {idx} goes wrong\n")
    return (gram_1, gram_2, total_gram_1)

#存储参数为json格式
def save_param_4gram(save_path, gram_1, gram_2, gram_3, gram_4, total_gram_1, vocab, common_chinese):
    data = {}
    data["gram_1"] = gram_1
    data["gram_2"] = gram_2
    data["gram_3"] = gram_3
    data["gram_4"] = gram_4
    data["total_gram_1"] = total_gram_1
    data["vocab"] = vocab
    data["common_chinese"] = common_chinese
    with open(save_path, "w", encoding="gbk") as file:
        json.dump(data, file, ensure_ascii=False, indent=4)

def save_param_3gram(save_path, gram_1, gram_2, gram_3, total_gram_1, vocab, common_chinese):
    data = {}
    data["gram_1"] = gram_1
    data["gram_2"] = gram_2
    data["gram_3"] = gram_3
    data["total_gram_1"] = total_gram_1
    data["vocab"] = vocab
    data["common_chinese"] = common_chinese
    with open(save_path, "w", encoding="gbk") as file:
        json.dump(data, file, ensure_ascii=False, indent=4)

def save_param_2gram(save_path, gram_1, gram_2, total_gram_1, vocab, common_chinese):
    data = {}
    data["gram_1"] = gram_1
    data["gram_2"] = gram_2
    data["total_gram_1"] = total_gram_1
    data["vocab"] = vocab
    data["common_chinese"] = common_chinese
    with open(save_path, "w", encoding="gbk") as file:
        json.dump(data, file, ensure_ascii=False, indent=4)

# 读取拼音汉字表和常用汉字表
def train_vocab_and_common_chinese():
    common_chinese = "" # 第一步 传入参数 统计1-gram 2-grams 3-grams（每个“组合”出现的次数）
    vocab = {} #每一项为 “拼音（字符串）：所有对应汉字（list）”
    pinyin_chinese_char_table = "../corpus/pinyin_chinese_char_table.txt"
    common_chinese_table = "../corpus/common_chinese_table.txt"
    vocab, common_chinese = load_vocab_and_common_chinese(pinyin_chinese_char_table, common_chinese_table)
    return vocab, common_chinese

# 训练1、2元参数
def train_2gram(train_sina, train_SMP, train_wenda, specified_save_path=None):
    gram_1 = {} 
    gram_2 = {}
    gram_3 = {} 
    gram_4 = {}
    total_gram_1 = 0
    save_path = "../saved_param/saved_param_"
    vocab, common_chinese = train_vocab_and_common_chinese()

    if train_sina == True:
        save_path = save_path + "sina_"
        i = 4
        while (i < 10):
            corpus_file = f"../corpus/sina_corpus/2016-0{i}.txt"
            gram_1, gram_2, total_gram_1 = load_corpus_sina_2gram(corpus_file, gram_1, gram_2, total_gram_1, common_chinese)
            i += 1
        i = 10
        while (i < 12):
            corpus_file = f"../corpus/sina_corpus/2016-{i}.txt"
            gram_1, gram_2, total_gram_1 = load_corpus_sina_2gram(corpus_file, gram_1, gram_2, total_gram_1, common_chinese)
            i += 1

    if train_SMP == True:
        save_path = save_path + "SMP_"
        corpus_file = f"../corpus/usual_train_new.txt"
        gram_1, gram_2, total_gram_1 = load_corpus_SMP_2gram(corpus_file, gram_1, gram_2, total_gram_1, common_chinese)

    if train_wenda == True:
        save_path = save_path + "wenda_"
        corpus_file = f"../corpus/baidu_corpus/baike_qa_train.json"
        gram_1, gram_2, total_gram_1 = load_corpus_wenda_2gram(corpus_file, gram_1, gram_2, total_gram_1, common_chinese)
    
    save_path = save_path + "2gram.json"
    if specified_save_path is not None:
        save_path = specified_save_path
    save_param_2gram(save_path, gram_1, gram_2, total_gram_1, vocab, common_chinese)

#训练1、2、3元参数
def train_3gram(train_sina, train_SMP, train_wenda, specified_save_path=None):
    gram_1 = {} 
    gram_2 = {}
    gram_3 = {} 
    gram_4 = {}
    total_gram_1 = 0
    save_path = "../saved_param/saved_param_"
    vocab, common_chinese = train_vocab_and_common_chinese()

    if train_sina == True:
        save_path = save_path + "sina_"
        i = 4
        while (i < 10):
            corpus_file = f"../corpus/sina_corpus/2016-0{i}.txt"
            gram_1, gram_2, gram_3, total_gram_1 = load_corpus_sina_3gram(corpus_file, gram_1, gram_2, gram_3, total_gram_1, common_chinese)
            i += 1
        i = 10
        while (i < 12):
            corpus_file = f"../corpus/sina_corpus/2016-{i}.txt"
            gram_1, gram_2, gram_3, total_gram_1 = load_corpus_sina_3gram(corpus_file, gram_1, gram_2, gram_3, total_gram_1, common_chinese)
            i += 1

    if train_SMP == True:
        save_path = save_path + "SMP_"
        corpus_file = f"../corpus/usual_train_new.txt"
        gram_1, gram_2, gram_3, total_gram_1 = load_corpus_SMP_3gram(corpus_file, gram_1, gram_2, gram_3, total_gram_1, common_chinese)

    if train_wenda == True:
        save_path = save_path + "wenda_"
        corpus_file = f"../corpus/baidu_corpus/baike_qa_train.json"
        gram_1, gram_2, gram_3, total_gram_1 = load_corpus_wenda_3gram(corpus_file, gram_1, gram_2, gram_3, total_gram_1, common_chinese)
    
    save_path = save_path + "3gram.json"
    if specified_save_path is not None:
        save_path = specified_save_path
    save_param_3gram(save_path, gram_1, gram_2, gram_3, total_gram_1, vocab, common_chinese)

#训练1、2、3、4元参数
def train_4gram(train_sina, train_SMP, train_wenda, specified_save_path=None):
    gram_1 = {} 
    gram_2 = {}
    gram_3 = {} 
    gram_4 = {}
    total_gram_1 = 0
    save_path = "../saved_param/saved_param_"
    vocab, common_chinese = train_vocab_and_common_chinese()

    if train_sina == True:
        save_path = save_path + "sina_"
        i = 4
        while (i < 10):
            corpus_file = f"../corpus/sina_corpus/2016-0{i}.txt"
            gram_1, gram_2, gram_3, gram_4, total_gram_1 = load_corpus_sina_4gram(corpus_file, gram_1, gram_2, gram_3, gram_4, total_gram_1, common_chinese)
            i += 1
        i = 10
        while (i < 12):
            corpus_file = f"../corpus/sina_corpus/2016-{i}.txt"
            gram_1, gram_2, gram_3, gram_4, total_gram_1 = load_corpus_sina_4gram(corpus_file, gram_1, gram_2, gram_3, gram_4, total_gram_1, common_chinese)
            i += 1

    if train_SMP == True:
        save_path = save_path + "SMP_"
        corpus_file = f"../corpus/usual_train_new.txt"
        gram_1, gram_2, gram_3, gram_4, total_gram_1 = load_corpus_SMP_4gram(corpus_file, gram_1, gram_2, gram_3, gram_4, total_gram_1, common_chinese)

    if train_wenda == True:
        save_path = save_path + "wenda_"
        corpus_file = f"../corpus/baidu_corpus/baike_qa_train.json"
        gram_1, gram_2, gram_3, gram_4, total_gram_1 = load_corpus_wenda_4gram(corpus_file, gram_1, gram_2, gram_3, gram_4, total_gram_1, common_chinese)
    
    save_path = save_path + "4gram.json"
    if specified_save_path is not None:
        save_path = specified_save_path
    save_param_4gram(save_path, gram_1, gram_2, gram_3, gram_4, total_gram_1, vocab, common_chinese)

#处理命令行参数
parser = argparse.ArgumentParser(description="Train n-gram model on specified corpus.")
parser.add_argument("--corpus", nargs="+", choices=["sina", "SMP", "wenda"], help="Corpora to train the model on.")
parser.add_argument("--gram", choices=["2gram", "3gram", "4gram"], help="N-gram model to train.")
parser.add_argument("--save_path", type=str, help="path to save param")
args = parser.parse_args()

train_sina = False
train_SMP = False
train_wenda = False
if 'sina' in args.corpus:
    train_sina = True
if 'SMP' in args.corpus:
    train_SMP = True
if 'wenda' in args.corpus:
    train_wenda = True
    
#训练
start_time = time.time()
if args.gram == "2gram":
    train_2gram(train_sina, train_SMP, train_wenda, args.save_path)

if args.gram == "3gram":
    train_3gram(train_sina, train_SMP, train_wenda, args.save_path)

if args.gram == "4gram":
    train_4gram(train_sina, train_SMP, train_wenda, args.save_path)
finish_train_time = time.time()
print(f"finish train in {finish_train_time - start_time} seconds")

