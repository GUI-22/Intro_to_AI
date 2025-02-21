import math
import sys
import json
import argparse

# 可调的参数
total_gram_1 = 1000000000
alpha = 0.99
beta = 0.99
gamma = 0.8
theta = 0.95
LONG_DIS = 100
penalty = 1.0

# 读取参数文件
def load_param_4_gram(save_path):
    with open(save_path, "r", encoding="gbk") as file:
        data = json.load(file)
    return (data["gram_1"], data["gram_2"], data["gram_3"], data["gram_4"], data["total_gram_1"], data["vocab"], data["common_chinese"])

def load_param_3_gram(save_path):
    with open(save_path, "r", encoding="gbk") as file:
        data = json.load(file)
    return (data["gram_1"], data["gram_2"], data["gram_3"], data["total_gram_1"], data["vocab"], data["common_chinese"])

def load_param_2_gram(save_path):
    with open(save_path, "r", encoding="gbk") as file:
        data = json.load(file)
    return (data["gram_1"], data["gram_2"], data["total_gram_1"], data["vocab"], data["common_chinese"])

#计算viterbi算法中的“距离”
def get_dis(num_gram, parameters, nodes):
    if num_gram == 1:
        if nodes not in gram_1:
            return LONG_DIS
        p = (float)(gram_1[nodes] / total_gram_1)
    elif num_gram == 2:
        if nodes[0] not in gram_1 or nodes not in gram_2:
            # return LONG_DIS
            return get_dis(1, [], nodes[1])
        p = parameters[0] * (float)(gram_2[nodes] / gram_1[nodes[0]]) + (1 - parameters[0]) * (float)(gram_1[nodes[1]] / total_gram_1)
    elif num_gram == 3:
        if nodes[0:2] not in gram_2 or nodes[1:3] not in gram_2 or nodes not in gram_3:
            return get_dis(2, [alpha], nodes[1:3])
        q = parameters[0] * (float)(gram_2[nodes[1:3]] / gram_1[nodes[1]]) + (1 - parameters[0]) * (float)(gram_1[nodes[2]] / total_gram_1)
        p = parameters[1] * (float)(gram_3[nodes] / gram_2[nodes[0:2]]) + (1 - parameters[1]) * q
    elif num_gram == 4:
        if nodes[0:3] not in gram_3 or nodes not in gram_4:
            return penalty * get_dis(3, [beta, gamma], nodes[1:4])
        r = parameters[0] * (float)(gram_2[nodes[2:4]] / gram_1[nodes[2]]) + (1 - parameters[0]) * (float)(gram_1[nodes[3]] / total_gram_1)
        q = parameters[1] * (float)(gram_3[nodes[1:4]] / gram_2[nodes[1:3]]) + (1 - parameters[1]) * r
        p = parameters[2] * (float)(gram_4[nodes] / gram_3[nodes[0:3]]) + (1 - parameters[2]) * q
    return - math.log(p)

#2元viterbi算法
def viterbi_gram_2(pinyin_sentence):

    last_one_nodes = [] # old_nodes是上一层的所有字
    last_one_dises = [] # 上一层每个字的dis
    last_one_sentences = [] # possible_sentences是“包含了上一层字”的候选句子，与last_one_nodes一一对应
    for char in vocab[pinyin_sentence[0]]:
        if char in gram_1:
            last_one_nodes.append(char)
    for node in last_one_nodes:
        last_one_dises.append(get_dis(1, [], node))
        last_one_sentences.append(node)


    for pinyin in pinyin_sentence[1:]:
        curr_nodes = [] # list里面是符合该拼音的所有字
        curr_dises = [] # 这一层每个字的dis，初始化为一个很大的double值
        curr_sentences = [] #初始化为一些空串
        for char in vocab[pinyin]:
            if char in gram_1:
                curr_nodes.append(char)
        for _ in curr_nodes:
            curr_dises.append(LONG_DIS)
            curr_sentences.append("")

        for curr_idx, curr_node in enumerate(curr_nodes):
            candidate_last_one_idx = 0
            for last_one_idx, (last_one_node, last_one_dis) in enumerate(zip(last_one_nodes, last_one_dises)):
                temp_dis = last_one_dis + get_dis(2, [alpha], last_one_node + curr_node)

                if temp_dis < curr_dises[curr_idx]:
                    curr_dises[curr_idx] = temp_dis
                    candidate_last_one_idx = last_one_idx
            curr_sentences[curr_idx] = last_one_sentences[candidate_last_one_idx] + curr_node

        last_one_nodes = curr_nodes
        last_one_dises = curr_dises
        last_one_sentences = curr_sentences

    #找最终返回的句子
    final_sentence = ""
    shortest_dis = LONG_DIS * 5
    for dis, sentence in zip(last_one_dises, last_one_sentences):
        if dis < shortest_dis:
            shortest_dis = dis
            final_sentence = sentence
    return final_sentence

# 3元viterbi算法
def viterbi_gram_3(pinyin_sentence):

    last_one_nodes = [] # old_nodes是上一层的所有字
    last_one_dises = [] # 上一层每个字的dis
    last_one_sentences = [] # possible_sentences是“包含了上一层字”的候选句子，与last_one_nodes一一对应
    for char in vocab[pinyin_sentence[0]]:
        if char in gram_1:
            last_one_nodes.append(char)
    for node in last_one_nodes:
        last_one_dises.append(get_dis(1, [], node))
        last_one_sentences.append(node)

    for pinyin_idx, pinyin in enumerate(pinyin_sentence):
        if pinyin_idx == 0:
            continue
        curr_nodes = [] 
        curr_dises = [] 
        curr_sentences = [] 
        for char in vocab[pinyin]:
            if char in gram_1:
                curr_nodes.append(char)
        for _ in curr_nodes:
            curr_dises.append(LONG_DIS)
            curr_sentences.append("")

        for curr_idx, curr_node in enumerate(curr_nodes):
            candidate_last_one_idx = 0
            for last_one_idx, (last_one_node, last_one_dis) in enumerate(zip(last_one_nodes, last_one_dises)):
                if pinyin_idx == 1:
                    temp_dis = last_one_dis + get_dis(2, [alpha], last_one_sentences[last_one_idx][-1:] + curr_node) 
                else:
                    temp_dis = last_one_dis + get_dis(3, [beta, gamma], last_one_sentences[last_one_idx][-2:] + curr_node) 
                if temp_dis < curr_dises[curr_idx]:
                    curr_dises[curr_idx] = temp_dis
                    candidate_last_one_idx = last_one_idx
            curr_sentences[curr_idx] = last_one_sentences[candidate_last_one_idx] + curr_node

        last_one_nodes = curr_nodes
        last_one_dises = curr_dises
        last_one_sentences = curr_sentences

    final_sentence = ""
    shortest_dis = LONG_DIS * 5
    for dis, sentence in zip(last_one_dises, last_one_sentences):
        if dis < shortest_dis:
            shortest_dis = dis
            final_sentence = sentence
    return final_sentence

#4元viterbi算法
def viterbi_gram_4(pinyin_sentence):

    last_one_nodes = [] 
    last_one_dises = [] 
    last_one_sentences = []
    for char in vocab[pinyin_sentence[0]]:
        if char in gram_1:
            last_one_nodes.append(char)
    for node in last_one_nodes:
        last_one_dises.append(get_dis(1, [], node))
        last_one_sentences.append(node)

    for pinyin_idx, pinyin in enumerate(pinyin_sentence):
        if pinyin_idx == 0:
            continue
        curr_nodes = [] 
        curr_dises = [] 
        curr_sentences = [] 
        for char in vocab[pinyin]:
            if char in gram_1:
                curr_nodes.append(char)
        for _ in curr_nodes:
            curr_dises.append(LONG_DIS)
            curr_sentences.append("")

        for curr_idx, curr_node in enumerate(curr_nodes):
            candidate_last_one_idx = 0
            for last_one_idx, (last_one_node, last_one_dis) in enumerate(zip(last_one_nodes, last_one_dises)):
                if pinyin_idx == 1:
                    temp_dis = last_one_dis + get_dis(2, [alpha], last_one_sentences[last_one_idx][-1:] + curr_node) 
                elif pinyin_idx == 2:
                    temp_dis = last_one_dis + get_dis(3, [beta, gamma], last_one_sentences[last_one_idx][-2:] + curr_node) 
                else:   
                    temp_dis = last_one_dis + get_dis(4, [beta, gamma, theta], last_one_sentences[last_one_idx][-3:] + curr_node)
                if temp_dis < curr_dises[curr_idx]:
                    curr_dises[curr_idx] = temp_dis
                    candidate_last_one_idx = last_one_idx
            curr_sentences[curr_idx] = last_one_sentences[candidate_last_one_idx] + curr_node

        last_one_nodes = curr_nodes
        last_one_dises = curr_dises
        last_one_sentences = curr_sentences

    final_sentence = ""
    shortest_dis = LONG_DIS * 5
    for dis, sentence in zip(last_one_dises, last_one_sentences):
        if dis < shortest_dis:
            shortest_dis = dis
            final_sentence = sentence
    return final_sentence

# 使用文件交互
def interact(viterbi):
    # 从标准输入读取数据
    input_data = []
    for line in sys.stdin:
        line = line.strip()  # 去除行首行尾的空白字符
        if line:
            pinyin_list = line.split()  # 用空格分隔拼音串
            input_data.append(pinyin_list)

    for pinyin_list in input_data:
        result = viterbi(pinyin_list)
        print(result)

#使用命令行交互
def interact_cmd(viterbi, times=10):
    for i in range(times):
        input_pinyin = input()
        pinyin_sentence = input_pinyin.strip().split()
        print(viterbi(pinyin_sentence))

# 处理传入的参数
def parse_args():
    parser = argparse.ArgumentParser(description="cmd params for pinyin.py")
    parser.add_argument("--param_path", type=str, help="Path to the parameter file.")
    parser.add_argument("--param_gram", choices=["2gram", "3gram", "4gram"], help="N-gram used in the parameter file.")
    parser.add_argument("--model_gram", choices=["2gram", "3gram", "4gram"], help="N-gram model to use.")
    parser.add_argument("--use_file", choices=["True", "False"], help="Whether to use file to read in or not")
    parser.add_argument("--times", type=int, help="Interact how many times in cmd.")
    return parser.parse_args()

args = parse_args()

# gram_n为n元词典
gram_3 = {}
gram_4 = {}
if args.param_gram == "2gram":
    gram_1, gram_2, total_gram_1, vocab, common_chinese = load_param_2_gram(args.param_path)
if args.param_gram == "3gram":
    gram_1, gram_2, gram_3, total_gram_1, vocab, common_chinese = load_param_3_gram(args.param_path)
if args.param_gram == "4gram":
    gram_1, gram_2, gram_3, gram_4, total_gram_1, vocab, common_chinese = load_param_4_gram(args.param_path)
total_gram_1 = 1000000000

# 命令行交互
if args.use_file == "False":
    if args.model_gram == "2gram":
        interact_cmd(viterbi_gram_2, args.times)
    if args.model_gram == "3gram":
        interact_cmd(viterbi_gram_3, args.times)
    if args.model_gram == "4gram":
        interact_cmd(viterbi_gram_4, args.times)

#文件交互
if args.use_file == "True":
    if args.model_gram == "2gram":
        interact(viterbi_gram_2)
    if args.model_gram == "3gram":
        interact(viterbi_gram_3)
    if args.model_gram == "4gram":
        interact(viterbi_gram_4)


