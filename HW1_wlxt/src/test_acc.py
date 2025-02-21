from tqdm import tqdm
import argparse
import math
import json
import argparse

total_gram_1 = 1000000000
alpha = 0.99
beta = 0.99
gamma = 0.8
theta = 0.95
LONG_DIS = 100
penalty = 1.0

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


# 第二步viterbi
# 要求pinyin_sentence是一个list，每个元素为str（即拼音）
def viterbi_gram_2(pinyin_sentence):

    last_one_nodes = [] # old_nodes是上一层的所有字
    last_one_dises = [] # 上一层每个字的dis
    last_one_sentences = [] # possible_sentences是“包含了上一层字”的候选句子，与last_one_nodes一一对应
    # 语料库中不常见的字 也不予考虑
    for char in vocab[pinyin_sentence[0]]:
        if char in gram_1:
            last_one_nodes.append(char)
    for node in last_one_nodes:
        last_one_dises.append(get_dis(1, [], node))
        last_one_sentences.append(node)


    for pinyin in pinyin_sentence[1:]:
        curr_nodes = [] # list里面是符合该拼音的所有字
        curr_dises = [] # 这一层每个字的dis，初始化为一个很大的double值
        curr_sentences = [] #初始化为一些""
        for char in vocab[pinyin]:
            if char in gram_1:
                curr_nodes.append(char)
        for _ in curr_nodes:
            curr_dises.append(LONG_DIS)
            curr_sentences.append("")


        # 注意：直接取for curr_idx, (curr_node, curr_dis) in enumerate(zip(curr_nodes, curr_dises)):并在内部修改curr_dis是不会更新到curr_dises的！！
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

    final_sentence = ""
    shortest_dis = LONG_DIS * 5
    for dis, sentence in zip(last_one_dises, last_one_sentences):
        if dis < shortest_dis:
            shortest_dis = dis
            final_sentence = sentence
    return final_sentence

def viterbi_gram_3(pinyin_sentence):

    last_one_nodes = [] # old_nodes是上一层的所有字
    last_one_dises = [] # 上一层每个字的dis
    last_one_sentences = [] # possible_sentences是“包含了上一层字”的候选句子，与last_one_nodes一一对应
    # 语料库中不常见的字 也不予考虑
    for char in vocab[pinyin_sentence[0]]:
        if char in gram_1:
            last_one_nodes.append(char)
    for node in last_one_nodes:
        last_one_dises.append(get_dis(1, [], node))
        last_one_sentences.append(node)

    for pinyin_idx, pinyin in enumerate(pinyin_sentence):
        if pinyin_idx == 0:
            continue
        curr_nodes = [] # list里面是符合该拼音的所有字
        curr_dises = [] # 这一层每个字的dis，初始化为一个很大的double值
        curr_sentences = [] #初始化为一些""
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

def viterbi_gram_4(pinyin_sentence):

    last_one_nodes = [] # old_nodes是上一层的所有字
    last_one_dises = [] # 上一层每个字的dis
    last_one_sentences = [] # possible_sentences是“包含了上一层字”的候选句子，与last_one_nodes一一对应
    # 语料库中不常见的字 也不予考虑
    for char in vocab[pinyin_sentence[0]]:
        if char in gram_1:
            last_one_nodes.append(char)
    for node in last_one_nodes:
        last_one_dises.append(get_dis(1, [], node))
        last_one_sentences.append(node)

    for pinyin_idx, pinyin in enumerate(pinyin_sentence):
        if pinyin_idx == 0:
            continue
        curr_nodes = [] # list里面是符合该拼音的所有字
        curr_dises = [] # 这一层每个字的dis，初始化为一个很大的double值
        curr_sentences = [] #初始化为一些""
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


num_total_sentence = 0
num_total_char = 0
num_correct_sentence = 0
num_correct_char = 0

std_input = []
std_output = []
my_output = []

def test_acc(viterbi, std_input_path = "./data/test/std_input.txt", std_output_path = "./data/test/std_output.txt", my_output_path = None):

    num_total_sentence = 0
    num_total_char = 0
    num_correct_sentence = 0
    num_correct_char = 0

    std_input = []
    std_output = []
    my_output = []

    with open(std_input_path, 'r') as in_file:
        for line in in_file:
            std_input.append(line.strip().split())
    with open(std_output_path, 'r', encoding='utf-8') as in_file:
        for line in in_file:
            std_output.append(str(line.strip()))

    for pinyin, std_char_sentence in tqdm(zip(std_input, std_output)):
        char_sentence = viterbi(pinyin)
        my_output.append(char_sentence)
        for char, std_char in zip(char_sentence, std_char_sentence):
            num_total_char += 1
            if char == std_char:
                num_correct_char += 1
        num_total_sentence += 1
        if char_sentence == std_char_sentence:
            num_correct_sentence += 1
    print(f"char_acc = {(float)(num_correct_char/num_total_char)}")
    print(f"sentence_acc = {(float)(num_correct_sentence)/num_total_sentence}")
    if my_output_path is not None:
        with open(my_output_path, 'w', encoding='utf-8') as file:
            for line in my_output:
                file.write(line)
                file.write("\n")
    return ((float)(num_correct_char/num_total_char), (float)(num_correct_sentence)/num_total_sentence)

def parse_args_test_acc():
    parser = argparse.ArgumentParser(description="cmd params for pinyin.py")
    parser.add_argument("--param_path", type=str, help="Path to the parameter file.")
    parser.add_argument("--std_input_path", type=str, help="Path to the std input file.")
    parser.add_argument("--std_output_path", type=str, help="Path to the std output file.")
    parser.add_argument("--model_output_path", type=str, help="Path to the model output file.")
    parser.add_argument("--param_gram", choices=["2gram", "3gram", "4gram"], help="N-gram used in the parameter file.")
    parser.add_argument("--model_gram", choices=["2gram", "3gram", "4gram"], help="N-gram model to use.")
    return parser.parse_args()


args = parse_args_test_acc()

gram_3 = {}
gram_4 = {}
if args.param_gram == "2gram":
    gram_1, gram_2, total_gram_1, vocab, common_chinese = load_param_2_gram(args.param_path)
if args.param_gram == "3gram":
    gram_1, gram_2, gram_3, total_gram_1, vocab, common_chinese = load_param_3_gram(args.param_path)
if args.param_gram == "4gram":
    gram_1, gram_2, gram_3, gram_4, total_gram_1, vocab, common_chinese = load_param_4_gram(args.param_path)
total_gram_1 = 1000000000

if args.model_gram == "2gram":
    test_acc(viterbi_gram_2, args.std_input_path, args.std_output_path, args.model_output_path)
if args.model_gram == "3gram":
    test_acc(viterbi_gram_3, args.std_input_path, args.std_output_path, args.model_output_path)
if args.model_gram == "4gram":
    test_acc(viterbi_gram_4, args.std_input_path, args.std_output_path, args.model_output_path)
