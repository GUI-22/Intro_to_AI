## 拼音输入法 README

- **注意：请前往清华云盘或github下载该总文件夹，里面有运行程序必要的参数文件，以及训练必要的语料文件**
- 云盘地址：https://cloud.tsinghua.edu.cn/library/44709e4c-df5c-4e46-ae4f-30430bf764b5/pinyin_input/
- github地址：https://github.com/EmiyaArcher2333/pinyin_input
- **注意：运行以下文件时，请保证在src目录下，请严格按照下面规定的参数格式来运行（推荐直接粘贴命令来运行）**

### 一、文件组织

1、**实际运行时文件组织**

```
├── README.md
├── corpus
│   ├── baidu_corpus
│   ├── common_chinese_table.txt
│   ├── pinyin_chinese_char_table.txt
│   ├── sina_corpus
│   └── usual_train_new.txt
├── data
│   ├── input.txt
│   └── output.txt
├── saved_param
│   ├── saved_param_only_wenda_3gram.json
│   ├── saved_param_sina_and_SMP_3gram.json
│   ├── saved_param_sina_and_SMP_4gram.json
│   └── saved_param_sina_and_SMP_and_wenda_3gram.json
├── src
│   ├── pinyin.py
│   ├── test_acc.py
│   └── train.py
└── test
    ├── std_input.txt
    └── std_output.txt
```

2、**上交到网络学堂的文件组织**

```
├── README.md
├── data
│   ├── input.txt
│   └── output.txt
└── src
    ├── pinyin.py
    ├── test_acc.py
    └── train.py
```

### 二、参数介绍

#### train.py

1、本文件用于从语料库中训练参数

- corpus参数：选择使用哪些语料库来训练，可以选择以下中的1个或多个

```bash
--corpus sina wenda
```

- gram参数：选择训练的参数包含“1元、2元”或“1、2、3元”或“1、2、3、4元”信息，分别对应2gram/3gram/4gram

```bash
--gram 3gram
```

- save_path参数：请手动指定存储参数的路径

```bash
--save_path ../saved_param/saved_param_sina_and_wenda_3gram.json
```

2、推荐的用例

```bash
python train.py --corpus sina wenda --gram 3gram --save_path ../saved_param/saved_param_sina_and_wenda_3gram.json

python train.py --corpus sina SMP wenda --gram 3gram --save_path ../saved_param/saved_param_sina_and_SMP_and_wenda_3gram.json

python train.py --corpus sina SMP --gram 3gram --save_path ../saved_param/saved_param_sina_and_SMP_3gram.json

python train.py --corpus sina SMP --gram 4gram --save_path ../saved_param/saved_param_sina_and_SMP_4gram.json

python train.py --corpus SMP --gram 4gram --save_path ../saved_param/saved_param_SMP_4gram.json
```

#### pinyin.py

- param_path参数：指定使用的参数路径

```bash
--param_path ../saved_param/saved_param_sina_and_SMP_and_wenda_3gram.json
```

- param_gram参数:参数文件里面的参数存储的几元信息，可选2gram 3gram 4gram

```bash
--param_gram 3gram
```

- model_gram参数：想使用的模型元数，可选2gram 3gram 4gram

```bash
--model_gram 3gram
```

- use_file参数：是否从文件中读取输入，并且输出到文件中，可选True False 如果选择True则需要用“<”“>”重定向，如果选择False则需要再命令行里面交互

```bash
--use_file True < ../data/input.txt > ../data/output.txt
```

注意：文件和命令行输入，都应该满足如下格式，即一行为一句话的拼音，中间用空格隔开

```python
qing hua da xue
```

- times参数：命令行内交互的次数，可选：正整数

```bash
--times 20
```

2、推荐的用例

```bash
python pinyin.py --param_path ../saved_param/saved_param_sina_and_SMP_and_wenda_3gram.json --param_gram 3gram --model_gram 3gram --use_file True < ../data/input.txt > ../data/output.txt

python pinyin.py --param_path ../saved_param/saved_param_sina_and_SMP_3gram.json --param_gram 3gram --model_gram 3gram --use_file True < ../data/input.txt > ../data/output.txt

python pinyin.py --param_path ../saved_param/saved_param_sina_and_SMP_4gram.json --param_gram 4gram --model_gram 4gram --use_file True < ../data/input.txt > ../data/output.txt

python pinyin.py --param_path ../saved_param/saved_param_only_wenda_3gram.json --param_gram 3gram --model_gram 3gram --use_file True < ../data/input.txt > ../data/output.txt
```

#### test_acc.py

- param_path参数：指定使用的参数路径

```bash
--param_path ../saved_param/saved_param_sina_and_SMP_and_wenda_3gram.json
```

- std_input_path参数：测试输入文件路径

```bash
--std_input_path ../test/std_input.txt
```

- std_output_path参数：测试样例标准答案路径

```bash
--std_output_path ../test/std_output.txt
```

- model_output_path参数：存储模型输出的文件路径

```bash
--model_output_path ../test/model_output.txt
```

- param_gram参数:参数文件里面的参数存储的几元信息，可选2gram 3gram 4gram

```bash
--param_gram 3gram
```

- model_gram参数：想使用的模型元数，可选2gram 3gram 4gram

```bash
--model_gram 3gram
```

2、推荐的用例

```bash
# 对于“sina SMP 百度问答”数据集，测试3元模型效果
python test_acc.py --param_path ../saved_param/saved_param_sina_and_SMP_and_wenda_3gram.json --std_input_path ../test/std_input.txt --std_output_path ../test/std_output.txt --model_output_path ../test/model_output.txt --param_gram 3gram --model_gram 3gram 

# 对于“sina SMP 百度问答”数据集，测试2元模型效果
python test_acc.py --param_path ../saved_param/saved_param_sina_and_SMP_and_wenda_3gram.json --std_input_path ../test/std_input.txt --std_output_path ../test/std_output.txt --model_output_path ../test/model_output.txt --param_gram 3gram --model_gram 2gram  

# 对于“sina SMP”数据集，测试3元模型效果
python test_acc.py --param_path ../saved_param/saved_param_sina_and_SMP_3gram.json --std_input_path ../test/std_input.txt --std_output_path ../test/std_output.txt --model_output_path ../test/model_output.txt --param_gram 3gram --model_gram 3gram  

# 对于“sina SMP”数据集，测试2元模型效果
python test_acc.py --param_path ../saved_param/saved_param_sina_and_SMP_3gram.json --std_input_path ../test/std_input.txt --std_output_path ../test/std_output.txt --model_output_path ../test/model_output.txt --param_gram 3gram --model_gram 2gram  

# 对于“sina SMP”数据集，测试4元模型效果
python test_acc.py --param_path ../saved_param/saved_param_sina_and_SMP_4gram.json --std_input_path ../test/std_input.txt --std_output_path ../test/std_output.txt --model_output_path ../test/model_output.txt --param_gram 3gram --model_gram 4gram

# 对于“百度问答”数据集，测试3元模型效果
python test_acc.py --param_path ../saved_param/saved_param_only_wenda_3gram.json --std_input_path ../test/std_input.txt --std_output_path ../test/std_output.txt --model_output_path ../test/model_output.txt --param_gram 3gram --model_gram 3gram
```

### 三、其他说明

1、因为4元模型的参数很大，约2G，若加载参数时候kill了，请移步服务器上处理（本人在完成作业时，在服务器上训练、测试的4元模型）（sina SMP wenda的三元参数也是 约1G）

2、使用pinyin.py 和 test_acc.py时，加载模型参数在1-3分钟不等，请耐心等待

3、上传至github或清华云盘的文件可能过期，若过期请邮箱联系本人 guo-yy22@mails.tsinghua.edu.cn
