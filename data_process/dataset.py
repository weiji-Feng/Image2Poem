import torch
from transformers import BertTokenizer
from tqdm import tqdm
import random
import json


class GPT2DataLoader:
    def __init__(
            self, corpus_path: str, tokenizer: BertTokenizer, seq_len=100,
            stride: int = None, batch_size=64, encoding="utf-8", corpus_lines=None, on_memory=True
    ):
        """
        实现从语料库中获取某一batch数据
        :param corpus_path: 文本预料所在位置
        :param tokenizer: 分词器,使用bert tokenizer
        :param seq_len: 最长的序列长度
        :param stride: 步长
        :param batch_size:batch大小，默认设置64
        :param encoding: 默认"utf-8"
        :param corpus_lines: 提供语料行数,默认为None
        :param on_memory: 是否将预料全部存储于内存中
        """
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.stride = stride
        self.on_memory = on_memory
        self.corpus_lines = corpus_lines
        self.corpus_path = corpus_path
        self.encoding = encoding
        self.batch_size = batch_size

        with open(corpus_path, "r", encoding=encoding) as f:
            if self.corpus_lines is None and not on_memory:
                for _ in tqdm(f, desc="Loading Dataset", total=corpus_lines):
                    self.corpus_lines += 1

            if on_memory:
                self.lines = []
                for line in tqdm(f, desc="Loading Data", total=corpus_lines):
                    # json 文件预处理, 其他文件做相应处理即可
                    json_line = json.loads(line)
                    self.lines.append(
                        "关键词：" + json_line["keywords"].replace(" ", "").strip() + "\t" + json_line["content"].strip()
                    )
                self.corpus_lines = len(self.lines)

        if not on_memory:
            self.file = open(corpus_path, "r", encoding=encoding)
            self.random_file = open(corpus_path, "r", encoding=encoding)

            for _ in range(random.randrange(self.corpus_lines if self.corpus_lines < 1000 else 1000)):
                self.random_file.__next__()

        self.get_trans_dataset()  # shuffle
        self.steps = len(self.lines) // batch_size

    def __len__(self):
        return self.steps

    def __getitem__(self, item):
        # 取一个batch的原始中文诗句
        batch_data = self.lines[item * self.batch_size: (item + 1) * self.batch_size]

        # 计算最大长度, add 3是因为[CLS]+s1+[SEP]+s2+[SEP]
        max_len = max([len(s.replace("\t", "")) for s in batch_data]) + 3
        tokens, targets, labels, segments, attn_masks = [], [], [], [], []

        for i in range(len(batch_data)):
            data_i = batch_data[i].split("\t")      # data_i = [keywords, poem]
            key_token, key_target, key_label = self.random_word(data_i[0], None)
            poet_token, poet_target, poet_label = self.random_word(data_i[1], keywords=data_i[0])
            # 加入 [CLS] and [SEP]
            # token是mask过的原句子ids, target是原句子的ids, label是关注的mask单词
            key_token = [self.tokenizer.cls_token_id] + key_token + [self.tokenizer.eos_token_id]
            key_target = [self.tokenizer.cls_token_id] + key_target + [self.tokenizer.eos_token_id]
            key_label = [0] + key_label + [0]   # 不是需要预测的token, label=0
            poet_token = poet_token + [self.tokenizer.sep_token_id]
            poet_target = poet_target + [self.tokenizer.sep_token_id]
            poet_label = poet_label + [0]       # 不是需要预测的token, label=0
            # 设置token_type_ids
            segment = [0 for _ in range(len(key_token))] + [1 for _ in range(len(poet_token))]
            # set attention mask
            attn_mask = [1 for _ in range(len(key_token) + len(poet_token))]

            padding = [self.tokenizer.pad_token_id for _ in range(max_len - len(key_token) - len(poet_token))]
            tokens_i = key_token + poet_token + padding
            target_i = key_target + poet_target + padding
            labels_i = key_label + poet_label + padding
            segment.extend(padding)
            attn_mask += [0 for _ in range(len(padding))]

            assert len(tokens_i) == len(target_i) == len(labels_i) == len(segment) == len(attn_mask) == max_len, "长度有错！"

            tokens.append(tokens_i)
            targets.append(target_i)
            labels.append(labels_i)
            segments.append(segment)
            attn_masks.append(attn_mask)

            del key_token, key_target, key_label, poet_token, poet_target, poet_label   # 清除内存

        tokens, targets = torch.tensor(tokens, dtype=torch.int64), torch.tensor(targets, dtype=torch.int64)
        labels, segments = torch.tensor(labels, dtype=torch.int64), torch.tensor(segments, dtype=torch.int64)
        attn_masks = torch.tensor(attn_masks, dtype=torch.int64)
        return tokens, targets, labels, segments, attn_masks

    def sentence2ids(self, sentence):
        tokens = self.tokenizer.tokenize(sentence)
        ids = self.tokenizer.convert_tokens_to_ids(tokens)
        return ids

    def get_trans_dataset(self):
        # 暂停使用原方案, 仅shuffle
        random.shuffle(self.lines)

    def get_corpus_line(self, item):
        if self.on_memory:
            return self.lines[item]
        else:
            line = self.file.__next__()
            while len(line) >= self.seq_len:
                line = self.file.__next__()
            if line is not None:
                self.file.close()
                self.file = open(self.corpus_path, "r", encoding=self.encoding)
                line = self.file.__next__()
            return line

    def random_word(self, sentence: str, keywords: str = None):
        """ :return mask_token_ids, token_ids, labels"""
        tokens = list(sentence.replace(" ", "").strip())
        keywords = set(list(keywords.replace(" ", "").strip())) if keywords is not None else None    # 关键字
        targets, labels = [], []
        for i, token in enumerate(tokens):
            prob = random.random()
            if prob < 0.15:
                # 设置mask的概率为15%
                tokens[i] = self.tokenizer.mask_token_id    # 设置该token为[MASK]
                targets.append(self.tokenizer.vocab.get(token, self.tokenizer.unk_token_id))
                if (keywords is not None and token in keywords) or token == "|":
                    # 关键词的mask,是我们关心的信息
                    labels.append(self.tokenizer.vocab.get(token, self.tokenizer.unk_token_id))
                else:
                    labels.append(0)
            else:
                # 不满足mask情况
                tokens[i] = self.tokenizer.vocab.get(token, self.tokenizer.unk_token_id)
                targets.append(self.tokenizer.vocab.get(token, self.tokenizer.unk_token_id))
                if token == "|":
                    labels.append(self.tokenizer.vocab.get(token, self.tokenizer.unk_token_id))
                else:
                    labels.append(0)
        return tokens, targets, labels


class T5DataLoader:
    def __init__(
            self, corpus_path: str, tokenizer: BertTokenizer, seq_len=100,
            stride: int = None, batch_size=64, encoding="utf-8", corpus_lines=None, on_memory=True
    ):
        """
        实现从语料库中获取某一batch数据
        :param corpus_path: 文本预料所在位置
        :param tokenizer: 分词器,使用bert tokenizer
        :param seq_len: 最长的序列长度
        :param stride: 步长
        :param batch_size:batch大小，默认设置64
        :param encoding: 默认"utf-8"
        :param corpus_lines: 提供语料行数,默认为None
        :param on_memory: 是否将预料全部存储于内存中
        """
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.stride = stride
        self.on_memory = on_memory
        self.corpus_lines = corpus_lines
        self.corpus_path = corpus_path
        self.encoding = encoding
        self.batch_size = batch_size

        self.lines = []
        self._create()      # 导入数据

        random.shuffle(self.lines)
        self.steps = len(self.lines) // batch_size

    def __len__(self):
        return self.steps

    def __getitem__(self, item):
        # 取一个batch的原始中文诗句
        batch_data = self.lines[item * self.batch_size: (item + 1) * self.batch_size]
        len_in = max([len(s[0].replace(" ", "").replace("[EOS]", "")) for s in batch_data]) + 6
        len_out = max([len(s[1].replace(" ", "").replace("[EOS]", "")) for s in batch_data]) + 3
        # len_in, len_out = max([len(s[0]) for s in batch_data]) + 2, max([len(s[1]) for s in batch_data]) + 2
        tokens, targets, attns_x, attns_y = [], [], [], []
        for i in range(self.batch_size):
            x = self.tokenizer.encode(batch_data[i][0])
            attn_x = [1 for _ in range(len(x))]
            attn_x += [0 for _ in range(len_in - len(x))]
            x.extend([self.tokenizer.pad_token_id for _ in range(len_in - len(x))])

            y = self.tokenizer.encode(batch_data[i][1])
            attn_y = [1 for _ in range(len(y))]
            attn_y += [0 for _ in range(len_out - len(y))]
            y += [self.tokenizer.pad_token_id for _ in range(len_out - len(y))]
            # 保存
            assert len(x) == len(attn_x) == len_in and len(y) == len(attn_y) == len_out, "长度不匹配！"
            tokens.append(x)
            targets.append(y)
            attns_x.append(attn_x)
            attns_y.append(attn_y)
        tokens, targets = torch.tensor(tokens, dtype=torch.int64), torch.tensor(targets, dtype=torch.int64)
        attns_x, attns_y = torch.tensor(attns_x, dtype=torch.int64), torch.tensor(attns_y, dtype=torch.int64)
        # ids = self.tokenizer(text=batch_input, text_target=batch_target, padding=True, return_tensors="pt")
        return tokens, targets, attns_x, attns_y

    def _create(self):
        with open(self.corpus_path, "r", encoding=self.encoding) as f:
            if self.on_memory:
                self.lines = []
                for line in tqdm(f, desc="Loading Data", total=self.corpus_lines):
                    json_line = json.loads(line)
                    keywords = json_line["keywords"].strip()
                    poems = json_line["content"].strip().split("|")    # 列表，一个元素是一句诗
                    for i in range(len(poems)):
                        x = "关键词：" + keywords + " [EOS] "
                        self.lines.append((x + " [EOS] ".join(poems[:i]) + " [EOS] ", poems[i] + " [EOS] "))
                        # self.lines.append((x + "|".join(poems[:i]), "|".join(poems[i:])))     # \t分割输入和预测输出
                # self.lines = self.lines[-200000:]
                self.corpus_lines = len(self.lines)
