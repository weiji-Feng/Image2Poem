"""
本文件用于训练一个专门用于古诗生成的GPT2模型,若有预训练模型的,可进行微调或直接忽略
"""
import json
import os
from transformers import BertTokenizer, T5Config, T5ForConditionalGeneration
import matplotlib.pyplot as plt

from data_process import WordVocab, T5DataLoader
from pretrain.trainer import MT5Trainer

"""
超参数设置：
"""


class CFG:
    def __init__(
            self,
            tokenizer_path: str = None,  # 预训练的tokenizer位置
            t5_path: str = "../config/t5_config",       # gpt2预训练模型位置
            corpus_path: str = "../datasets/CCPC/ccpc_train_v1.0.json",  # 文本预料所在位置
            save_path: str = "../config/t5_config/",
            epochs: int = 10,
            batch_size: int = 32,
            max_grad_norm: int = 1,
            lr: float = 1.5e-5,
            betas=(0.9, 0.999),
            weight_decay: float = 0.01,
            gradient_accumulation: int = 1,
            warmup_steps: int = 10000,
            with_cuda: bool = True,
            cuda_device: str = None,
            log_freq: int = 150,
    ):
        self.tokenizer_path = tokenizer_path
        self.t5_path = t5_path
        self.corpus_path = corpus_path
        self.save_path = save_path
        self.epochs = epochs
        self.max_grad_norm = max_grad_norm
        self.lr = lr
        self.batch_size = batch_size
        self.betas = betas
        self.weight_decay = weight_decay
        self.gradient_accumulation = gradient_accumulation
        self.warmup_steps = warmup_steps
        self.with_cuda = with_cuda
        self.cuda_device = cuda_device
        self.log_freq = log_freq


cfg = CFG()
if not os.path.isdir(cfg.save_path):
    os.mkdir(cfg.save_path)

# 1. 训练一个tokenizer分词器并保存

if isinstance(cfg.tokenizer_path, str):
    tokenizer = BertTokenizer.from_pretrained(cfg.tokenizer_path)
else:
    print("Training a new tokenizer...")
    # 语料库重建
    text = []
    with open(cfg.corpus_path, "r", encoding="utf-8") as f:
        for line in f.readlines():
            dict_line = json.loads(line)
            text.append("关键词：" + dict_line["keywords"] + dict_line["content"])

    vocab = WordVocab(texts=text)
    vocab.save_vocab_txt(os.path.join(cfg.save_path, "vocab.txt"))

    tokenizer = BertTokenizer(vocab_file=os.path.join(cfg.save_path, "vocab.txt"), eos_token="[EOS]", pad_token="[PAD]")

# 2. 创建一个GPT2模型
if cfg.t5_path is not None:
    t5model = T5ForConditionalGeneration.from_pretrained(cfg.t5_path)
else:
    print("Creating a new T5 model...")
    config = T5Config(
        vocab_size=tokenizer.vocab_size,
        d_model=768,
        d_kv=64,
        num_layers=12,
        decoder_start_token_id=tokenizer.cls_token_id,
        pad_token_id=tokenizer.pad_token_id,
        bos_token_id=tokenizer.cls_token_id,
        eos_token_id=tokenizer.sep_token_id
    )
    t5model = T5ForConditionalGeneration(config=config)

# 3. 建立DataLoader
mt5_loader = T5DataLoader(cfg.corpus_path, tokenizer, batch_size=cfg.batch_size)
# x = [tokenizer.decode(mt5_loader[3][0][i].tolist()) for i in range(mt5_loader[3][0].shape[0])]
# for index in x:
#     print(index)
# x = [tokenizer.decode(mt5_loader[3][1][i].tolist()) for i in range(mt5_loader[3][1].shape[0])]
# for index in x:
#     print(index)

# 进行训练
t5_trainer = MT5Trainer(
    model=t5model,
    dataset=mt5_loader,
    epochs=cfg.epochs,
    max_grad_norm=cfg.max_grad_norm,
    lr=cfg.lr,
    betas=cfg.betas,
    weight_decay=cfg.weight_decay,
    gradient_accumulation=cfg.gradient_accumulation,
    warmup_steps=cfg.warmup_steps,
    with_cuda=cfg.with_cuda,
    cuda_device=cfg.cuda_device,
    log_freq=cfg.log_freq,
    save_path=cfg.save_path
)

# results = t5_trainer.train()
#
# # 绘制损失曲线
# plt.plot(list(range(1, len(results)+1)), results)
# plt.title("Loss of Each Train Epochs")
# plt.show()
