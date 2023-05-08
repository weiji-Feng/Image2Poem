"""
本文件用于训练一个专门用于古诗生成的GPT2模型,若有预训练模型的,可进行微调或直接忽略
"""
import json
import os
from transformers import BertTokenizer, GPT2LMHeadModel, GPT2Config
import matplotlib.pyplot as plt

from data_process import WordVocab, GPT2DataLoader
from pretrain.trainer import GPT2Trainer

"""
超参数设置：
"""


class CFG:
    def __init__(
            self,
            tokenizer_path: str = None,  # 预训练的tokenizer位置
            gpt2_path: str = None,       # gpt2预训练模型位置
            corpus_path: str = "../datasets/CCPC/ccpc_train_v1.0.json",  # 文本预料所在位置
            save_path: str = "../config/gpt_config/",
            epochs: int = 30,
            max_grad_norm: int = 1,
            lr: float = 1.8e-5,
            betas=(0.9, 0.999),
            weight_decay: float = 0.01,
            gradient_accumulation: int = 1,
            warmup_steps: int = 10000,
            with_cuda: bool = True,
            cuda_device: str = None,
            log_freq: int = 200,
    ):
        self.tokenizer_path = tokenizer_path
        self.gpt2_path = gpt2_path
        self.corpus_path = corpus_path
        self.save_path = save_path
        self.epochs = epochs
        self.max_grad_norm = max_grad_norm
        self.lr = lr
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

    tokenizer = BertTokenizer(vocab_file=os.path.join(cfg.save_path, "vocab.txt"), eos_token="[EOS]")

# 2. 创建一个GPT2模型
if cfg.gpt2_path is not None:
    gpt2model = GPT2LMHeadModel.from_pretrained(cfg.gpt2_path)
else:
    print("Creating a new GPT model...")
    config = GPT2Config(
        vocab_size=tokenizer.vocab_size,
        n_embd=768,
        n_layer=12,
        n_head=12,
        bos_token_id=tokenizer.cls_token_id,
        eos_token_id=tokenizer.sep_token_id
    )
    gpt2model = GPT2LMHeadModel(config=config)

# 3. 建立DataLoader
gpt_loader = GPT2DataLoader(cfg.corpus_path, tokenizer, batch_size=64)

# 进行训练
gpt_trainer = GPT2Trainer(
    model=gpt2model,
    dataset=gpt_loader,
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

results = gpt_trainer.train()

# 绘制损失曲线
plt.plot(list(range(1, len(results)+1)), results)
plt.title("Loss of Each Train Epochs")
plt.show()
