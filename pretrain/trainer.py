import torch
from torch.optim import AdamW
import torch.nn as nn
from datetime import datetime
from tqdm import tqdm
from transformers import GPT2LMHeadModel, get_linear_schedule_with_warmup, T5ForConditionalGeneration

from data_process import GPT2DataLoader, T5DataLoader


class GPT2Trainer:
    """
    BERTTrainer make the pretrained BERT model with two LM training method.
        1. Masked Language Model : 3.3.1 Task #1: Masked LM
        2. Next Sentence prediction : 3.3.2 Task #2: Next Sentence Prediction
    please check the details on README.md with simple example.
    """

    def __init__(self, model: GPT2LMHeadModel, dataset: GPT2DataLoader, epochs: int = 10, max_grad_norm: float = 1.0,
                 lr: float = 1e-4, betas=(0.9, 0.999), weight_decay: float = 0.01, gradient_accumulation: float = 1,
                 warmup_steps=10000, with_cuda: bool = True, cuda_device=None, log_freq: int = 10, save_path: str = ""):

        cuda_condition = torch.cuda.is_available() and with_cuda
        self.device = torch.device("cuda:0" if cuda_condition else "cpu")
        print("使用设备: ", self.device)
        self.max_grad_norm = max_grad_norm
        self.gradient_accumulation = gradient_accumulation
        self.epochs = epochs

        # This BERT model will be saved every epoch
        self.save_path = save_path
        self.model = model.to(self.device)
        print("模型架构：")
        print(self.model)
        print("="*20)

        if with_cuda and torch.cuda.device_count() > 1:
            print("Using %d GPUS for BERT" % torch.cuda.device_count())
            self.model = nn.DataParallel(self.model, device_ids=cuda_device)

        # Setting the train loader
        self.train_data = dataset
        print("训练数据量：", len(self.train_data))

        # Setting the CrossEntropy Loss, which will ignore data with label=0
        self.loss_fn = nn.CrossEntropyLoss(ignore_index=0)

        # Setting the Adam optimizer with hyper-param
        self.optim = AdamW(self.model.parameters(), lr=lr, betas=betas, weight_decay=weight_decay)
        total_step = dataset.steps / gradient_accumulation * epochs
        self.scheduler = get_linear_schedule_with_warmup(
            self.optim, num_warmup_steps=warmup_steps, num_training_steps=total_step)

        self.log_freq = log_freq

        print("Total Parameters: ", sum([p.nelement() for p in self.model.parameters()]))

    def train(self):
        # 训练轮数： self.epochs
        str_code = "train"

        # set the model state
        self.model.train()
        avg_loss, overall_step = 0.0, int(0)
        epoch_losses = []     # 保存每一轮训练的损失

        for epoch in range(self.epochs):
            epoch_loss = 0.0
            # Setting the tqdm progress bar
            data_iter = tqdm(
                range(len(self.train_data)), desc="EP_%s:%d" % (str_code, epoch + 1),
                total=len(self.train_data), bar_format="{l_bar}{r_bar}")

            for i in data_iter:
                # 0. batch_data will be sent into the device(GPU or cpu)
                tokens, targets, labels, segments, attn_masks = self.train_data[i]
                tokens, targets = tokens.to(self.device), targets.to(self.device)
                labels, segments = labels.to(self.device), segments.to(self.device)
                attn_masks = attn_masks.to(self.device)

                # 1. forward the next_sentence_prediction and masked_lm model
                outputs = self.model(
                    input_ids=tokens, labels=targets, attention_mask=attn_masks, token_type_ids=segments
                )

                # 2. loss of train result
                pred_word_loss, logits = outputs[:2]
                # logits size = (B, N, C) => (B, C, N)
                pred_key_loss = self.loss_fn(logits.permute(0, 2, 1), labels * attn_masks)

                # 3. backward and optimization only in train
                loss = 0.9 * pred_word_loss + 0.1 * pred_key_loss

                epoch_loss += loss.item()
                #  optimizer step
                if (overall_step + 1) % self.gradient_accumulation == 0:
                    avg_loss += loss.item()
                    self.optim.zero_grad()
                    loss.backward()
                    self.optim.step()
                    self.scheduler.step()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.max_grad_norm)

                if (overall_step + 1) % self.log_freq == 0:
                    print('now time: {}:{}. Step {} of of epoch {}, loss {}'.format(
                        datetime.now().hour,
                        datetime.now().minute,
                        i + 1,
                        epoch + 1,
                        avg_loss * self.gradient_accumulation / (self.log_freq / self.gradient_accumulation)
                    ))
                    avg_loss = 0
                overall_step += 1

            # 一轮结束后, 检验是否需要保存模型
            if len(epoch_losses) == 0 or epoch_loss / self.train_data.steps < min(epoch_losses):
                self.save(epoch + 1)
            epoch_losses.append(epoch_loss)
        return epoch_losses

    def save(self, epoch):
        print("Epoch {:03d} model saved on {}".format(epoch, self.save_path))
        self.model.save_pretrained(self.save_path)


class MT5Trainer:
    """
    BERTTrainer make the pretrained BERT model with two LM training method.
        1. Masked Language Model : 3.3.1 Task #1: Masked LM
        2. Next Sentence prediction : 3.3.2 Task #2: Next Sentence Prediction
    please check the details on README.md with simple example.
    """

    def __init__(self, model: T5ForConditionalGeneration, dataset: T5DataLoader, epochs: int = 10, max_grad_norm: float = 1.0,
                 lr: float = 1e-4, betas=(0.9, 0.999), weight_decay: float = 0.01, gradient_accumulation: float = 1,
                 warmup_steps=10000, with_cuda: bool = True, cuda_device=None, log_freq: int = 10, save_path=""):

        cuda_condition = torch.cuda.is_available() and with_cuda
        self.device = torch.device("cuda:0" if cuda_condition else "cpu")
        print("使用设备: ", self.device)
        self.save_path = save_path
        self.max_grad_norm = max_grad_norm
        self.gradient_accumulation = gradient_accumulation
        self.epochs = epochs

        # This BERT model will be saved every epoch
        self.model = model.to(self.device)
        print("模型架构：")
        print(self.model)
        print("="*20)

        if with_cuda and torch.cuda.device_count() > 1:
            print("Using %d GPUS for BERT" % torch.cuda.device_count())
            self.model = nn.DataParallel(self.model, device_ids=cuda_device)

        # Setting the train loader
        self.train_data = dataset
        print("训练数据量：", len(self.train_data))

        # Setting the CrossEntropy Loss, which will ignore data with label=0
        self.loss_fn = nn.CrossEntropyLoss(ignore_index=0)  # ignore padding

        # Setting the Adam optimizer with hyper-param
        self.optim = AdamW(self.model.parameters(), lr=lr, betas=betas, weight_decay=weight_decay)
        total_step = dataset.steps / gradient_accumulation * epochs
        self.scheduler = get_linear_schedule_with_warmup(
            self.optim, num_warmup_steps=warmup_steps, num_training_steps=total_step)

        self.log_freq = log_freq

        print("Total Parameters: ", sum([p.nelement() for p in self.model.parameters()]))

    def train(self):
        # 训练轮数： self.epochs
        str_code = "train"

        # set the model state
        self.model.train()
        avg_loss, overall_step = 0.0, int(0)
        epoch_losses = []     # 保存每一轮训练的损失

        for epoch in range(self.epochs):
            epoch_loss = 0.0
            # Setting the tqdm progress bar
            data_iter = tqdm(
                range(len(self.train_data)), desc="EP_%s:%d" % (str_code, epoch + 1),
                total=len(self.train_data), bar_format="{l_bar}{r_bar}")

            for i in data_iter:
                # 0. batch_data will be sent into the device(GPU or cpu)
                tokens, targets, attn_x, attn_y = self.train_data[i]
                tokens, targets = tokens.to(self.device), targets.to(self.device)
                attn_x, attn_y = attn_x.to(self.device), attn_y.to(self.device)

                # 1. forward the next_sentence_prediction and masked_lm model
                outputs = self.model.forward(input_ids=tokens, labels=targets)
                # outputs = self.model(
                #     input_ids=tokens, attention_mask=attn_x,
                #     decoder_input_ids=targets, decoder_attention_mask=attn_y,
                #     labels=targets
                # )

                # 2. loss of train result
                loss, _ = outputs[:2]
                # # logits size = (B, N, C) => (B, C, N)
                # pred_key_loss = self.loss_fn(logits.permute(0, 2, 1), targets * attn_y)

                # 3. backward and optimization only in train
                # loss = 0.85 * pred_word_loss + 0.15 * pred_key_loss

                epoch_loss += loss.item()
                #  optimizer step
                if (overall_step + 1) % self.gradient_accumulation == 0:
                    avg_loss += loss.item()
                    self.optim.zero_grad()
                    loss.backward()
                    self.optim.step()
                    self.scheduler.step()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.max_grad_norm)

                if (overall_step + 1) % self.log_freq == 0:
                    print('now time: {}:{}. Step {} of of epoch {}, loss {}'.format(
                        datetime.now().hour,
                        datetime.now().minute,
                        i + 1,
                        epoch + 1,
                        avg_loss * self.gradient_accumulation / (self.log_freq / self.gradient_accumulation)
                    ))
                    avg_loss = 0
                overall_step += 1

            # 一轮结束后, 检验是否需要保存模型
            if len(epoch_losses) == 0 or epoch_loss / self.train_data.steps < min(epoch_losses):
                self.save(epoch + 1)
            epoch_losses.append(epoch_loss)
        return epoch_losses

    def save(self, epoch):
        print("Epoch {:03d} model saved on {}".format(epoch, self.save_path))
        self.model.save_pretrained(self.save_path)
