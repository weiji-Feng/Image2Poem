import torch
from transformers import BertTokenizer, GPT2LMHeadModel


tokenizer = BertTokenizer(vocab_file="../weights/vocab.txt", eos_token="[EOS]", pad_token="[PAD]")
print(tokenizer.eos_token_id)
gpt2model = GPT2LMHeadModel.from_pretrained("../weights")

# 输入关键词：
inputs_text = "关键词：鸟 花草 山 飞流"
max_length = 60
input_ids = []
input_ids.extend(tokenizer.encode(inputs_text))
input_ids = input_ids[:-1] + [tokenizer.eos_token_id]
inputs = {"input_ids": torch.tensor([input_ids]), "attention_mask": torch.tensor([[1 for _ in range(len(input_ids))]])}
outputs = gpt2model.generate(**inputs, max_length=60, num_beams=4,
                             no_repeat_ngram_size=2, early_stopping=True, do_sample=True)
print(tokenizer.decode(outputs[0]))
