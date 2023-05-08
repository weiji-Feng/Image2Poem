import torch
from transformers import BertTokenizer, T5ForConditionalGeneration

tokenizer = BertTokenizer(vocab_file="../t5_config/vocab.txt", eos_token="[EOS]")
t5model = T5ForConditionalGeneration.from_pretrained("../t5_config")

# 输入关键词：
inputs_text = "关键词：仙山 丹青 山水 林壑 飞泉 [EOS]"
input_ids = []
input_ids.extend(tokenizer.encode(inputs_text))
outputs = None
print(tokenizer.decode(input_ids))
for i in range(4):
    inputs = {"input_ids": torch.tensor([input_ids])}
    outputs = t5model.generate(**inputs, max_length=100, num_beams=3,
                               no_repeat_ngram_size=1, early_stopping=True, do_sample=True)
    input_ids = input_ids[:-1] + outputs.squeeze(0)[1:-1].tolist() + input_ids[-1:]

print(tokenizer.decode(input_ids))
