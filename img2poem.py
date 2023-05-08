"""
使用预训练模型进行图像生成古诗任务
若想训练一个自己的T5/GPT模型,详见./pretrain的相关文件
"""
import torch
from transformers import ChineseCLIPProcessor, ChineseCLIPModel
from transformers import BertTokenizer, T5ForConditionalGeneration
from transformers import GPT2LMHeadModel
from PIL import Image
from tqdm import tqdm


def cosine_similarity(x, y):
    return torch.sum(x * y) / (torch.sqrt(torch.sum(pow(x, 2))) * torch.sqrt(torch.sum(pow(y, 2))))


class GeneratePoemFromImage:
    """
    _create_keyword_dict: 调用CLIP模型的text_encoder对关键词进行向量编码，存储在字典中
    top_k_keywords: 计算图像的向量编码和所有关键词的余弦相似度,选择相似度最高的K个关键词
    generate_: 古诗生成
    generate_poem: 多次生成,取和关键词最贴合的几首古诗作为输出
    """

    def __init__(self,
                 lm_tokenizer: BertTokenizer,  # t5模型的tokenizer,用于分词
                 LanguageModel,  # 预训练的语言模型, 这里适合使用GPT2 & T5
                 clip_processor: ChineseCLIPProcessor,
                 clip: ChineseCLIPModel,  # clip预训练模型
                 keyword_path: str = None,
                 keyword_dict_path: str = None,
                 dict_save_path: str = "./datasets/keywords_dict.pt",
                 top_k: int = 8,
                 ):
        self.lm_tokenizer = lm_tokenizer
        self.lm_model = LanguageModel
        self.clip_processor = clip_processor
        self.clip_model = clip
        self.keyword_path = keyword_path
        self.keyword_dict_path = keyword_dict_path
        self.dict_save_path = dict_save_path
        self.keyword_dict = None
        self.tok_k = top_k

        assert self.keyword_path is not None or self.keyword_dict_path is not None, "关键词不存在! 请提供关键词再作诗!"

        if self.keyword_dict_path is None:
            self.keyword_dict = self._create_keyword_dict(self.keyword_path, self.dict_save_path)
        else:
            self.keyword_dict = torch.load(self.keyword_dict_path)

    def _create_keyword_dict(self, keyword_root=None, save_root="./datasets/keywords_dict.pt"):
        assert self.keyword_path is not None or keyword_root is not None, "请提供keyword.txt"

        root = keyword_root if keyword_root is not None else self.keyword_path
        keyword_name, text_feature = [], []
        with open(root, "r", encoding="utf-8") as f:
            data = f.readlines()
            for line in tqdm(data, total=len(data)):
                keyword_name.append(line.strip())
                feature = self.clip_processor(text=line.strip(), return_tensors="pt")
                text_feature.append(self.clip_model.get_text_features(**feature))
        text_feature = torch.cat(text_feature, dim=0)  # 将列表转换为一个矩阵
        text_feature = text_feature / text_feature.norm(p=2, dim=-1, keepdim=True)  # Normalize
        keyword_dict = {keyword_name[i]: text_feature[i:i + 1] for i in range(len(keyword_name))}
        torch.save(keyword_dict, save_root)
        return keyword_dict

    def top_k_keywords(self, img_feature):
        text_features = torch.cat(list(self.keyword_dict.values()), dim=0)
        similar = torch.tensor([
            cosine_similarity(text_features[i], img_feature) for i in range(text_features.shape[0])])
        top_k = torch.topk(similar, k=self.tok_k)
        # for v, i in zip(top_k.values.squeeze(0).tolist(), top_k.indices.squeeze(0).tolist()):
        #     print(list(self.keyword_dict.keys())[i], v)
        return top_k.indices.squeeze(0).tolist()

    def generate_(self, image_path):
        image = Image.open(image_path)
        image = self.clip_processor(images=image, return_tensors="pt")
        img_feature = self.clip_model.get_image_features(**image)
        img_feature = img_feature / img_feature.norm(p=2, dim=-1, keepdim=True)

        top_keyword_index = self.top_k_keywords(img_feature)
        top_keywords = [list(self.keyword_dict.keys())[i] for i in top_keyword_index]
        prompt = "关键词：" + " ".join(top_keywords) + " [EOS] "

        input_ids = self.lm_tokenizer.encode(prompt)
        for _ in range(4):
            inputs = {"input_ids": torch.tensor([input_ids])}
            outputs = self.lm_model.generate(**inputs, max_length=100, num_beams=1,
                                             no_repeat_ngram_size=1, early_stopping=False, do_sample=True)
            input_ids = input_ids[:-1] + outputs.squeeze(0)[1:-1].tolist() + input_ids[-1:]
        return top_keywords, input_ids

    def generate_poem(self, image_path, epochs=50):
        def check_each_poem(path):
            keys, output_ids = self.generate_(path)
            output = self.lm_tokenizer.decode(output_ids)
            output = output.replace(" ", "").replace("[CLS]", "").replace("[SEP]", "")
            poet = output.split("[EOS]")[1:-1]
            if len(set([len(sentence) for sentence in poet])) > 1:
                return "", 0
            poet = "，".join(poet) + "。"

            # 检查诗与关键词的匹配性
            poet_feature = self.clip_processor(text=poet, return_tensors="pt")  # 构造诗的向量
            poet_feature = self.clip_model.get_text_features(**poet_feature)
            similarity = sum([cosine_similarity(self.keyword_dict[k], poet_feature) for k in keys]) / len(keys)
            return poet, similarity.item()

        candidate = [check_each_poem(image_path) for _ in tqdm(range(epochs), total=epochs)]
        return candidate


if __name__ == "__main__":
    import argparse

    # create a parser
    parser = argparse.ArgumentParser(description='Generate a poem from an image.')
    # add arguments
    parser.add_argument('--vocab_path', type=str, default="./config/t5_config/vocab.txt",
                        help='the path of the tokenizer vocab')
    parser.add_argument('--model_type', type=str, default="T5", help='choose language model, \'T5\' or \'GPT2\'')
    parser.add_argument('--model_path', type=str, default="./config/t5_config", help='the path of language model')
    parser.add_argument('--clip_path', type=str, default="./config/Chinese_CLIP",
                        help='the path of CLIP processor & model')
    parser.add_argument('--image_path', default="./datasets/images/feiliu.jpg", help='the path of the image file')
    parser.add_argument('--keyword_path', default="./datasets/keywords.txt", help='the path of poem keywords')
    parser.add_argument('--keyword_dict_path', type=str, default="./datasets/keywords_dict.pt",
                        help='the path to save keywords and its text encoder vector')
    parser.add_argument('--epochs', type=int, default=50, help='the number of epochs for training (default: 50)')
    parser.add_argument('--top_k', type=int, default=5, help='the number of top candidates to show (default: 5)')
    # parse arguments
    args = parser.parse_args()

    # build model
    tokenizer = BertTokenizer(vocab_file=args.vocab_path, eos_token="[EOS]")

    language_model = None
    assert args.model_type == 'T5' or args.model_type == 'GPT2', "语言模型不支持!请使用T5或GPT2."
    if args.model_type == 'T5':
        language_model = T5ForConditionalGeneration.from_pretrained(args.model_path)
    elif args.model_type == 'GPT2':
        language_model = GPT2LMHeadModel.from_pretrained(args.model_path)
    processor = ChineseCLIPProcessor.from_pretrained(args.clip_path)
    clip_model = ChineseCLIPModel.from_pretrained(args.clip_path)

    generator = GeneratePoemFromImage(
        lm_tokenizer=tokenizer, LanguageModel=language_model,
        clip_processor=processor, clip=clip_model,
        keyword_path=args.keyword_path,
        keyword_dict_path=args.keyword_dict_path,   # 若有自己的keyword.txt文件，这里可以设置为None
        dict_save_path=args.keyword_dict_path,
        top_k=args.top_k
    )

    candidates = generator.generate_poem(args.image_path, epochs=args.epochs)
    candidates = sorted(candidates, key=lambda x: x[1], reverse=True)     # 根据相似度高低排序
    for p, v in candidates[:10]:
        print("诗句: ", p, "\t 相似度评分: ", v)
