from collections import Counter
import pickle
from tqdm import tqdm


class TorchVocab(object):
    def __init__(self, counter, max_size=None, min_freq=1, specials=None,
                 vector=None, unk_init=None, vector_cache=None):

        if specials is None:
            specials = ['<pad>', '<oov>']
        self.freq = counter
        counter = counter.copy()
        min_freq = max(min_freq, 1)

        self.itos = list(specials)
        # frequencies of special tokens are not counted when building vocabulary
        # in frequency order
        for tok in specials:
            del counter[tok]

        max_size = None if max_size is None else max_size + len(self.itos)

        # sort by frequency, then alphabetically
        words_and_frequencies = sorted(counter.items(), key=lambda tup: tup[0])
        words_and_frequencies.sort(key=lambda tup: tup[1], reverse=True)

        for word, freq in words_and_frequencies:
            if freq < min_freq or len(self.itos) == max_size:
                break
            self.itos.append(word)

        # stoi is simply a reverse dict for itos
        self.stoi = {tok: i for i, tok in enumerate(self.itos)}

        self.vector = vector
        # if vector is not None:
        #     self.load_vector(vector, unk_init=unk_init, cache=vector_cache)
        # else:
        assert unk_init is None and vector_cache is None

    def __eq__(self, other):
        if self.freq != other.freq:
            return False
        if self.stoi != other.stoi:
            return False
        if self.itos != other.itos:
            return False
        if self.vector != other.vector:
            return False
        return True

    def __len__(self):
        return len(self.itos)

    def vocab_rerank(self):
        self.stoi = {word: i for i, word in enumerate(self.itos)}

    def extend(self, v, sort=False):
        words = sorted(v.itos) if sort else v.itos
        for w in words:
            if w not in self.stoi:
                self.itos.append(w)
                self.stoi[w] = len(self.stoi) - 1


class Vocab(TorchVocab):
    def __init__(self, counter, max_size=None, min_freq=1):
        super(Vocab, self).__init__(counter, specials=["[PAD]", "[UNK]", "[SEP]", "[CLS]", "[MASK]", "[EOS]"],
                                    max_size=max_size, min_freq=min_freq)

        self.pad_index = 0
        self.unk_index = 1
        self.sep_index = 2
        self.sos_index = 3
        self.mask_index = 4
        self.eos_index = 5

    def to_seq(self, sentence, seq_len, with_eos=False, with_sos=False):
        pass

    def from_seq(self, seq, join=False, with_pad=False):
        pass

    @staticmethod
    def load_vocab(vocab_path: str):
        with open(vocab_path, "rb", encoding='utf-8') as f:
            return pickle.load(f)

    def save_vocab(self, vocab_path):
        with open(vocab_path, "wb") as f:
            pickle.dump(self, f)

    def save_vocab_txt(self, vocab_path):
        with open(vocab_path, "w", encoding="utf-8") as f:
            for key, _ in self.stoi.items():
                f.write(str(key) + "\n")


class WordVocab(Vocab):
    def __init__(self, texts, max_size=None, min_freq=1):
        print("Building Vocab")
        stopwords = ["$", "0", "1", "2", "3", "4", "5", "6", "7", "8", "9",
                     "?", "_", "“", "”", "、", "《", "》", "[", "]", "\""]
        counter = Counter()
        for line in tqdm(texts):
            if isinstance(line, list):
                words = line
            else:
                words = line.strip().replace("\n", "").replace("\t", "")  # 这里不能使用split
                for w in stopwords:
                    words.replace(w, "")
                words = list(words)

            for word in words:
                counter[word] += 1
        super(WordVocab, self).__init__(counter=counter, max_size=max_size, min_freq=min_freq)

    def to_seq(self, sentence, seq_len=None, with_eos=False, with_sos=False, with_len=False):
        if isinstance(sentence, str):
            sentence = list(sentence.strip())       # 若是英文，则是sentence.strip().split()

        seq = [self.stoi.get(word, self.unk_index) for word in sentence]

        if with_eos:
            seq += [self.sep_index]
        if with_sos:
            seq = [self.sos_index] + seq

        origin_seq_len = len(seq)

        if seq_len is None:
            pass
        elif len(seq) <= seq_len:
            seq += [self.pad_index for _ in range(seq_len - len(seq))]
        else:
            seq = seq[:seq_len]

        return (seq, origin_seq_len) if with_len else seq

    def from_seq(self, seq, join=False, with_pad=False):
        words = [self.itos[idx] if idx < len(self.itos) else "<%d>" % idx
                 for idx in seq if not with_pad or idx != self.pad_index]
        return " ".join(words) if join else words

    @staticmethod
    def load_vocab(vocab_path: str):
        with open(vocab_path, "rb") as f:
            return pickle.load(f)
