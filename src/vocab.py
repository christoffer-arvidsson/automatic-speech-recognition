from collections import defaultdict, Counter   
import itertools

class Vocab:
    def __init__(self):
        self.UNKNOWN = "<UNK>"
        self.PAD = "_"
        self.BOS = "<"
        self.EOS = ">"

        self.dummies = [self.UNKNOWN, self.PAD, self.BOS, self.EOS]

    @staticmethod
    def tokenize(sentence):
        # return sentence.split(" ")
        return list(sentence)

    def build(self, sentences):
        lower = [s.lower() for s in sentences]
        tokens = [self.tokenize(s) for s in lower]

        word_freqs = Counter(w for s in tokens for w in s)
        word_freqs = sorted(((f, w) for w, f in word_freqs.items()), reverse=True)
        self.itos = self.dummies + [w for _, w in word_freqs]
        self.stoi = {w:i for i, w in enumerate(self.itos)}

    def encode_sentence(self, sentence):
        tokens = self.tokenize(sentence)
        unk = self.stoi[self.UNKNOWN]
        bos = self.stoi[self.BOS]
        eos = self.stoi[self.EOS]

        return [bos] + [self.stoi.get(w.lower(), unk) for w in tokens] + [eos]

    def encode_docs(self, docs):
        encoded = [self.encode_sentence(s) for s in docs] 
        pad_token = self.stoi[self.PAD]
        padded = list(zip(*itertools.zip_longest(*encoded, fillvalue=pad_token)))

        return padded

    def decode_tokens(self, tokens):
        decoded = ""
        for c in tokens:
            decoded += self.itos[c]

            if c == self.itos[self.EOS]:
                break

        return decoded

