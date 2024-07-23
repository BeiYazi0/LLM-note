import copy

from tokenizers import pre_tokenizers
from collections import defaultdict

import numpy as np

corpus = [
    "This is the Hugging Face Course.",
    "This chapter is about tokenization.",
    "This section shows several tokenizer algorithms.",
    "Hopefully, you will be able to understand how they are trained and generate tokens.",
]


class BasicTokenizer:
    def __init__(self, corpus, vocab_size, pre_tokenizer):
        self.vocab_size = vocab_size
        self.vocabs = []

        # init pre tokenize function
        self.pre_tokenizer = pre_tokenizer

        # pre tokenize
        self.pre_tokenized_corpus = [self.pre_tokenizer.pre_tokenize_str(text) for text in corpus]
        print(self.pre_tokenized_corpus)

    # 基于word2splits统计vocabs中相邻两个pair的词频pair2count
    def _compute_pair2score(self, word2splits, word2count):
        raise NotImplementedError

    # 统计当前频率最高的相邻pair
    def _compute_most_score_pair(self, pair2count):
        '''
        Args:
            pair2count: dict{pair: (char or str, char: str) - count: int}

        Returns:
            best_pair: (char or str, char: str)
        '''
        best_pair = None
        max_freq = None
        for pair, freq in pair2count.items():
            if max_freq is None or max_freq < freq:
                best_pair = pair
                max_freq = freq
        return best_pair

    # 根据更新后的vocab重新对word2count进行切分。具体实现上，可以直接在旧的word2split上应用新的合并规则
    def _merge_pair(self, a, b, new_token, word2splits):
        '''
        Args:
            a: char or str
            b: char or str
            word2splits: dict{subword: str - split subword: list[char or str]}

        Returns:
            new_word2splits: dict{subword: str - split subword: list[char or str]}
        '''
        new_word2splits = dict()
        for word, split in word2splits.items():
            if len(split) == 1:
                new_word2splits[word] = split
                continue
            i = 0
            while i < len(split) - 1:
                if split[i] == a and split[i + 1] == b:
                    split = split[:i] + [new_token] + split[i + 2:]
                else:
                    i += 1
            new_word2splits[word] = split
        return new_word2splits

    def _init_vocab(self, word2count):
        raise NotImplementedError

    def _new_token_get(self, best_pair):
        raise NotImplementedError

    def _train(self):
        # 统计每个预分词的词频
        word2count = defaultdict(int)
        for split_text in self.pre_tokenized_corpus:
            for word, _ in split_text:
                word2count[word] += 1

        word2splits = self._init_vocab(word2count)

        # 重复循环直到整个词表的大小达到预先设定的词表大小。
        while len(self.vocabs) < self.vocab_size:
            pair2score = self._compute_pair2score(word2splits, word2count)
            best_pair = self._compute_most_score_pair(pair2score)
            new_token = self._new_token_get(best_pair)
            self.vocabs.append(new_token)
            word2splits = self._merge_pair(best_pair[0], best_pair[1], new_token, word2splits)

    def _encode_word(self, word):
        raise NotImplementedError

    def tokenize(self, text: str) -> list[str]:
        words = [word for word, _ in self.pre_tokenizer.pre_tokenize_str(text)]
        encoded_words = [self._encode_word(word) for word in words]
        return sum(encoded_words, [])


class BPETokenizer(BasicTokenizer):
    def __init__(self, corpus, vocab_size):
        self.merge_rules = []

        pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
        super().__init__(corpus, vocab_size, pre_tokenizer)

        self._train()
        print(self.merge_rules)

    # 基于word2splits统计vocabs中相邻两个pair的词频pair2count
    def _compute_pair2score(self, word2splits, word2count):
        '''
        Args:
            word2splits: dict{subword: str - split subword: list[char or str]}
            word2count: dict{subword: str - count: int}

        Returns:
            pair2count: dict{pair: (char or str, char: str) - count: int}
        '''
        pair2count = defaultdict(int)
        for word, word_count in word2count.items():
            split = word2splits[word]
            if len(split) == 1:
                continue
            for i in range(len(split) - 1):
                pair = (split[i], split[i + 1])
                pair2count[pair] += word_count
        return pair2count

    def _init_vocab(self, word2count):
        # 先获得字符级别的小词表
        vocab_set = set()
        for word in word2count:
            vocab_set.update(list(word))
        self.vocabs = list(vocab_set)

        # 基于小词表就可以对每个预分词进行切分
        word2splits = {word: [c for c in word] for word in word2count}
        return word2splits

    def _new_token_get(self, best_pair):
        self.merge_rules.append(best_pair)  # 在合并规则中添加合并规则
        return best_pair[0] + best_pair[1]

    def tokenize(self, text: str) -> list[str]:
        # pre tokenize
        words = [word for word, _ in self.pre_tokenizer.pre_tokenize_str(text)]
        # split into char level
        splits = [[c for c in word] for word in words]
        # apply merge rules
        for merge_rule in self.merge_rules:
            for index, split in enumerate(splits):
                i = 0
                while i < len(split) - 1:
                    if split[i] == merge_rule[0] and split[i + 1] == merge_rule[1]:
                        split = split[:i] + ["".join(merge_rule)] + split[i + 2:]
                    else:
                        i += 1
                splits[index] = split
        return sum(splits, [])


class WPTokenizer(BasicTokenizer):
    def __init__(self, corpus, vocab_size):
        pre_tokenizer = pre_tokenizers.BertPreTokenizer()
        super().__init__(corpus, vocab_size, pre_tokenizer)

        self._train()
        self.vocabs.extend(['[CLS]', '[MASK]', '[PAD]', '[SEP]', '[UNK]'])
        print(self.vocabs)

    # 基于word2splits统计vocabs中相邻两个pair的词频pair2count
    def _compute_pair2score(self, word2splits, word2count):
        '''
        Args:
            word2splits: dict{subword: str - split subword: list[char or str]}
            word2count: dict{subword: str - count: int}

        Returns:
            pair2count: dict{pair: (char or str, char: str) - count: int}
        '''
        vocab2count = defaultdict(int)
        pair2count = defaultdict(int)
        for word, word_count in word2count.items():
            splits = word2splits[word]
            if len(splits) == 1:
                vocab2count[splits[0]] += word_count
                continue
            for i in range(len(splits) - 1):
                pair = (splits[i], splits[i + 1])
                vocab2count[splits[i]] += word_count
                pair2count[pair] += word_count
            vocab2count[splits[-1]] += word_count
        scores = {
            pair: freq / (vocab2count[pair[0]] * vocab2count[pair[1]])
            for pair, freq in pair2count.items()
        }
        return scores

    def _init_vocab(self, word2count):
        # 先获得字符级别的小词表
        vocab_set = set()
        for word in word2count:
            vocab_set.add(word[0])
            vocab_set.update(["##" + c for c in word[1:]])
        self.vocabs = list(vocab_set)

        # 基于小词表就可以对每个预分词进行切分
        word2splits = {word: [word[0]] + ['##' + c for c in word[1:]] for word in word2count}

        return word2splits

    def _new_token_get(self, best_pair):
        return best_pair[0] + best_pair[1][2:] if best_pair[1].startswith('##') else best_pair[1]

    def _encode_word(self, word):
        tokens = []
        while len(word) > 0:
            i = len(word)
            while i > 0 and word[:i] not in self.vocabs:
                i -= 1
            if i == 0:
                return ["[UNK]"]
            tokens.append(word[:i])
            word = word[i:]
            if len(word) > 0:
                word = f"##{word}"
        return tokens


class UNGramTokenizer(BasicTokenizer):
    def __init__(self, corpus, vocab_size):
        self.percent_to_remove = 0.1

        pre_tokenizer = pre_tokenizers.Metaspace(replacement='_')
        super().__init__(corpus, vocab_size, pre_tokenizer)

        self._train()
        print(self.model)

    def _compute_loss(self, model, word2count):
        loss = 0
        for word, freq in word2count.items():
            _, word_loss = self._encode_word(word, model)
            loss += freq * word_loss
        return loss

    # 尝试移除model中的一个子词，并计算移除后新的model在全部语料上的loss
    def _compute_scores(self, model, word2count):
        scores = {}
        model_loss = self._compute_loss(model, word2count)
        for token, score in model.items():
            # We always keep tokens of length 1
            if len(token) == 1:
                continue
            model_without_token = copy.deepcopy(model)
            _ = model_without_token.pop(token)
            scores[token] = self._compute_loss(model_without_token, word2count) - model_loss
        return scores

    def _init_vocab(self, word2count):
        char2count = defaultdict(int)
        sub_word2count = defaultdict(int)
        for word, count in word2count.items():
            for i in range(len(word)):
                char2count[word[i]] += count
                for j in range(i + 2, len(word) + 1):
                    sub_word2count[word[i:j]] += count
        sorted_sub_words = sorted(sub_word2count.items(), key=lambda x: x[1], reverse=True)
        # init a large vocab with 300
        tokens = list(char2count.items()) + sorted_sub_words[: 300 - len(char2count)]
        self.vocabs = {token: count for token, count in tokens}

        total_count = sum([count for token, count in self.vocabs.items()])
        model = {token: -np.log(count / total_count) for token, count in self.vocabs.items()}

        return model

    def _train(self):
        # 统计每个预分词的词频
        word2count = defaultdict(int)
        for split_text in self.pre_tokenized_corpus:
            for word, _ in split_text:
                word2count[word] += 1

        model = self._init_vocab(word2count)

        # 重复循环直到整个词表的大小达到预先设定的词表大小。
        while len(model) > self.vocab_size:
            scores = self._compute_scores(model, word2count)
            sorted_scores = sorted(scores.items(), key=lambda x: x[1])
            # Remove percent_to_remove tokens with the lowest scores.
            for i in range(int(len(model) * self.percent_to_remove)):
                _ = self.vocabs.pop(sorted_scores[i][0])
            total_count = sum([count for token, count in self.vocabs.items()])
            model = {token: -np.log(count / total_count) for token, count in self.vocabs.items()}

        self.model = model

    # Viterbi算法
    def _encode_word(self, word, model):
        best_segmentations = [{"start": 0, "score": 1}] + [{"start": None, "score": None} for _ in range(len(word))]
        for start_idx in range(len(word)):
            # This should be properly filled by the previous steps of the loop
            best_score_at_start = best_segmentations[start_idx]["score"]
            for end_idx in range(start_idx + 1, len(word) + 1):
                token = word[start_idx:end_idx]
                if token in model and best_score_at_start is not None:  # 可匹配的 start_idx
                    score = model[token] + best_score_at_start
                    # If we have found a better segmentation (lower score) ending at end_idx
                    if (
                            best_segmentations[end_idx]["score"] is None
                            or best_segmentations[end_idx]["score"] > score
                    ):
                        best_segmentations[end_idx] = {"start": start_idx, "score": score}
        segmentation = best_segmentations[-1]
        if segmentation["score"] is None:
            # We did not find a tokenization of the word -> unknown
            return ["<unk>"], None
        score = segmentation["score"]
        start = segmentation["start"]
        end = len(word)
        tokens = []
        while start != 0:
            tokens.insert(0, word[start:end])
            next_start = best_segmentations[start]["start"]
            end = start
            start = next_start
        tokens.insert(0, word[start:end])
        return tokens, score

    def tokenize(self, text: str) -> list[str]:
        words = [word for word, _ in self.pre_tokenizer.pre_tokenize_str(text)]
        encoded_words = [self._encode_word(word, self.model)[0] for word in words]
        return sum(encoded_words, [])


if __name__ == "__main__":
    example = "This is not a token."
    tokenizer = BPETokenizer(corpus, 50)
    res = tokenizer.tokenize(example)
    print(res)

    example = "This is the Hugging Face course!"
    tokenizer = WPTokenizer(corpus, 70)
    res = tokenizer.tokenize(example)
    print(res)

    example = "This is the Hugging Face course."
    tokenizer = UNGramTokenizer(corpus, 100)
    res = tokenizer.tokenize(example)
    print(res)
