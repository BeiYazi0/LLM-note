# Tokenizer

[大模型基础组件 - Tokenizer](https://zhuanlan.zhihu.com/p/651430181)

Tokenizer分词算法是NLP大模型最基础的组件，基于Tokenizer可以将文本转换成独立的token列表，进而转换成输入的向量成为计算机可以理解的输入形式。

1. 根据不同的切分粒度可以把tokenizer分为: 基于词的切分，基于字的切分和基于subword的切分。基于subword的切分是目前的主流切分方式。
2. subword 的切分包括: BPE(/BBPE), WordPiece 和 Unigram 三种分词模型。其中 WordPiece 可以认为是一种特殊的 BPE。
3. 完整的分词流程包括：文本归一化，预切分，基于分词模型的切分，后处理。
4. SentencePiece 是一个分词工具，内置 BEP 等多种分词方法，基于 Unicode 编码并且将空格视为特殊的 token。这是当前大模型的主流分词方案。

|分词方法|典型模型|
|---|---|
|BPE	|GPT, GPT-2, GPT-J, GPT-Neo, RoBERTa, BART, LLaMA, ChatGLM-6B, Baichuan|
|WordPiece|BERT, DistilBERT，MobileBERT|
|Unigram	|AlBERT, T5, mBART, XLNet|

---

## 基于subword的切分

### 特点

基于subword的切分能很好平衡基于词切分和基于字切分的优缺点，也是目前主流最主流的切分方式。基于词和字的切分都会存在一定的问题，直接应用的效果比较差。

基于词的切分，会造成:
   1. 词表规模过大
   2. 一定会存在UNK，造成信息丢失
   3. 不能学习到词缀之间的关系，例如：dog与dogs，happy与unhappy

基于字的切分，会造成:
   5. 每个token的信息密度低
   6. 序列过长，解码效率很低

所以基于词和基于字的切分方式是两个极端，其优缺点也是互补的。而折中的subword就是一种相对平衡的方案。

subword的基本切分原则是：
   1. 高频词依旧切分成完整的整词
   2. 低频词被切分成有意义的子词，例如 dogs => [dog, ##s]

基于subword的切分可以实现：
   4. 词表规模适中，解码效率较高
   5. 不存在UNK，信息不丢失
   6. 能学习到词缀之间的关系

基于subword的切分包括：BPE，WordPiece 和 Unigram 三种分词模型。

### 切分流程

Tokenizer包括训练和推理两个环节。训练阶段指得是从语料中获取一个分词器模型。推理阶段指的是给定一个句子，基于分词模型切分成一连串的token。

基本的流程包括归一化，预分词，基于分词模型的切分，后处理4个步骤。

[归一化](https://huggingface.co/docs/tokenizers/api/normalizers)包括删除多余的换行和空格，转小写，移除音调等。

[预分词](https://huggingface.co/docs/tokenizers/api/pre-tokenizers)把句子切分成更小的“词”单元。可以基于空格或者标点进行切分。

```python
input: Hello, how are  you?

pre-tokenize:
[BERT]: [('Hello', (0, 5)), (',', (5, 6)), ('how', (7, 10)), ('are', (11, 14)), ('you', (16, 19)), ('?', (19, 20))]

[GPT2]: [('Hello', (0, 5)), (',', (5, 6)), ('Ġhow', (6, 10)), ('Ġare', (10, 14)), ('Ġ', (14, 15)), ('Ġyou', (15, 19)), ('?', (19, 20))]

[t5]: [('▁Hello,', (0, 6)), ('▁how', (7, 10)), ('▁are', (11, 14)), ('▁you?', (16, 20))] 
```

BERT的tokenizer就是直接基于空格和标点进行切分。GPT2也是基于空格和标签，但是空格会保留成特殊字符“Ġ”。
T5则只基于空格进行切分，标点不会切分。并且空格会保留成特殊字符"▁"，并且句子开头也会添加特殊字符"▁"。

基于[分词模型](https://huggingface.co/docs/tokenizers/api/models)(BPE, WordPiece 和 Unigram)进行切分。

[后处理](https://huggingface.co/docs/tokenizers/api/post-processors)包括一些特殊的分词逻辑，例如添加sepcial token：[CLS],[SEP]等。

训练语料

```python
corpus = [
    "This is the Hugging Face Course.",
    "This chapter is about tokenization.",
    "This section shows several tokenizer algorithms.",
    "Hopefully, you will be able to understand how they are trained and generate tokens.",
]
```

---

## BPE

Byte-Pair Encoding(BPE)是最广泛采用的subword分词器。

    训练方法：从字符级的小词表出发，训练产生合并规则以及一个词表
    编码方法：将文本切分成字符，再应用训练阶段获得的合并规则
    经典模型：GPT, GPT-2, RoBERTa, BART, LLaMA, ChatGLM等

### 训练

首先进行预切分处理。这里采用gpt2的预切分逻辑。具体会按照空格和标点进行切分，并且空格会保留成特殊的字符“Ġ”。

```python
    def __init__(self, corpus, vocab_size):
        self.vocab_size = vocab_size
        self.merge_rules = []

        # init pre tokenize function
        self.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)

        # pre tokenize
        self.pre_tokenized_corpus = [self.pre_tokenizer.pre_tokenize_str(text) for text in corpus]
        print(self.pre_tokenized_corpus)

        self._train()
```

获得的pre_tokenized_corpus如下，每个单元分别为[word, (start_index, end_index)]。

```python
[[('This', (0, 4)), ('Ġis', (4, 7)), ('Ġthe', (7, 11)), ('ĠHugging', (11, 19)), ('ĠFace', (19, 24)), ('ĠCourse', (24, 31)), ('.', (31, 32))], 
[('This', (0, 4)), ('Ġchapter', (4, 12)), ('Ġis', (12, 15)), ('Ġabout', (15, 21)), ('Ġtokenization', (21, 34)), ('.', (34, 35))], 
[('This', (0, 4)), ('Ġsection', (4, 12)), ('Ġshows', (12, 18)), ('Ġseveral', (18, 26)), ('Ġtokenizer', (26, 36)), ('Ġalgorithms', (36, 47)), ('.', (47, 48))], 
[('Hopefully', (0, 9)), (',', (9, 10)), ('Ġyou', (10, 14)), ('Ġwill', (14, 19)), ('Ġbe', (19, 22)), ('Ġable', (22, 27)), ('Ġto', (27, 30)), ('Ġunderstand', (30, 41)), ('Ġhow', (41, 45)), ('Ġthey', (45, 50)), ('Ġare', (50, 54)), ('Ġtrained', (54, 62)), ('Ġand', (62, 66)), ('Ġgenerate', (66, 75)), ('Ġtokens', (75, 82)), ('.', (82, 83))]]
```

进一步统计每个整词的词频，获得字符级别的小词表，基于小词表就可以对每个整词进行切分。

训练流程如下：
1. 基于 word2splits 统计 vocabs 中相邻两个 pair 的词频 pair2count；
2. 统计当前频率最高的相邻 pair，在合并规则中添加相应的合并规则；
3. 根据更新后的 vocab 重新对 word2count 进行切分。具体实现上，可以直接在旧的 word2split 上应用新的合并规则；
4. 重复上述步骤直到整个词表的大小达到预先设定的词表大小。

```python
    def _train(self):
        # 统计每个预分词的词频
        word2count = defaultdict(int)
        for split_text in self.pre_tokenized_corpus:
            for word, _ in split_text:
                word2count[word] += 1

        # 先获得字符级别的小词表
        vocab_set = set()
        for word in word2count:
            vocab_set.update(list(word))
        vocabs = list(vocab_set)

        # 基于小词表就可以对每个预分词进行切分
        word2splits = {word: [c for c in word] for word in word2count}

        # 重复循环直到整个词表的大小达到预先设定的词表大小。
        while len(vocabs) < self.vocab_size:
            pair2score = self._compute_pair2score(word2splits, word2count)
            best_pair = self._compute_most_score_pair(pair2score)
            vocabs.append(best_pair[0] + best_pair[1])
            self.merge_rules.append(best_pair) # 在合并规则中添加合并规则
            word2splits = self._merge_pair(best_pair[0], best_pair[1], word2splits)

        print(self.merge_rules)
```

假定最终词表的大小为50，经过上述迭代后我们获得的合并规则如下。

```python
[('Ġ', 't'), ('i', 's'), ('e', 'r'), ('Ġ', 'a'), ('Ġt', 'o'), ('e', 'n'), ('T', 'h'), ('Th', 'is'), ('o', 'u'), ('s', 'e'), ('Ġto', 'k'),
 ('Ġtok', 'en'), ('n', 'd'), ('Ġ', 'is'), ('Ġt', 'h'), ('Ġth', 'e'), ('i', 'n'), ('Ġa', 'b'), ('Ġtoken', 'i'), ('Ġtokeni', 'z')]
```

### 推理

在推理阶段，给定一个句子，我们需要将其切分成一个token的序列。 具体实现上需要先对句子进行预分词并切分成字符级别的序列，然后根据合并规则进行合并。

```python
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
```

"This is not a token." 转换成的 token 列表如下。

```python
['This', 'Ġis', 'Ġ', 'n', 'o', 't', 'Ġa', 'Ġtoken', '.']
```

---

## BBPE

Byte-level BPE (BBPE) 算法是 BPE 算法的进一步升级。

核心思想是用 byte 来构建最基础的词表而不是字符。首先将文本按照 UTF-8 进行编码，每个字符在 UTF-8 的表示中占据 1-4 个 byte。
在 byte 序列上再使用 BPE 算法，进行 byte level 的相邻合并。

优点：可以更好的处理跨语言和不常见字符的特殊问题(例如，颜文字)，相比传统的 BPE 更节省词表空间，每个 token 也能获得更充分的训练。

缺点：在解码阶段，一个 byte 序列可能解码后不是一个合法的字符序列，需要采用动态规划的算法进行解码，使其能解码出尽可能多的合法字符。

---

## WordPiece

WordPiece 分词与 BPE 非常类似，只是在训练阶段合并 pair 的策略不是 pair 的频率而是互信息。

$$socre = log(p(ab)) - (log(p(a)) + log(p(b))) = log(p(ab)/p(a)p(b))$$

一个 pair 的频率很高，但是其中 pair 的一部分的频率更高，这时候不一定需要进行该 pair 的合并。 
而如果一个 pair 的频率很高，并且这个 pair 的两个部分都是只出现在这个 pair 中，就说明这个 pair 很值得合并。

    训练方法：从字符级的小词表出发，训练产生合并规则以及一个词表
    编码方法：将文本切分成词，对每个词在词表中进行最大前向匹配
    经典模型：BERT及其系列DistilBERT，MobileBERT等

### 训练

首先进行预切分处理。这里采用bert的预切分逻辑。具体会按照空格和标点进行切分，并且每个标点符号都视为一个单词。

```python
    def __init__(self, corpus, vocab_size):
        self.vocab_size = vocab_size
        self.vocabs = []

        # init pre tokenize function
        self.pre_tokenizer = pre_tokenizers.BertPreTokenizer()

        # pre tokenize
        self.pre_tokenized_corpus = [self.pre_tokenizer.pre_tokenize_str(text) for text in corpus]
        print(self.pre_tokenized_corpus)

        self._train()
        self.vocabs.extend(['[CLS]', '[MASK]', '[PAD]', '[SEP]', '[UNK]'])
        print(self.vocabs)
```

获得的pre_tokenized_corpus如下，每个单元分别为[word, (start_index, end_index)]。

```python
[[('This', (0, 4)), ('is', (5, 7)), ('the', (8, 11)), ('Hugging', (12, 19)), ('Face', (20, 24)), ('Course', (25, 31)), ('.', (31, 32))], 
 [('This', (0, 4)), ('chapter', (5, 12)), ('is', (13, 15)), ('about', (16, 21)), ('tokenization', (22, 34)), ('.', (34, 35))], 
 [('This', (0, 4)), ('section', (5, 12)), ('shows', (13, 18)), ('several', (19, 26)), ('tokenizer', (27, 36)), ('algorithms', (37, 47)), ('.', (47, 48))], 
 [('Hopefully', (0, 9)), (',', (9, 10)), ('you', (11, 14)), ('will', (15, 19)), ('be', (20, 22)), ('able', (23, 27)), ('to', (28, 30)), ('understand', (31, 41)), ('how', (42, 45)), ('they', (46, 50)), ('are', (51, 54)), ('trained', (55, 62)), ('and', (63, 66)), ('generate', (67, 75)), ('tokens', (76, 82)), ('.', (82, 83))]]
```

进一步统计每个整词的词频，获得字符级别的小词表，基于小词表（注意这里如果字符不是不一个词的开始，需要添加上特殊字符"##"）就可以对每个整词进行切分。

训练流程如下：
1. 基于 word2splits 统计 vocabs 中相邻两个 pair 的互信息 p(ab)/p(a)p(b)；
2. 统计当前互信息最高的相邻 pair，并添加到词表中；
3. 根据更新后的 vocab 重新对 word2count 进行切分。具体实现上，可以直接在旧的 word2split 上应用新的合并规则；
4. 重复上述步骤直到整个词表的大小达到预先设定的词表大小。

```python
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
```

假定最终词表的大小为 70，经过上述迭代后我们获得的词表如下。

```python
['F', '##o', '##b', 'b', '##v', 'u', '##e', 'H', 'i', '##w', '##r', '##z', 'h', '##i', '##s', 'C', '##h', '##k', '##a', 
 '##m', '##c', '##d', '##n', 'g', '##p', 'T', '##u', 'w', '##l', '##y', 'y', '##t', ',', '.', '##f', 's', 'a', 'c', '##g', 
 't', 'ab', '##fu', 'Fa', 'Fac', '##ct', '##ful', '##full', '##fully', 'Th', 'ch', '##hm', 'cha', 'chap', 'chapt', '##thm', 
 'Hu', 'Hug', 'Hugg', 'sh', 'th', 'is', '##thms', '##za', '##zat', '##ut', '##ta', '##at', '##sta', '##ra', '##rsta', '[CLS]', '[MASK]', '[PAD]', '[SEP]', '[UNK]']
```

注意词表中添加了特殊的token：[CLS], [MASK], [PAD], [SEP], [UNK]。

### 推理

在推理阶段，给定一个句子，我们需要将其切分成一个token的序列。
具体实现上需要先对句子进行预分词，然后对每个词进行在词表中进行最大前向的匹配。如果词表中不存在则为UNK。

```python
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

    def tokenize(self, text: str) -> list[str]:
        words = [word for word, _ in self.pre_tokenizer.pre_tokenize_str(text)]
        encoded_words = [self._encode_word(word) for word in words]
        return sum(encoded_words, [])
```

"This is the Hugging Face course!" 转换成的 token 列表如下。

```python
['Th', '##i', '##s', 'is', 'th', '##e', 'Hugg', '##i', '##n', '##g', 'Fac', '##e', 'c', '##o', '##u', '##r', '##s', '##e', '[UNK]']
```

---

## Unigram

Unigram 分词与 BPE 和 WordPiece 不同，是基于一个大词表逐步裁剪成一个小词表。

通过 Unigram 语言模型计算删除不同 subword 造成的损失来衡量 subword 的重要性，保留重要性较高的子词。

    训练方法：从包含字符和全部子词的大词表出发，逐步裁剪出一个小词表，并且每个词都有自己的分数。
    编码方法：将文本切分成词，对每个词基于Viterbi算法求解出最佳解码路径。
    经典模型：AlBERT, T5, mBART, Big Bird, XLNet

### 训练

首先进行预切分处理。这里采用 xlnet 的预切分逻辑。具体会按照空格进行切分，标点不会切分。并且空格会保留成特殊字符"▁"，句子开头也会添加特殊字符"▁"。

```python
    def __init__(self, corpus, vocab_size):
        self.vocab_size = vocab_size
        self.vocabs = []

        # init pre tokenize function
        self.pre_tokenizer = pre_tokenizers.Metaspace(replacement='_', add_prefix_space=True)

        # pre tokenize
        self.pre_tokenized_corpus = [self.pre_tokenizer.pre_tokenize_str(text) for text in corpus]
        print(self.pre_tokenized_corpus)

        self._train()
        self.vocabs.extend(['[CLS]', '[MASK]', '[PAD]', '[SEP]', '[UNK]'])
        print(self.vocabs)
```

获得的pre_tokenized_corpus如下，每个单元分别为[word, (start_index, end_index)]。

```python
[[('_This', (0, 4)), ('_is', (4, 7)), ('_the', (7, 11)), ('_Hugging', (11, 19)), ('_Face', (19, 24)), ('_Course.', (24, 32))], 
 [('_This', (0, 4)), ('_chapter', (4, 12)), ('_is', (12, 15)), ('_about', (15, 21)), ('_tokenization.', (21, 35))], 
 [('_This', (0, 4)), ('_section', (4, 12)), ('_shows', (12, 18)), ('_several', (18, 26)), ('_tokenizer', (26, 36)), ('_algorithms.', (36, 48))], 
 [('_Hopefully,', (0, 10)), ('_you', (10, 14)), ('_will', (14, 19)), ('_be', (19, 22)), ('_able', (22, 27)), ('_to', (27, 30)), ('_understand', (30, 41)), ('_how', (41, 45)), ('_they', (45, 50)), ('_are', (50, 54)), ('_trained', (54, 62)), ('_and', (62, 66)), ('_generate', (66, 75)), ('_tokens.', (75, 83))]]
```

统计词表的全部子词和词频，取前 300 个词，构成最初的大词表。为了避免 OOV（out of vocab），char级别的词均需要保留。

训练流程如下：
1. 基于 vocabs 中 token 的频数 count 统计 vocabs 中各 token 的 loss = -log(count / total_count)，获得 model={token:loss}；
2. 基于 model 以及 Viterbi 算法就可以求解出，输入的一个词的最佳分词路径，进而计算整个语料 word2count 上的 loss；
3. 尝试移除 model 中的一个子词，并计算移除后新的 model 在全部语料上的 loss，从而获得这个子词的 score，即删除这个子词使得 loss 新增的量；
4. 为了提升迭代效率，从 vocabs 批量删除前 10% 的结果，即让整体 loss 增量最小的前 10% 的词。(删除这些词对整体 loss 的影响不大。)
5. 重复上述步骤直到整个词表的大小达到预先设定的词表大小。

```python
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
```

假定最终词表的大小为 100，经过上述迭代后我们获得的词表如下。

```python
{'_': 2.318585434340487, 'T': 4.653960350157523, 'h': 3.5553480614894135, 'i': 3.1876232813640963, 's': 3.1876232813640963, ...
 'sev': 5.752572638825633, 'seve': 5.752572638825633, 'sever': 5.752572638825633, 'severa': 5.752572638825633, 'several': 5.752572638825633}
```

### 推理

在推理阶段，给定一个句子，需要将其切分成一个 token 的序列。 具体实现上先对句子进行预分词，然后对每个词基于 Viterbi 算法进行解码。

```python
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
        encoded_words = [self._encode_word(word)[0] for word in words]
        return sum(encoded_words, [])
```

"This is the Hugging Face course." 转换成的 token 列表如下。

```python
['_This', '_is', '_the', '_Hugging', '_Face', '_', 'c', 'ou', 'r', 's', 'e', '.']
```

---