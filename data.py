import os
import copy
import numpy as np
from tqdm import tqdm
from tokenizer import SentencepieceTokenizer
from random import randrange, randint, shuffle, choice, random
import torch
from torch.utils.data import Dataset, DataLoader
# file_path = '../wiliam_dataset/news000.txt'



# vocab = SentencepieceTokenizer(model_path='./vocab/tokenizer.model')
# vocab.text_to_idx("안녕하세요 만나서 반가워요")

# line_cnt = 0
# with open(input, "r") as in_f:
#     for line in in_f:
#         line_cnt += 1


def load_file(file_path, vocab):
    docs = []

    line_cnt = 0
    with open(file_path, "r") as in_f:
        for line in in_f:
            line_cnt += 1

    with open(file_path, "r") as f:
        doc = []
        for i, line in enumerate(tqdm(f, total=line_cnt, desc=f"Loading {file_path}", unit=" lines")):
            line = line.strip()
            if line == "":
                if 0 < len(doc):
                    docs.append(doc)
                    doc = []
            else:
                pieces = vocab.tok.encode_as_pieces(line)
                if 0 < len(pieces):
                    doc.append(pieces)
        if doc:
            docs.append(doc)
    return docs

""" 쵀대 길이 초과하는 토큰 자르기 """
def trim_tokens(tokens_a, tokens_b, max_seq):
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_seq:
            break

        if len(tokens_a) > len(tokens_b):
            del tokens_a[0]
        else:
            tokens_b.pop()

def get_vocab_list(vocab):
    vocab_list = []
    for i in range(len(vocab)):
        vocab_list.append(vocab.idx_to_token(i))
    return vocab_list


def create_pretrain_mask(tokens, mask_cnt, vocab_list):
    cand_idx = []
    for (i, token) in enumerate(tokens):
        if token == "[CLS]" or token == "[SEP]":
            continue
        if 0 < len(cand_idx) and not token.startswith(u"\u2581"):
            cand_idx[-1].append(i)
        else:
            cand_idx.append([i])
    shuffle(cand_idx)

    mask_lms = []
    for index_set in cand_idx:
        if len(mask_lms) >= mask_cnt:
            break
        if len(mask_lms) + len(index_set) > mask_cnt:
            continue
        for index in index_set:
            masked_token = None
            if random() < 0.8: # 80% replace with [MASK]
                masked_token = "[MASK]"
            else:
                if random() < 0.5: # 10% keep original
                    masked_token = tokens[index]
                else: # 10% random word
                    masked_token = choice(vocab_list)
            mask_lms.append({"index": index, "label": tokens[index]})
            tokens[index] = masked_token
    mask_lms = sorted(mask_lms, key=lambda x: x["index"])
    mask_idx = [p["index"] for p in mask_lms]
    mask_label = [p["label"] for p in mask_lms]

    return tokens, mask_idx, mask_label


def create_pretrain_bert_instances(docs, doc_idx, doc, n_seq, mask_prob, cls_prob, vocab_list):
    # for CLS], [SEP], [SEP]
    max_seq = n_seq - 3
    tgt_seq = max_seq
    
    instances = []
    current_chunk = []
    current_length = 0
    for i in range(len(doc)):
        current_chunk.append(doc[i]) # line
        current_length += len(doc[i])
        if i == len(doc) - 1 or current_length >= tgt_seq:
            if 0 < len(current_chunk):
                a_end = 1
                if 1 < len(current_chunk):
                    a_end = randrange(1, len(current_chunk))
                tokens_a = []
                for j in range(a_end):
                    tokens_a.extend(current_chunk[j])
                
                tokens_b = []
                if len(current_chunk) == 1 or random() < cls_prob:                  
                    is_next = 0
                    tokens_b_len = tgt_seq - len(tokens_a)
                    random_doc_idx = doc_idx
                    while doc_idx == random_doc_idx:
                        random_doc_idx = randrange(0, len(docs))
                    random_doc = docs[random_doc_idx]
                    random_start = randrange(0, len(random_doc))
                    for j in range(random_start, len(random_doc)):
                        tokens_b.extend(random_doc[j])
                else:
                    is_next = 1
                    for j in range(a_end, len(current_chunk)):
                        tokens_b.extend(current_chunk[j])

                trim_tokens(tokens_a, tokens_b, max_seq)
                assert 0 < len(tokens_a)
                assert 0 < len(tokens_b)

                tokens = ["[CLS]"] + tokens_a + ["[SEP]"] + tokens_b + ["[SEP]"]
                segment = [0] * (len(tokens_a) + 2) + [1] * (len(tokens_b) + 1)
                # print(tokens)
                tokens, mask_idx, mask_label = create_pretrain_mask(tokens, int((len(tokens) - 3) * mask_prob), vocab_list)

                instance = {
                    "tokens": tokens,
                    "segment": segment,
                    "is_next": is_next,
                    "mask_idx": mask_idx,
                    "mask_label": mask_label
                }
                instances.append(instance)

            current_chunk = []
            current_length = 0
    return instances

def create_pretrain_albert_instances(docs, doc_idx, doc, n_seq, mask_prob, cls_prob, vocab_list):
    # for CLS], [SEP], [SEP]
    max_seq = n_seq - 3
    tgt_seq = max_seq
    
    instances = []
    current_chunk = []
    current_length = 0
    for i in range(len(doc)):
        current_chunk.append(doc[i]) # line
        current_length += len(doc[i])
        if i == len(doc) - 1 or current_length >= tgt_seq:
            if 0 < len(current_chunk):
                a_end = 1
                if 1 < len(current_chunk):
                    a_end = randrange(1, len(current_chunk))
                tokens_a = []
                for j in range(a_end):
                    tokens_a.extend(current_chunk[j])
                
                tokens_b = []
                is_next = 1
                for j in range(a_end, len(current_chunk)):
                    tokens_b.extend(current_chunk[j])

                trim_tokens(tokens_a, tokens_b, max_seq)
                assert 0 < len(tokens_a)
                assert 0 < len(tokens_b)

                if len(current_chunk) == 1 or random() < cls_prob:
                    is_next = 0
                    # switch sentence order
                    tokens_tmp = tokens_a
                    tokens_a = tokens_b
                    tokens_b = tokens_tmp
                
                tokens = ["[CLS]"] + tokens_a + ["[SEP]"] + tokens_b + ["[SEP]"]
                segment = [0] * (len(tokens_a) + 2) + [1] * (len(tokens_b) + 1)
                # print(tokens)
                tokens, mask_idx, mask_label = create_pretrain_mask(tokens, int((len(tokens) - 3) * mask_prob), vocab_list)

                instance = {
                    "tokens": tokens,
                    "segment": segment,
                    "is_next": is_next,
                    "mask_idx": mask_idx,
                    "mask_label": mask_label
                }
                instances.append(instance)

            current_chunk = []
            current_length = 0
    return instances



# docs = load_file(file_path, vocab)

# vocab_list = get_vocab_list(vocab)
# doc_idx = 0
# doc = docs[0]
# n_seq=128
# mask_prob=0.2
# cls_prob = 0.1
# generation_type = 'albert'

# instances = []
# for doc_idx in range(len(docs)):
#     instance = create_pretrain_bert_instances(docs, doc_idx, docs[doc_idx], n_seq, mask_prob, cls_prob, vocab_list)
#     instances.append(instance)

# instances = create_pretrain_bert_instances(docs, doc_idx, doc, n_seq, mask_prob, cls_prob, vocab_list)
# instances = create_pretrain_albert_instances(docs, doc_idx, doc, n_seq, mask_prob, cls_prob, vocab_list)


class PretrainDataset(Dataset):

    def __init__(self, file, tok, max_len, mask_prob, cls_prob, generation_type):
        super().__init__()
        self.tok = tok
        self.max_len = max_len
        vocab_list = get_vocab_list(tok)
        docs = load_file(file, tok)
        generation_type = generation_type.lower()
        assert generation_type in ['bert', 'albert']
        self.instances = list()
        
        for doc_idx in range(len(docs)):
            if generation_type == 'bert':
                instance = create_pretrain_bert_instances(docs, doc_idx, docs[doc_idx], max_len, mask_prob, cls_prob, vocab_list)
            elif generation_type == 'albert':
                instance = create_pretrain_albert_instances(docs, doc_idx, docs[doc_idx], max_len, mask_prob, cls_prob, vocab_list)
            for ins in instance:
                self.instances.append(ins)
    
    def get_padding_data(self, inputs):
        if len(inputs) < self.max_len:
            if type(inputs) == list:
                inputs.extend([0] *(self.max_len - len(inputs)))
            else:
                pad = np.array([0] *(self.max_len - len(inputs)))
                inputs = np.concatenate([inputs, pad])

        return inputs

    
    def __getitem__(self, idx):
        instance = self.instances[idx]
        # print(instance['tokens'])
        input_ids = [self.tok.token_to_idx(token) for token in instance['tokens']]
        valid_length = [len(input_ids)]
        input_ids = self.get_padding_data(input_ids)
        
        segment_ids = instance['segment']
        segment_ids = self.get_padding_data(segment_ids)

        label_cls = [instance['is_next']]

        mask_idx = np.array(instance["mask_idx"], dtype=np.int)

        mask_label = np.array([self.tok.token_to_idx(p) for p in instance["mask_label"]], dtype=np.int)
        label_lm = np.full(self.max_len, dtype=np.int, fill_value=-1)
        label_lm[mask_idx] = mask_label
        
        ##generate origin inputs
        origin_ids = copy.deepcopy(input_ids)
        for i, v in zip(mask_idx, mask_label):
            origin_ids[i] = v

        mask_idx = self.get_padding_data(mask_idx)
        mask_label = self.get_padding_data(mask_label)
        


    #         return (input_ids,
#                 valid_length,
#                 segment_ids,
#                 label_cls,
#                 mask_idx,
#                 mask_label,
#                 label_lm)

        return (torch.tensor(input_ids),
                torch.tensor(origin_ids),
                torch.tensor(valid_length),
                torch.tensor(segment_ids),
                torch.tensor(label_cls),
                torch.tensor(mask_idx),
                torch.tensor(mask_label),
                torch.tensor(label_lm))

    def __len__(self):
        return len(self.instances)
        

# dataset = PretrainDataset(file_path, vocab, 128, 0.2, 0.5, 'bert')


# len(dataset.instances[0])
# a = dataset.__getitem__(0)

# a = [1,2,3]
# if type(a) == list:
#     print(1)