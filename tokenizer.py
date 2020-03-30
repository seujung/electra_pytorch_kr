#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import unicode_literals, print_function, division

from collections import Counter
from tqdm import tqdm
from sentencepiece import SentencePieceTrainer

import dill
import sentencepiece as sp

# ==================================================


class SentencepieceTokenizer:
    """Wrapper class for sentencepiece tokenizer

    This class supports methods for NLP task based on google sentencepiece
    """

    def __init__(
        self,
        model_path: str = None,
        pad_token: str = "[PAD]",
        cls_token: str = "[CLS]",
        sep_token: str = "[SEP]",
        unknown_token: str = "<unk>",
        start_token: str = "<s>",
        end_token: str = "</s>",
    ):
        """Initialize SentencepieceTokenizer with special tokens

        Args:
            model_path (str):    tokenizer instance (based on sentencepiece)
            pad_token (str):     pad token for padding of sentence
            cls_token (str):     cls token
            sep_token (str):     separate token for set separation of sentence
            unknown_token (str): unknown token
            start_token (str):   start token for set start of sentence
            end_token (str):     end token for set end of sentence
        """
        self._unknown_token = unknown_token
        self._pad_token = pad_token
        self._cls_token = cls_token
        self._start_token = start_token
        self._sep_token = sep_token
        self._end_token = end_token
        self.tok = sp.SentencePieceProcessor()
        self.except_idx = self.get_except_idx()
        self.model_name = ""

        if model_path is not None:
            self.tok.Load(model_path)

    def get_except_idx(self):
        except_list = list()
        except_list.append(self.token_to_idx(self._unknown_token))
        except_list.append(self.token_to_idx(self._pad_token))
        except_list.append(self.token_to_idx(self._cls_token))
        except_list.append(self.token_to_idx(self._start_token))
        except_list.append(self.token_to_idx(self._sep_token))
        except_list.append(self.token_to_idx(self._unknown_token))
        except_list.append(self.token_to_idx(self._end_token))
        return except_list

    def tokenize(self, text, to_id=True):
        if to_id:
            return self.tok.EncodeAsIds(text)
        else:
            return self.tok.EncodeAsPieces(text)

    def token_to_text(self, token):
        return self.tok.decode_pieces(token)

    def idx_to_token(self, idx):
        return self.tok.IdToPiece(idx)

    def token_to_idx(self, token):
        return self.tok.PieceToId(token)

    def idx_to_text(self, idx):
        text = list()

        for i in idx:
            if i not in self.except_idx:
                text.append(self.idx_to_token(i))
        return self.tok.DecodePieces(text)

    def text_to_idx(
        self,
        text: str,
        max_seq_len: int = None,
        use_pad: bool = False,
        cls_token: bool = False,
        start_token: bool = False,
        end_token: bool = False,
    ):
        """
        convert text to token indices

        Args:
            text(str): target text
            max_seq_len(int): max sequence length
            use_pad(bool): whether use padding(default: False)
            start_token: whether use start_token(default: False)
            end_token: whether use end_token before padding(default: False)

        Return:
            token indices(list)
        """

        idx = self.tokenize(text)

        if max_seq_len is None:
            max_seq_len = len(idx)

        if cls_token:
            idx = [self.token_to_idx(self._cls_token)] + idx

        if start_token:
            idx = [self.token_to_idx(self._start_token)] + idx

        if end_token:
            idx = idx + [self.token_to_idx(self._end_token)]

        if use_pad:
            if start_token or end_token:
                idx += [self.token_to_idx(self._pad_token)] * (
                    max_seq_len - (len(idx) + 1)
                )
            else:
                idx += [self.token_to_idx(self._pad_token)] * (max_seq_len - len(idx))

        return idx[:max_seq_len]

    def train(
        self,
        input_path: list,
        model_prefix: str,
        character_coverage=0.9995,
        vocab_size=None,
        model_type: str = "bpe",
        control_symbols: list = ["[PAD]", "[SEP]", "[MASK]", "[CLS]", "<s>", "</s>"],
    ):
        """
        Function for train tokenizer

        Args:
            input_path (str):
            model_prefix (str):
            character_coverage (float):
            vocab_size (float):
            model_type (str):
            control_symbols (list):
        """

        if character_coverage is None and vocab_size is None:
            print("at least character_coverage or vocab_size should be given!")
            assert character_coverage or vocab_size

        coverage_conditions = ""
        if character_coverage is not None:
            coverage_condition = f" --character_coverage={str(character_coverage)} "
        else:
            coverage_condition = f" --vocab_size={str(vocab_size)} "

        symbol_list = ""
        for i in control_symbols:
            symbol_list += i + ","
            
        input_list = ""
        for i in input_path:
            input_list += i + ","

        args = (
            "--input={} "
            "--model_prefix={} "
            "--model_type={} "
            "--control_symbols={} "
            "--bos_id=5 --eos_id=6 --unk_id=1".format(
                input_list, model_prefix, model_type, symbol_list
            )
            
        )

        args += coverage_condition
        
        print(args)

        SentencePieceTrainer.Train(args)

    def __repr__(self):
        unk = '"{}"'.format(self._unknown_token) if self._unknown_token else "None"
        return "Vocab(size={}, unk={}, pad={})".format(
            len(self.tok), unk, self._pad_token
        )

    def __len__(self):
        return len(self.tok)

