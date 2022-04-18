import sys
import numpy as np

import torch


class Vocab:
    def __init__(self, tokens, bos="_BOS_", eos="_EOS_", unk='_UNK_', start_tau=0, end_tau=1):
        """
        A special class that converts lines of tokens into matrices and backwards
        """

        assert all(tok in tokens for tok in (bos, eos, unk))

        self.tokens = tokens
        self.token_to_ix = {t: i for i, t in enumerate(tokens)}
        self.bos, self.eos, self.unk = bos, eos, unk

        self.bos_ix = self.token_to_ix[bos]
        self.eos_ix = self.token_to_ix[eos]
        self.unk_ix = self.token_to_ix[unk]

        self.start_tau = start_tau
        self.end_tau = end_tau

    def __len__(self):
        return len(self.tokens)

    @staticmethod
    def from_unique_activities(unique_activities, bos="_BOS_", eos="_EOS_", unk='_UNK_'):
        tokens = [bos, eos, unk] + unique_activities

        return Vocab(tokens, bos, eos, unk)

    def tokenize(self, session):
        """converts string to a list of tokens"""
        tokens = [activity if activity in self.token_to_ix else self.unk
                  for activity in session]

        return tokens + [self.eos]

    def to_matrix(self, sessions, taus, dtype=torch.int64, max_len=None):
        """
        convert variable length token sequences into  fixed size matrix
        example usage:
        >>>print(to_matrix(words[:3],source_to_ix))
        [[15 22 21 28 27 13 -1 -1 -1 -1 -1]
         [30 21 15 15 21 14 28 27 13 -1 -1]
         [25 37 31 34 21 20 37 21 28 19 13]]
        """
        sessions = list(map(self.tokenize, sessions))
        max_len = max_len or max(map(len, sessions))

        a_matrix = torch.full((len(sessions), max_len), self.eos_ix, dtype=dtype)
        t_matrix = torch.full((len(sessions), max_len), self.end_tau, dtype=torch.float)
        for i, seq in enumerate(sessions):
            row_ix = list(map(self.token_to_ix.get, seq[-max_len:]))
            a_matrix[i, :len(row_ix)] = torch.as_tensor(row_ix)
            t_matrix[i, :len(row_ix) - 1] = torch.as_tensor(taus[i][-max_len + 1:])

        return a_matrix, t_matrix

    def to_lines(self, matrix):
        """ Convert matrix of token (actions) ids into sequences """
        pass
