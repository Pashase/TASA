import torch
import torch.nn as nn

from vocab import Vocab


class TASA(nn.Module):
    def __init__(self, inp_voc, emb_size=64, hid_size=128, seq_len=100):
        """
        Complex LSTM model
        """
        super().__init__()

        self.inp_voc = inp_voc
        self.hid_size = hid_size

        self.emb_inp = nn.Embedding(len(inp_voc), emb_size)
        self.enc0 = nn.LSTM(emb_size, hid_size, batch_first=True)

        self.dec0 = nn.LSTMCell(emb_size, hid_size)
        self.logits = nn.Linear(hid_size, len(inp_voc))
        self.linear_influence = nn.Linear(seq_len, seq_len)
        self.sigmoid = nn.Sigmoid()

    def forward(self, sessions, taus):
        """ Apply model in training mode """
        inp_emb = self.emb_inp(sessions)
        deltas = self.sigmoid(self.linear_influence(taus))
        v_hats = inp_emb * deltas.unsqueeze(-1)
        last_state = self.encode(v_hats)

        return self.decode(last_state, sessions, taus)

    def encode(self, inp, **flags):
        """
        Takes symbolic input sequence, computes initial state
        :param inp: matrix of input tokens [batch, time]
        :returns: initial decoder state tensors, one or many
        """
        batch_size = inp.shape[0]
        output, last_state = self.enc0(inp)

        return [last_state]

    def decode_step(self, prev_state, prev_v_hats, **flags):
        """
        Takes previous decoder state and tokens, returns new state and logits for next tokens
        :param prev_state: a list of previous decoder state tensors, same as returned by encode(...)
        :param prev_v_hats: previous output tokens, an int vector of [batch_size]
        :return: a list of next decoder state tensors, a tensor of logits [batch, len(out_voc)]
        """
        prev_lstm0_state = prev_state[0]
        prev_lstm0_state = (prev_lstm0_state[0].squeeze(0), prev_lstm0_state[1].squeeze(0))

        new_lstm0_state = self.dec0(prev_v_hats, prev_lstm0_state)
        new_dec_state = [new_lstm0_state]
        output_logits = self.logits(new_lstm0_state[0])

        return new_dec_state, output_logits

    def decode(self, initial_state, sessions, taus, **flags):
        """ Iterate over reference tokens (out_tokens) with decode_step """

        batch_size = sessions.shape[0]
        state = initial_state

        logits_sequence = []
        S, T = sessions.clone(), taus.clone()
        S[:0] = inp_voc.bos_ix
        T[:0] = inp_voc.start_tau

        with torch.no_grad():
            inp_emb = self.emb_inp(S)
            deltas = self.sigmoid(self.linear_influence(T))
            v_hats = inp_emb * deltas.unsqueeze(-1)

        for i in range(sessions.shape[1]):
            state, logits = self.decode_step(state, v_hats[:, i])
            logits_sequence.append(logits)

        return torch.stack(logits_sequence, dim=1)

    def decode_inference(self, initial_state, max_len=100, **flags):
        """ Generate translations from model (greedy version) """

        batch_size, device = len(initial_state[0]), initial_state[0].device
        state = initial_state
        outputs = [torch.full([batch_size], self.out_voc.bos_ix, dtype=torch.int64,
                              device=device)]
        all_states = [initial_state]

        for i in range(max_len):
            state, logits = self.decode_step(state, outputs[-1])
            outputs.append(logits.argmax(dim=-1))
            all_states.append(state)

        return torch.stack(outputs, dim=1), all_states
