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
        # self.emb_out = nn.Embedding(len(out_voc), emb_size)
        self.enc0 = nn.LSTM(emb_size, hid_size, batch_first=True)

        # self.dec_start = nn.Linear(hid_size, hid_size)
        self.dec0 = nn.LSTMCell(emb_size, hid_size)
        self.logits = nn.Linear(hid_size, len(inp_voc))
        self.softmax = nn.Softmax()
        self.linear_influence = nn.Linear(seq_len, seq_len)
        self.sigmoid = nn.Sigmoid()

    def forward(self, sessions, taus):
        """ Apply model in training mode """
        inp_emb = self.emb_inp(sessions)
        # print(inp_emb.shape, taus.shape)
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

        #         enc_seq, [last_state_but_not_really] = self.enc0(inp_emb)
        #         # enc_seq: [batch, time, hid_size], last_state: [batch, hid_size]

        #         # note: last_state is not _actually_ last because of padding, let's find the real last_state
        #         lengths = (inp != self.inp_voc.eos_ix).to(torch.int64).sum(dim=1).clamp_max(inp.shape[1] - 1)
        #         last_state = enc_seq[torch.arange(len(enc_seq)), lengths]
        #         # ^-- shape: [batch_size, hid_size]

        #         dec_start = self.dec_start(last_state)
        output, last_state = self.enc0(inp)
        # return [dec_start]

        return [last_state]

    def decode_step(self, prev_state, prev_v_hats, **flags):
        """
        Takes previous decoder state and tokens, returns new state and logits for next tokens
        :param prev_state: a list of previous decoder state tensors, same as returned by encode(...)
        :param prev_v_hats: previous output tokens, an int vector of [batch_size]
        :return: a list of next decoder state tensors, a tensor of logits [batch, len(out_voc)]
        """
        prev_lstm0_state = prev_state[0]
        #         with torch.no_grad():
        #             inp_emb = self.emb_inp(prev_session)
        #             print(prev_session.shape, prev_taus.shape)
        #             deltas = self.sigmoid(self.linear_influence(prev_taus))
        #             v_hats = inp_emb * deltas.unsqueeze(-1)

        prev_lstm0_state = (prev_lstm0_state[0].squeeze(0), prev_lstm0_state[1].squeeze(0))
        # print(prev_v_hats.shape, prev_lstm0_state[0].shape, prev_lstm0_state[1].shape)

        new_lstm0_state = self.dec0(prev_v_hats, prev_lstm0_state)
        new_dec_state = [new_lstm0_state]
        output_logits = self.logits(new_lstm0_state[0])

        return new_dec_state, output_logits

    def decode(self, initial_state, sessions, taus, **flags):
        """ Iterate over reference tokens (out_tokens) with decode_step """

        batch_size = sessions.shape[0]
        state = initial_state

        # initial logits: always predict BOS
        #         onehot_bos = F.one_hot(torch.full([batch_size], self.out_voc.bos_ix, dtype=torch.int64),
        #                                num_classes=len(self.out_voc)).to(device=out_tokens.device)
        #         first_logits = torch.log(onehot_bos.to(torch.float32) + 1e-9)

        #         logits_sequence = [first_logits]

        logits_sequence = []
        S, T = sessions.clone(), taus.clone()
        S[:0] = inp_voc.bos_ix
        T[:0] = inp_voc.start_tau

        with torch.no_grad():
            inp_emb = self.emb_inp(S)
            deltas = self.sigmoid(self.linear_influence(T))
            v_hats = inp_emb * deltas.unsqueeze(-1)
        # print(v_hats.shape)

        for i in range(sessions.shape[1]):
            state, logits = self.decode_step(state, v_hats[:, i])
            logits_sequence.append(logits)

        return self.softmax(torch.stack(logits_sequence, dim=1))

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

    def translate_lines(self, inp_lines, **kwargs):
        # TODO
        inp = inp_voc.to_matrix(inp_lines).to(DEVICE)
        initial_state = self.encode(inp, **kwargs)
        out_ids, states = self.decode_inference(initial_state, **kwargs)

        return out_voc.to_lines(out_ids.cpu().numpy()), states
