__author__ = 'Dung Doan'

import torch
import torch.nn as nn
import torch.nn.functional as F
from models.linear import BiLinear
from models.attention import BiAAttention
import numpy as np
from models.variational_rnn import VarMaskedRNN, VarMaskedLSTM, VarMaskedFastLSTM, VarMaskedGRU

class BiRecurrentConvBiAffine(nn.Module):
    def __init__(self, word_dim, num_words, char_dim, num_chars, pos_dim, num_pos, num_filters, kernel_size, rnn_mode, hidden_size, num_layers, num_labels, arc_space, type_space,
                 embedd_word=None, embedd_char=None, embedd_pos=None, p_in=0.33, p_out=0.33, p_rnn=(0.33, 0.33), biaffine=True, pos=True, char=True):
        super(BiRecurrentConvBiAffine, self).__init__()

        self.word_embedd = nn.Embedding(num_words, word_dim, _weight=embedd_word)
        self.pos_embedd = nn.Embedding(num_pos, pos_dim, _weight=embedd_pos) if pos else None
        self.char_embedd = nn.Embedding(num_chars, char_dim, _weight=embedd_char) if char else None
        self.conv1d = nn.Conv1d(char_dim, num_filters, kernel_size, padding=kernel_size - 1) if char else None
        self.dropout_in = nn.Dropout2d(p=p_in)
        self.dropout_out = nn.Dropout2d(p=p_out)
        self.num_labels = num_labels
        self.pos = pos
        self.char = char

        if rnn_mode == 'RNN':
            RNN = VarMaskedRNN
        elif rnn_mode == 'LSTM':
            RNN = VarMaskedLSTM
        elif rnn_mode == 'FastLSTM':
            RNN = VarMaskedFastLSTM
        elif rnn_mode == 'GRU':
            RNN = VarMaskedGRU
        else:
            raise ValueError('Unknown RNN mode: %s' % rnn_mode)

        dim_enc = word_dim
        if pos:
            dim_enc += pos_dim
        if char:
            dim_enc += num_filters

        self.rnn = RNN(dim_enc, hidden_size, num_layers=num_layers, batch_first=True, bidirectional=True, dropout=p_rnn)

        out_dim = hidden_size * 2

        self.arc_h = nn.Linear(out_dim, arc_space)
        self.arc_c = nn.Linear(out_dim, arc_space)
        self.attention = BiAAttention(arc_space, arc_space, 1, biaffine=biaffine)

        self.type_h = nn.Linear(out_dim, type_space)
        self.type_c = nn.Linear(out_dim, type_space)
        self.bilinear = BiLinear(type_space, type_space, self.num_labels)

    def _get_rnn_output(self, input_word, input_char, input_pos, mask=None, length=None, hx=None):
        word = self.word_embedd(input_word)

        word = self.dropout_in(word)

        input = word

        if self.char:
            char = self.char_embedd(input_char)
            char_size = char.size()

            char = char.view(char_size[0]*char_size[1], char_size[2], char_size[3]).transpose(1,2)

            char, _ = self.conv1d(char).max(dim=2)

            char = torch.tanh(char).view(char_size[0], char_size[1], -1)

            char = self.dropout_in(char)

            input = torch.cat([input, char], dim=2)

        if self.pos:
            pos = self.pos_embedd(input_pos)

            pos = self.dropout_in(pos)
            input = torch.cat([input, pos], dim=2)

        # output, hn = self.rnn(input, mask, hx=hx)
        output, hn = self.rnn(input, mask)

        output = self.dropout_out(output.transpose(1,2)).transpose(1,2)

        arc_h = F.elu(self.arc_h(output))
        arc_c = F.elu(self.arc_c(output))

        type_h = F.elu(self.type_h(output))
        type_c = F.elu(self.type_c(output))

        arc = torch.cat([arc_h, arc_c], dim=1)
        type = torch.cat([type_h, type_c], dim=1)

        arc = self.dropout_out(arc.transpose(1, 2)).transpose(1, 2)
        arc_h, arc_c = arc.chunk(2, 1)

        type = self.dropout_out(type.transpose(1, 2)).transpose(1, 2)
        type_h, type_c = type.chunk(2, 1)
        type_h = type_h.contiguous()
        type_c = type_c.contiguous()

        return (arc_h, arc_c), (type_h, type_c), hn, mask, length

    def forward(self, input_word, input_char, input_pos, mask=None, length=None, hx=None):
        arc, type, _, mask, length = self._get_rnn_output(input_word, input_char, input_pos, mask=mask, length=length, hx=hx)

        out_arc = self.attention(arc[0], arc[1], mask_d=mask, mask_e=mask).squeeze(dim=1)
        return out_arc, type, mask, length

    def _decode_types(self, out_type, heads, leading_symbolic):
        type_h, type_c = out_type
        batch, max_len, _ = type_h.size()

        batch_index = torch.arange(0, batch).type_as(type_h.data).long()

        type_h = type_h[batch_index, heads.t()].transpose(0, 1).contiguous()

        out_type = self.bilinear(type_h, type_c)

        out_type = out_type[:, :, leading_symbolic:]

        _, types = out_type.max(dim=2)

        return types + leading_symbolic

    def decode(self, input_word, input_char, input_pos, mask=None, length=None, hx=None, leading_symbolic=0):
        out_arc, out_type, mask, length = self.forward(input_word, input_char, input_pos, mask=mask, length=length, hx=hx)
        out_arc = out_arc.data
        batch, max_len, _ = out_arc.size()

        out_arc = out_arc + torch.diag(out_arc.new(max_len).fill_(-np.inf))

        if mask is not None:
            minus_mask = (1 - mask.data).byte().unsqueeze(2)
            out_arc.masked_fill_(minus_mask, -np.inf)

        _, heads = out_arc.max(dim=1)

        types = self._decode_types(out_type, heads, leading_symbolic)

        return heads.cpu().numpy(), types.data.cpu().numpy()

    def decode_chuLiuEdmonds(self, energies, lengths, leading_symbolic=0, labeled=True):

        def find_cycle(par):
            added = np.zeros([length], np.bool)
            added[0] = True
            cycle = set()
            findcycle = False
            for i in range(1, length):
                if findcycle:
                    break

                if added[i] or not curr_nodes[i]:
                    continue

                tmp_cycle = set()
                tmp_cycle.add(i)
                added[i] = True
                findcycle = True
                l = i

                while par[l] not in tmp_cycle:
                    l = par[l]
                    if added[l]:
                        findcycle = False
                        break
                    added[l] = True
                    tmp_cycle.add(l)

                if findcycle:
                    lorg = l
                    cycle.add(lorg)
                    l = par[lorg]
                    while l != lorg:
                        cycle.add(l)
                        l = par[l]
                    break

            return findcycle, cycle

        def chuLiuEdmonds():
            par = np.zeros([length], dtype=np.int32)

            par[0] = -1
            for i in range(1, length):
                if curr_nodes[i]:
                    max_score = score_matrix[0, i]
                    par[i] = 0
                    for j in range(1, length):
                        if j == i or not curr_nodes[j]:
                            continue

                        new_score = score_matrix[j, i]
                        if new_score > max_score:
                            max_score = new_score
                            par[i] = j

            findcycle, cycle = find_cycle(par)
            if not findcycle:
                final_edges[0] = -1
                for i in range(1, length):
                    if not curr_nodes[i]:
                        continue

                    pr = oldI[par[i], i]
                    ch = oldO[par[i], i]
                    final_edges[ch] = pr
                return

            cyc_len = len(cycle)
            cyc_weight = 0.0
            cyc_nodes = np.zeros([cyc_len], dtype=np.int32)
            id = 0
            for cyc_node in cycle:
                cyc_nodes[id] = cyc_node
                id += 1
                cyc_weight += score_matrix[par[cyc_node], cyc_node]

            rep = cyc_nodes[0]
            for i in range(length):
                if not curr_nodes[i] or i in cycle:
                    continue

                max1 = float("-inf")
                wh1 = -1
                max2 = float("-inf")
                wh2 = -1

                for j in range(cyc_len):
                    j1 = cyc_nodes[j]
                    if score_matrix[j1, i] > max1:
                        max1 = score_matrix[j1, i]
                        wh1 = j1

                    scr = cyc_weight + score_matrix[i, j1] - score_matrix[par[j1], j1]

                    if scr > max2:
                        max2 = scr
                        wh2 = j1

                score_matrix[rep, i] = max1
                oldI[rep, i] = oldI[wh1, i]
                oldO[rep, i] = oldO[wh1, i]
                score_matrix[i, rep] = max2
                oldO[i, rep] = oldO[i, wh2]
                oldI[i, rep] = oldI[i, wh2]

            rep_cons = []
            for i in range(cyc_len):
                rep_cons.append(set())
                cyc_node = cyc_nodes[i]
                for cc in reps[cyc_node]:
                    rep_cons[i].add(cc)

            for i in range(1, cyc_len):
                cyc_node = cyc_nodes[i]
                curr_nodes[cyc_node] = False
                for cc in reps[cyc_node]:
                    reps[rep].add(cc)

            chuLiuEdmonds()

            found = False
            wh = -1
            for i in range(cyc_len):
                for repc in rep_cons[i]:
                    if repc in final_edges:
                        wh = cyc_nodes[i]
                        found = True
                        break
                if found:
                    break

            l = par[wh]
            while l != wh:
                ch = oldO[par[l], l]
                pr = oldI[par[l], l]
                final_edges[ch] = pr
                l = par[l]

        if labeled:
            assert energies.ndim == 4
        else:
            assert energies.ndim == 3
        input_shape = energies.shape
        batch_size = input_shape[0]
        max_length = input_shape[2]

        pars = np.zeros([batch_size, max_length], dtype=np.int32)
        types = np.zeros([batch_size, max_length], dtype=np.int32) if labeled else None

        for i in range(batch_size):
            energy = energies[i]

            length = lengths[i]

            if labeled:
                energy = energy[leading_symbolic:, :length, :length]
                label_id_matrix = energy.argmax(axis=0) + leading_symbolic
                energy = energy.max(axis=0)
            else:
                energy = energy[:length, :length]
                label_id_matrix = None

            orig_score_matrix = energy
            score_matrix = np.array(orig_score_matrix, copy=True)

            oldI = np.zeros([length, length], dtype=np.int32)
            oldO = np.zeros([length, length], dtype=np.int32)
            curr_nodes = np.zeros([length], dtype=np.bool)
            reps = []

            for s in range(length):
                orig_score_matrix[s, s] = 0.0
                score_matrix[s, s] = 0.0
                curr_nodes[s] = True
                reps.append(set())
                reps[s].add(s)
                for t in range(s + 1, length):
                    oldI[s, t] = s
                    oldO[s, t] = t

                    oldI[t, s] = t
                    oldO[t, s] = s

            final_edges = dict()
            chuLiuEdmonds()
            par = np.zeros([max_length], np.int32)
            if labeled:
                type = np.ones([max_length], np.int32)
                type[0] = 0
            else:
                type = None

            for ch, pr in final_edges.items():
                par[ch] = pr
                if labeled and ch != 0:
                    type[ch] = label_id_matrix[pr, ch]

            par[0] = 0
            pars[i] = par
            if labeled:
                types[i] = type

        return pars, types


    def decode_mst(self, input_word, input_char, input_pos, mask=None, length=None, hx=None, leading_symbolic=0):
        out_arc, out_type, mask, length = self.forward(input_word, input_char, input_pos, mask=mask, length=length, hx=hx)

        type_h, type_c = out_type
        batch, max_len, type_space = type_h.size()

        if length is None:
            if mask is None:
                length = [max_len for _ in range(batch)]
            else:
                length = mask.data.sum(dim=1).long().cpu().numpy()

        type_h = type_h.unsqueeze(2).expand(batch, max_len, max_len, type_space).contiguous()
        type_c = type_c.unsqueeze(1).expand(batch, max_len, max_len, type_space).contiguous()

        out_type = self.bilinear(type_h, type_c)

        if mask is not None:
            minus_inf = -1e8
            minus_mask = (1 - mask) * minus_inf
            out_arc = out_arc + minus_mask.unsqueeze(2) + minus_mask.unsqueeze(1)

        loss_arc = F.log_softmax(out_arc, dim=1)
        loss_type = F.log_softmax(out_type, dim=3).permute(0, 3, 1, 2)

        energy = torch.exp(loss_arc.unsqueeze(1) + loss_type)

        return self.decode_chuLiuEdmonds(energy.data.cpu().numpy(), length, leading_symbolic=leading_symbolic, labeled=True)


    def loss(self, input_word, input_char, input_pos, heads, types, mask=None, length=None, hx=None):
        out_arc, out_type, mask, length = self.forward(input_word, input_char, input_pos, mask=mask, length=length, hx=hx)
        batch, max_len, _ = out_arc.size()

        if length is not None and heads.size(1) != mask.size(1):
            heads = heads[:, :max_len]
            types = types[:, :max_len]

        type_h, type_c = out_type

        batch_index = torch.arange(0, batch).type_as(out_arc.data).long()

        type_h = type_h[batch_index, heads.data.t()].transpose(0, 1).contiguous()

        out_type = self.bilinear(type_h, type_c)

        if mask is not None:
            minus_inf = 1e-8
            minus_mask = (1 - mask) * minus_inf
            out_arc = out_arc + minus_mask.unsqueeze(2) + minus_mask.unsqueeze(1)

        loss_arc = F.log_softmax(out_arc, dim=1)

        loss_type = F.log_softmax(out_type, dim=2)

        if mask is not None:
            loss_arc = loss_arc * mask.unsqueeze(2) * mask.unsqueeze(1)
            loss_type = loss_type * mask.unsqueeze(2)

            num = mask.sum() - batch
        else:
            num = float(max_len - 1) * batch

        child_index = torch.arange(0, max_len).view(max_len, 1).expand(max_len, batch)
        child_index = child_index.type_as(out_arc.data).long()

        loss_arc = loss_arc[batch_index, heads.data.t(), child_index][1:]
        loss_type = loss_type[batch_index, child_index, types.data.t()][1:]

        return -loss_arc.sum()/num, -loss_type.sum()/num


def is_uni_punctuation(word):
    # match = re.match("^[^\w\s]+$", word.decode('utf-8', flags=re.UNICODE))
    # return match is not None
    return False


def is_punctuation(word, pos, punc_set=None):
    if punc_set is None:
        return is_uni_punctuation(word)
    else:
        return pos in punc_set


def eval(words, postags, heads_pred, types_pred, heads, types, word_alphabet, pos_alphabet, lengths, punct_set=None, symbolic_root=False, symbolic_end=False):
    batch_size, _ = words.shape
    ucorr = 0.
    lcorr = 0.
    total = 0.
    ucomplete_match = 0.
    lcomplete_match = 0.

    ucorr_nopunc = 0.
    lcorr_nopunc = 0.
    total_nopunc = 0.
    ucomplete_match_nopunc = 0.
    lcomplete_match_nopunc = 0.

    corr_root = 0.
    total_root = 0.
    start = 1 if symbolic_root else 0
    end = 1 if symbolic_end else 0
    for i in range(batch_size):
        ucm = 1.
        lcm = 1.
        ucm_nopunc = 1.
        lcm_nopunc = 1.
        for j in range(start, lengths[i] - end):
            word = word_alphabet.get_instance(words[i, j])
            # word = word.encode('utf8')

            pos = pos_alphabet.get_instance(postags[i, j])
            # pos = pos.encode('utf8')

            total += 1
            if heads[i, j] == heads_pred[i, j]:
                ucorr += 1
                if types[i, j] == types_pred[i, j]:
                    lcorr += 1
                else:
                    lcm = 0
            else:
                ucm = 0
                lcm = 0

            if not is_punctuation(word, pos, punct_set):
                total_nopunc += 1
                if heads[i, j] == heads_pred[i, j]:
                    ucorr_nopunc += 1
                    if types[i, j] == types_pred[i, j]:
                        lcorr_nopunc += 1
                    else:
                        lcm_nopunc = 0
                else:
                    ucm_nopunc = 0
                    lcm_nopunc = 0

            if heads[i, j] == 0:
                total_root += 1
                corr_root += 1 if heads_pred[i, j] == 0 else 0

        ucomplete_match += ucm
        lcomplete_match += lcm
        ucomplete_match_nopunc += ucm_nopunc
        lcomplete_match_nopunc += lcm_nopunc

    return (ucorr, lcorr, total, ucomplete_match, lcomplete_match), \
           (ucorr_nopunc, lcorr_nopunc, total_nopunc, ucomplete_match_nopunc, lcomplete_match_nopunc), \
           (corr_root, total_root), batch_size

