import math
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import logging
import torchtext

logger = logging.getLogger(__name__)


def l2norm(X, dim, eps=1e-8):
    """L2-normalize columns of X
    """
    norm = torch.pow(X, 2).sum(dim=dim, keepdim=True).sqrt() + eps
    X = torch.div(X, norm)
    return X


def maxk_pool1d_var(x, dim, k, lengths):
    results = list()
    lengths = list(lengths.cpu().numpy())
    lengths = [int(x) for x in lengths]
    for idx, length in enumerate(lengths):
        k = min(k, length)
        max_k_i = maxk(x[idx, :length, :], dim - 1, k).mean(dim - 1)
        results.append(max_k_i)
    results = torch.stack(results, dim=0)
    return results


def maxk_pool1d(x, dim, k):
    max_k = maxk(x, dim, k)
    return max_k.mean(dim)


def maxk(x, dim, k):
    index = x.topk(k, dim=dim)[1]
    return x.gather(dim, index)


class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.output_dim = output_dim
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))
        self.bns = nn.ModuleList(nn.BatchNorm1d(k) for k in h + [output_dim])

    def forward(self, x):
        B, N, D = x.size()
        x = x.reshape(B * N, D)
        for i, (bn, layer) in enumerate(zip(self.bns, self.layers)):
            x = F.relu(bn(layer(x))) if i < self.num_layers - 1 else layer(x)
        x = x.view(B, N, self.output_dim)
        return x

def positional_encoding_1d(d_model, length):
    """
    :param d_model: dimension of the model
    :param length: length of positions
    :return: length*d_model position matrix
    """
    if d_model % 2 != 0:
        raise ValueError("Cannot use sin/cos positional encoding with "
                         "odd dim (got dim={:d})".format(d_model))
    pe = torch.zeros(length, d_model)
    position = torch.arange(0, length).unsqueeze(1)
    div_term = torch.exp((torch.arange(0, d_model, 2, dtype=torch.float) *
                          -(math.log(10000.0) / d_model)))
    pe[:, 0::2] = torch.sin(position.float() * div_term)
    pe[:, 1::2] = torch.cos(position.float() * div_term)

    return pe


class GPO(nn.Module):
    def __init__(self, d_pe, d_hidden):
        super(GPO, self).__init__()
        self.d_pe = d_pe
        self.d_hidden = d_hidden

        self.pe_database = {}
        self.gru = nn.GRU(self.d_pe, d_hidden, 1, batch_first=True, bidirectional=True)
        self.linear = nn.Linear(self.d_hidden, 1, bias=False)

    def compute_pool_weights(self, lengths, features):
        max_len = int(lengths.max())
        pe_max_len = self.get_pe(max_len)
        pes = pe_max_len.unsqueeze(0).repeat(lengths.size(0), 1, 1).to(lengths.device)
        mask = torch.arange(max_len).expand(lengths.size(0), max_len).to(lengths.device)
        mask = (mask < lengths.long().unsqueeze(1)).unsqueeze(-1)
        pes = pes.masked_fill(mask == 0, 0)

        self.gru.flatten_parameters()
        packed = pack_padded_sequence(pes, lengths.cpu(), batch_first=True, enforce_sorted=False)
        out, _ = self.gru(packed)
        padded = pad_packed_sequence(out, batch_first=True)
        out_emb, out_len = padded
        out_emb = (out_emb[:, :, :out_emb.size(2) // 2] + out_emb[:, :, out_emb.size(2) // 2:]) / 2
        scores = self.linear(out_emb)
        scores[torch.where(mask == 0)] = -10000

        weights = torch.softmax(scores / 0.1, 1)
        return weights, mask

    def forward(self, features, lengths):
        """
        :param features: features with shape B x K x D
        :param lengths: B x 1, specify the length of each data sample.
        :return: pooled feature with shape B x D
        """
        pool_weights, mask = self.compute_pool_weights(lengths, features)

        features = features[:, :int(lengths.max()), :]
        sorted_features = features.masked_fill(mask == 0, -10000)
        sorted_features = sorted_features.sort(dim=1, descending=True)[0]
        sorted_features = sorted_features.masked_fill(mask == 0, 0)

        pooled_features = (sorted_features * pool_weights).sum(1)
        return pooled_features, pool_weights

    def get_pe(self, length):
        """

        :param length: the length of the sequence
        :return: the positional encoding of the given length
        """
        length = int(length)
        if length in self.pe_database:
            return self.pe_database[length]
        else:
            pe = positional_encoding_1d(self.d_pe, length)
            self.pe_database[length] = pe
            return pe

def get_text_encoder(opt,word2idx):
 
    return EncoderText_gru(opt.vocab_size, opt.embed_size, opt.word_dim, opt.num_layers,
                                use_bi_gru=opt.use_bi_gru,
                                no_txtnorm=opt.no_txtnorm,word2idx=word2idx)


def get_image_encoder(opt):
    return EncoderImageAggr(opt.img_dim, opt.embed_size, opt.no_imgnorm)

def get_padding_mask(lens):
    """
    :param lens: length of the sequence
    :return: 
    """
    batch = lens.shape[0]
    max_l = int(lens.max())
    mask = torch.arange(max_l).expand(batch, max_l).to(lens.device)
    mask = (mask>=lens.long().unsqueeze(dim=1)).to(lens.device)
    return mask
 

class EncoderImageAggr(nn.Module):
    def __init__(self, img_dim, embed_size, no_imgnorm=False):
        super(EncoderImageAggr, self).__init__()
        self.embed_size = embed_size
        self.no_imgnorm = no_imgnorm
        self.fc = nn.Linear(img_dim, embed_size)
        self.mlp = MLP(img_dim, embed_size // 2, embed_size, 2)
        self.linear1 = nn.Linear(embed_size, embed_size)
        
        self.dropout = nn.Dropout(0.1)
        self.init_weights()
     
  
    def forward(self, images, image_lengths):
        """Extract image feature vectors."""
        # images = self.dropout(images)
        features = self.fc(images)
        features = self.mlp(images) + features

        img_emb= features

        features_external = self.linear1(features) 
        max_len = int(image_lengths.max())
        mask = torch.arange(max_len).expand(image_lengths.size(0), max_len).to(image_lengths.device)
        mask = (mask < image_lengths.long().unsqueeze(1)).unsqueeze(-1)

        features_external = features_external.masked_fill(mask == 0,-10000)
        features_k_softmax = nn.Softmax(dim=1)(features_external-torch.max(features_external,dim=1)[0].unsqueeze(1))
        
        # features_k_softmax = nn.Softmax(dim=1)(features_external)
        attn = features_k_softmax.masked_fill(mask == 0,0)
        feature_img = torch.sum(attn * img_emb,dim=1)

        if not self.no_imgnorm:
            feature_img = l2norm(feature_img, dim=-1)
 
        return feature_img


    def init_weights(self):
        """Xavier initialization for the fully connected layer
        """
        for m in self.children():
            if isinstance(m, nn.Linear):
                r = np.sqrt(6.) / np.sqrt(m.in_features + m.out_features)
                m.weight.data.uniform_(-r, r)
                m.bias.data.fill_(0)
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


class EncoderText_gru(nn.Module):
    def __init__(self, vocab_size, embed_size, word_dim, num_layers, use_bi_gru=True, no_txtnorm=False,word2idx=None):
        super(EncoderText_gru, self).__init__()
        self.embed_size = embed_size
        self.no_txtnorm = no_txtnorm
        # word embedding
        self.embed = nn.Embedding(vocab_size, word_dim)
        # caption embedding 
        self.rnn = nn.GRU(word_dim, embed_size, num_layers, batch_first=True, bidirectional=use_bi_gru)
        self.init_weights(word2idx)
        self.linear1 = nn.Linear(embed_size, embed_size)
        self.dropout = nn.Dropout(0.1)
        self.init_weights(word2idx)
 

    def init_weights(self, word2idx):
        if word2idx is None:
            self.embed.weight.data.uniform_(-0.1, 0.1)
        else:
            path = os.path.join('/sde1/qinyang/projects/Cross-modal/TCL/vocab', 'vector_cache')
            print(path)
            wemb = torchtext.vocab.GloVe(cache=path)

            # quick-and-dirty trick to improve word-hit rate
            missing_words = []
            for word, idx in word2idx.items():
                if word not in wemb.stoi:
                    word = word.replace('-', '').replace('.', '').replace("'", '')
                    if '/' in word:
                        word = word.split('/')[0]
                if word in wemb.stoi:
                    self.embed.weight.data[idx] = wemb.vectors[wemb.stoi[word]]
                else:
                    missing_words.append(word)

            # self.embed.requires_grad = False
            print('Words: {}/{} found in vocabulary; {} words missing'.format(
                len(word2idx) - len(missing_words), len(word2idx), len(missing_words)))

 
    def forward(self, x, lengths, train=False):
        """Handles variable size captions
        """
        x_emb = self.embed(x) 
        if train:
            x_emb = self.dropout(x_emb) 
        lengths = lengths.clamp(min=1)#
        sorted_seq_lengths, indices = torch.sort(lengths, descending=True)
        _, desorted_indices = torch.sort(indices, descending=False)#

        self.rnn.flatten_parameters()
        x_emb_rnn=x_emb[indices]
        packed = pack_padded_sequence(x_emb_rnn, sorted_seq_lengths.cpu(), batch_first=True,enforce_sorted=True)

        # Forward propagate RNN
        out, _ = self.rnn(packed)

        # Reshape *final* output to (batch_size, hidden_size)
        cap_emb_rnn, cap_len = pad_packed_sequence(out, batch_first=True)
        cap_emb = cap_emb_rnn[desorted_indices]
        
        cap_emb = (cap_emb[:, :, :cap_emb.size(2) // 2] + cap_emb[:, :, cap_emb.size(2) // 2:]) / 2

        max_len = int(lengths.max())
        mask = torch.arange(max_len).expand(lengths.size(0), max_len).to(lengths.device)
        mask = (mask < lengths.long().unsqueeze(1)).unsqueeze(-1)
       
        cap_emb = cap_emb[:, :int(lengths.max()), :] 
        cap_external = self.linear1(cap_emb) 
        cap_external = cap_external.masked_fill(mask == 0,-10000) 
        attn = nn.Softmax(dim=1)(cap_external-torch.max(cap_external,dim=1)[0].unsqueeze(1))
     
        attn = attn.masked_fill(mask == 0,0)
        feature_cap = torch.sum(attn * cap_emb,dim=1)

        # normalization in the joint embedding space
        if not self.no_txtnorm:
            feature_cap = l2norm(feature_cap, dim=-1)

        return feature_cap