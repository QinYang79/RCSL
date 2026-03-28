"""COCO dataset loader"""
import torch
import torch.utils.data as data
import os
import numpy as np
import random
import nltk

import logging

import torch.nn.functional as F
logger = logging.getLogger(__name__)


class Img_dataset(data.Dataset):
    def __init__(self, images):
        self.img_length = len(images)
        self.images = images
    
    def __getitem__(self, img_id):
        return torch.Tensor(self.images[img_id]), img_id

    def __len__(self):
        return self.img_length

class Cap_dataset(data.Dataset):
    def __init__(self, captions, vocab):
        self.cap_length = len(captions)
        self.captions = captions
        self.vocab = vocab
    
    def __getitem__(self, cap_id):
        caption = self.captions[cap_id]
        tokens = nltk.tokenize.word_tokenize(caption.lower())
        caption = list()
        caption.append(self.vocab('<start>'))
        caption.extend([self.vocab(token) for token in tokens])
        caption.append(self.vocab('<end>'))
        target = torch.Tensor(caption)
        return target, cap_id

    def __len__(self):
        return self.cap_length

def collate_fn_img(data):
    images, img_ids = zip(*data)
    img_lengths = [len(image) for image in images]
    all_images = torch.zeros(len(images), max(img_lengths), images[0].size(-1))
    for i, image in enumerate(images):
        end = img_lengths[i]
        all_images[i, :end] = image[:end]
    img_lengths = torch.Tensor(img_lengths)
    return  all_images, img_lengths, list(img_ids)

def collate_fn_cap(data):
    captions, cap_ids = zip(*data)
    lengths = [len(cap) for cap in captions]
    targets = torch.zeros(len(captions), max(lengths)).long()
    for i, cap in enumerate(captions):
        end = lengths[i]
        targets[i, :end] = cap[:end]
    lengths = torch.Tensor(lengths)
    return  targets, lengths, list(cap_ids)


class PrecompDataset_gru(data.Dataset):
    """
    Load precomputed captions and image features
    Possible options: f30k_precomp, coco_precomp
    """

    def __init__(self,opt, data_path, data_split, vocab):
        self.opt = opt
        self.vocab = vocab
        self.init_txt = opt.init_txt
        self.caption_enhance = opt.caption_enhance
        self.img_enhance = opt.img_enhance
        self.data_split = data_split
        self.paired_length = opt.paired_length
        self.data_path = data_path

        loc = data_path + '/' 
        self.memory_bank = None
        # Captions
        self.captions = []
        with open(loc + '%s_caps.txt' % data_split, 'r', encoding="utf-8") as f:
            for line in f:
                self.captions.append(line.strip())

        # Image features
        self.images = np.load(loc + '%s_ims.npy' % data_split)
        self.img_length = self.images.shape[0]
        # rkiros data has redundancy in images, we divide by 5, 10crop doesn't
        self.length = len(self.captions)
        # self.img_length = len(self.images)
        print(f"{data_split} captions", self.length)
        print(f"{data_split} images", len(self.images))
        self.shuffle_inx = np.arange(self.img_length)

        self.old_length = self.length 
        if data_split == 'train' and self.paired_length > 0:
            if self.paired_length < -1:
                self.paired_length = self.length
 
            np.random.seed(123)
            inx = np.arange(self.img_length)
            np.random.shuffle(inx)
            noisy_inx = inx[int(self.paired_length):]  # paired_length < cap_len
            shuffle_inx = np.array(noisy_inx)
            np.random.shuffle(shuffle_inx)
            self.shuffle_inx[noisy_inx] = shuffle_inx
            print('eg', self.shuffle_inx[0:10])
            self.labels = []
            for i in range(self.length):
                if self.shuffle_inx[i//5] == i//5:
                    self.labels.append(1)
                else:
                    self.labels.append(0)
            print(f'paired {sum(self.labels)}')    
            np.random.seed(random.randint(1,1024))

        if self.images.shape[0] != self.length:
            self.im_div = 5
        else:
            self.im_div = 1
        # the development set for coco is large and so validation would be slow
        if data_split == 'dev':
            self.length = 5000

        
    def re_sort(self):
        self.re_idx = []
        if self.opt.stage != 'mining': 
            # Tips: Randomly select some samples to avoid being too sparse
            for i in range(self.old_length):
                if 'coco' in self.data_path:
                    if self.labels[i] == 0:
                        prob = random.random()
                        if prob>0.8:
                            self.re_idx.append(i)
                    else:
                        self.re_idx.append(i)
                else:
                    self.re_idx.append(i)
        else:
            self.re_idx = [i for i in range(self.old_length)] 

        self.length = len(self.re_idx)     
        print(f'{self.opt.stage} resort length: {self.length}')  

    def process_caption(self, caption, enhance=None, ag =0.2):
        if enhance is None: 
           enhance = self.caption_enhance if self.data_split == 'train' else False
        if not enhance:
            tokens = nltk.tokenize.word_tokenize(caption.lower())
            caption = list()
            caption.append(self.vocab('<start>'))
            caption.extend([self.vocab(token) for token in tokens])
            caption.append(self.vocab('<end>'))
            target = torch.Tensor(caption)
            return target
        else:
            # Convert caption (string) to word ids.
            tokens = ['<start>', ]
            tokens.extend(nltk.tokenize.word_tokenize(caption.lower()))
            tokens.append('<end>')
            deleted_idx = []
            for i, token in enumerate(tokens):
                prob = random.random()
                if prob < ag:
                    prob /= ag
                    # 50% randomly change token to mask token
                    if prob < 0.5:
                        if self.init_txt == 'glove':
                            # <mask> is replaced by <unk> when 'glove' is used for initialization, otherwise an error will occur！
                            tokens[i] = self.vocab.word2idx['<unk>']
                        else:
                            tokens[i] = self.vocab.word2idx['<mask>']
                    # 10% randomly change token to random token
                    elif prob < 0.8:
                        tokens[i] = random.randrange(len(self.vocab))
                    # 40% randomly remove the token
                    else:
                        tokens[i] = self.vocab(token)
                        deleted_idx.append(i)
                else:
                    tokens[i] = self.vocab(token)
            if len(deleted_idx) != 0:
                tokens = [tokens[i] for i in range(len(tokens)) if i not in deleted_idx]
            target = torch.Tensor(tokens)
            return target

    def process_image(self, image, enhance=None, ag=0.2):
        if enhance is None:
            enhance = self.img_enhance if self.data_split == 'train' else False
        if enhance:  # Size augmentation on region features.
            num_features = image.shape[0]
            rand_list = np.random.rand(num_features)
            tmp = image[np.where(rand_list > ag)]
            while tmp.size(1) <= 1:
                rand_list = np.random.rand(num_features)
                tmp = image[np.where(rand_list > ag)]
            return tmp
        else:
            return image

    def __getitem__(self, index):
        # handle the image redundancy
        if self.data_split == 'train':
            index = self.re_idx[index]
        img_id = self.shuffle_inx[index // self.im_div]
        image = torch.Tensor(self.images[img_id])
        caption = self.captions[index]
        # Convert caption (string) to word ids.
        if self.data_split == 'train':
            image_list, target_list= [], [] 
            image_list.append(self.process_image(image, True))
            target_list.append(self.process_caption(caption, True))   
            
            label = self.labels[index]   
            if self.opt.stage == 'mining':
                m_img = torch.Tensor(self.images[int(self.memory_bank['hard_t2i'][index][0])])
                m_txt = self.captions[int(self.memory_bank['hard_i2t'][img_id][0])]
                image_list.append(self.process_image(m_img,False))
                target_list.append(self.process_caption(m_txt,False))
 
            return image_list, target_list, index, img_id, label
        else:
            target = self.process_caption(caption)
            image = self.process_image(image)
            return image, target, index, img_id

    def __len__(self):
        return self.length

def collate_fn(data):
    def deal_fn_img(images):
        img_lengths = [len(image) for image in images]
        all_images = torch.zeros(len(images), max(img_lengths), images[0].size(-1))
        for i, image in enumerate(images):
            end = img_lengths[i]
            all_images[i, :end] = image[:end]
        img_lengths = torch.Tensor(img_lengths)
       
        return all_images, img_lengths
        

    def deal_fn_cap(captions):
        lengths = [len(cap) for cap in captions]
        targets = torch.zeros(len(captions), max(lengths)).long()
        for i, cap in enumerate(captions):
            end = lengths[i]
            targets[i, :end] = cap[:end]
        lengths = torch.Tensor(lengths)
        return targets, lengths
            
    if len(data[0])==5:
        data.sort(key=lambda x: len(x[1][0]), reverse=True)
        image_lists, caption_lists, ids, img_ids, labels = zip(*data)
        a,b,c,d = [],[],[],[]
        for i in range(len(image_lists[0])):
            images = [image_list[i] for image_list in list(image_lists)]
            a_,b_ = deal_fn_img(images)
            a.append(a_)
            b.append(b_) 
            
        for i in range(len(caption_lists[0])):
            captions = [caption_list[i] for caption_list in list(caption_lists)]
            c_,d_ = deal_fn_cap(captions)
            c.append(c_)
            d.append(d_)  
        return a,b,c,d, np.array(list(img_ids)), np.array(list(ids)), list(labels)
    else:
        data.sort(key=lambda x: len(x[1]), reverse=True)
        images, captions, ids, img_ids = zip(*data)
        a,b = deal_fn_img(images)
        c,d = deal_fn_cap(captions)
        return  a,b,c,d,list(img_ids),list(ids)

def get_loader(data_path, data_split, vocab_or_tokenizer, opt, batch_size=100,
               shuffle=True, num_workers=2, train=True):
    """Returns torch.utils.data.DataLoader for custom coco dataset."""
    if train:
        drop_last = False
    else:
        drop_last = False

    data_set = PrecompDataset_gru(opt, data_path, data_split, vocab_or_tokenizer)
    data_loader = torch.utils.data.DataLoader(dataset=data_set,
                                              batch_size=batch_size,
                                              shuffle=shuffle,
                                              pin_memory=False,
                                              collate_fn=collate_fn,
                                              num_workers=num_workers,
                                              drop_last=drop_last)

    return data_loader


def get_loaders(data_name, vocab_or_tokenizer, batch_size, workers, opt):
    # get the data path
    dpath = os.path.join(opt.data_path, data_name)
    train_loader = get_loader(dpath, 'train', vocab_or_tokenizer, opt,
                              batch_size, True, workers, train=True)
    val_loader = get_loader(dpath, 'dev', vocab_or_tokenizer, opt,
                            batch_size, False, workers, train=False)
    test_loader = get_loader(dpath, 'test', vocab_or_tokenizer, opt,
                             batch_size, False, workers, train=False)
    return train_loader, val_loader, test_loader


def get_test_loader(split_name, data_name, vocab_or_tokenizer, batch_size, workers, opt):
    dpath = os.path.join(opt.data_path, data_name)
    # get the test_loader
    test_loader = get_loader(dpath, split_name, vocab_or_tokenizer, opt,
                             batch_size, False, workers, train=False)
    return test_loader
