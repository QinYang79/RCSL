import torch
import torch.nn as nn
import torch.nn.init
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
from torch.nn.utils import clip_grad_norm_
import numpy as np
from encoders  import get_image_encoder, get_text_encoder
import logging
logger = logging.getLogger(__name__)
import torch.nn.functional as F

def get_sim(a, b):
    sims = a.mm(b.t()) # have normed
    return sims

def l2norm(X, dim, eps=1e-8):
    """L2-normalize columns of X
    """
    norm = torch.pow(X, 2).sum(dim=dim, keepdim=True).sqrt() + eps
    X = torch.div(X, norm)
    return X
def l2norm_3d(X):
    """L2-normalize columns of X
    """
    norm = torch.pow(X, 2).sum(dim=2, keepdim=True).sqrt()
    X = torch.div(X, norm)
    return X


def neg_loss(scores,paired_l,th):
    selelct_ = (scores.diag()>th).data.cpu().numpy().tolist()
    mask_diag = torch.zeros_like(scores.diag()).to(scores.device)
    for i, v in enumerate(selelct_):
        if v and i >= paired_l:
            mask_diag[i] = 1

    no_eye = 1 - torch.eye(scores.size(0)).cuda()
    eps = 1e-20
    scores = torch.exp(scores/0.07)
    p1 = scores / torch.sum(scores, dim=1, keepdim=True) + eps
    p2 = scores.t() / torch.sum(scores.t(), dim=1, keepdim=True) + eps
    loss_neg = - torch.log(1 - (p1*no_eye)).sum(1) 
    loss_neg += - torch.log(1 - (p2*no_eye)).sum(1)

    return  (loss_neg * mask_diag).sum()

class ContrastiveLoss(nn.Module):
    """
    Compute contrastive loss (max-margin based)
    """

    def __init__(self, opt, margin=0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin
        self.beta = 0

    def align(self, d):
        return d.diag().mean()

    def uniform(self, d):
        d = d - torch.eye(d.shape[0]).to(d.device) * d
        return d.mul(-1).exp().mean()

    def forward(self, scores, paired_l):
 
        diagonal = scores.diag().view(scores.size(0), 1)
        d1 = diagonal.expand_as(scores)
        d2 = diagonal.t().expand_as(scores)
        scores_ = torch.ones_like(scores).mul(-1)
        scores_[:paired_l,:paired_l]  = scores[:paired_l,:paired_l] 
        scores = scores_

        cost_s = (self.margin + scores - d1).clamp(min=0)
        cost_im = (self.margin + scores - d2).clamp(min=0)

        mask = torch.eye(scores.size(0),scores.size(1)) > .5
        I = Variable(mask)
        if torch.cuda.is_available():
            I = I.cuda()
        cost_s = cost_s.masked_fill_(I, 0)
        cost_im = cost_im.masked_fill_(I, 0)

        cost_s = cost_s.max(1)[0]
        cost_im = cost_im.max(0)[0]

        return cost_s[0:paired_l].sum() + cost_im[0:paired_l].sum()



class ContrastiveLoss_v1(nn.Module):
    """
    Compute contrastive loss (max-margin based)
    """

    def __init__(self, opt, margin=0):
        super(ContrastiveLoss_v1, self).__init__()
        self.margin = margin
        self.beta = opt.beta

    def forward(self, scores, scores_txt, scores_img, paired_l):
        beta = 1
        scores_txt[scores_txt>beta] = 0
        scores_txt[scores_txt<=beta] = 1
    
        scores_img[scores_img>beta] = 0
        scores_img[scores_img<=beta] = 1
        
        scores1 = scores * scores_txt
        scores2 = scores * scores_img
        
        cost_s = scores1.max(1)[0]
        cost_im = scores2.max(0)[0]

        cost_s = (self.margin + scores.diag() - cost_s).clamp(min=0)
        cost_im = (self.margin + scores.diag() - cost_im).clamp(min=0)
 
        return cost_s[0:paired_l].sum() + cost_im[0:paired_l].sum()
    
class SVSE(nn.Module):
    def __init__(self, opt,word2idx):
        super(SVSE, self).__init__()
        self.opt = opt
        self.parallel = False
        # based models
        self.img_enc = get_image_encoder(opt)
        self.txt_enc = get_text_encoder(opt,word2idx)

        self.step = 0
        self.logger = None
       
        if torch.cuda.is_available():
            self.img_enc.cuda()
            self.txt_enc.cuda()
            cudnn.benchmark = True

        params = list(self.img_enc.parameters())
        params += list(self.txt_enc.parameters())
        self.params = params
        self.optimizer = torch.optim.AdamW(self.params, lr=opt.learning_rate)

        self.memory_bank = None
        self.criterion = ContrastiveLoss(opt, 0.2)
        # self.criterion_v1 = ContrastiveLoss_v1(opt, 0.2)
        
    def reinit_optimizer(self):
        self.optimizer = torch.optim.AdamW(self.params, lr=self.opt.learning_rate)

    def make_data_parallel(self):
        self.img_enc = nn.DataParallel(self.img_enc)
        self.txt_enc = nn.DataParallel(self.txt_enc)
        self.parallel = True

    def reinit_optimizer(self):
        self.optimizer = torch.optim.AdamW(self.params, lr=self.opt.learning_rate)

    def train_start(self):
        """switch to train mode
        """
        self.img_enc.train()
        self.txt_enc.train()
        

    def val_start(self):
        """switch to evaluate mode
        """
        self.img_enc.eval()
        self.txt_enc.eval()

    def state_dict(self):
        state_dict = [self.img_enc.state_dict(), self.txt_enc.state_dict()]
        return state_dict

    def load_state_dict(self, state_dict):
        self.img_enc.load_state_dict(state_dict[0], strict=False)
        self.txt_enc.load_state_dict(state_dict[1], strict=False)

    def forward_emb(self, images, captions, caption_lengths, image_lengths,train=False):
        if torch.cuda.is_available():
            images = images.cuda()
            captions = captions.cuda()
            caption_lengths = caption_lengths.cuda()
            image_lengths = image_lengths.cuda()
        img_embs = self.img_enc(images, image_lengths)
        cap_embs = self.txt_enc(captions, caption_lengths,train=train)
        return img_embs, cap_embs
        
    def forward_imgs(self, images,image_lengths):
        if torch.cuda.is_available():
            images = images.cuda()
            image_lengths = image_lengths.cuda()
        img_embs = self.img_enc(images, image_lengths)
        return img_embs
    
    def forward_caps(self,captions, caption_lengths):
        if torch.cuda.is_available():
            captions = captions.cuda()
            caption_lengths = caption_lengths.cuda()
        cap_embs = self.txt_enc(captions, caption_lengths)
        return cap_embs
    
    def robust_mining_loss(self, scores, tau = 0.02):

        p1 = F.softmax(scores/tau, dim=1)
        p2 = F.softmax(scores.t()/tau, dim=1)

        loss1 =   (1- p1.diag()) 
        loss2 =   (1- p2.diag())
        loss = loss1.sum() + loss2.sum()
        return loss/2

    def train_emb(self, images, captions, image_lengths=None, caption_lengths=None, img_ids=None, cap_ids=None, labels=None, epoch=None):
        self.step += 1
        self.logger.update('Step', self.step)
        self.logger.update('lr', self.optimizer.param_groups[0]['lr'])
        tau = self.opt.tau
        b_l = len(labels)
        # ranking
        paired_l = sum(labels) 
        tindex = list(np.where(np.array(labels) == 1)[0])
        findex = list(np.where(np.array(labels) == 0)[0])
        sort_index = tindex+findex

        # 64 imgs 64 text -> 2     64 text 64 img
        img_ids = np.concatenate((img_ids[tindex],img_ids[findex]))
        cap_ids = np.concatenate((cap_ids[tindex],cap_ids[findex]))
        ag_times = len(images)
 
        # compute embs
        img_embs_dict, cap_embs_dict = dict(),dict()
        
        for i in range(ag_times):
            img_embs, cap_embs = self.forward_emb(images[i], captions[i], caption_lengths[i],  image_lengths[i],train=True)
            img_embs_dict[i] = img_embs[sort_index]
            cap_embs_dict[i] = cap_embs[sort_index]
    
        loss1 = torch.tensor(0.).cuda()
        loss3 = torch.tensor(0.).cuda() 
        if paired_l > 0:
            sims = img_embs_dict[0]@ cap_embs_dict[0].t()
            loss1 =  self.criterion(sims, paired_l)  

            def lalign(x, y, alpha=2):
                return (x - y).norm(dim=1).pow(alpha)[0:paired_l].mean()
            def lunif(x, t=2):
                sq_pdist = torch.pdist(x, p=2).pow(2)
                return sq_pdist.mul(-t).exp().mean().log()
        
            loss3 +=  lalign(img_embs_dict[0], cap_embs_dict[0])  +  (lunif(img_embs_dict[0]) + lunif(cap_embs_dict[0])) / 2
      
        loss2 = torch.tensor(0.).cuda()
        if self.opt.stage == 'mining':     
            p_up_txt = torch.cat([cap_embs_dict[0][:paired_l], cap_embs_dict[1][paired_l:]],dim=0)
            p_up_img = torch.cat([img_embs_dict[0][:paired_l], img_embs_dict[1][paired_l:]],dim=0)
            
            sims1 = p_up_img @ cap_embs_dict[0].t()
            sims2 = p_up_txt @ img_embs_dict[0].t() 
     
            loss2 += self.robust_mining_loss(sims1,  tau=tau) 
            loss2 += self.robust_mining_loss(sims2,  tau=tau)
        
        loss = loss1 + loss2 + loss3

        if loss != 0:
            self.optimizer.zero_grad()
            loss.backward()
            if self.opt.grad_clip > 0:
                clip_grad_norm_(self.params, self.opt.grad_clip)
            self.optimizer.step()
        
            self.logger.update('pl', paired_l, b_l)
            self.logger.update('loss_paired', loss1.data.item(), b_l)
            self.logger.update('loss_rcm', loss2.data.item(), b_l)
            self.logger.update('loss_reg', loss3.data.item(), b_l) 
            self.logger.update('loss', loss.data.item(), b_l)
