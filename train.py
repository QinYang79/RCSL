# coding=utf-8
import logging
import os
import time
import numpy as np
import torch 

import shutil
import opts
import tensorboard_logger as tb_logger

import data
from utils import save_config, cosine_similarity_matrix
from evaluation import AverageMeter, LogCollector, encode_data, evalrank, i2t, t2i
from model import SVSE
from vocab import deserialize_vocab, deserialize_vocab_glove
import warnings

warnings.filterwarnings("ignore")


def adjust_learning_rate(model, epoch, lr_schedules):
    logger = logging.getLogger(__name__)
    """Sets the learning rate to the initial LR
       decayed by 10 every opt.lr_update epochs"""
    
    if epoch in lr_schedules:
        logger.info('Current epoch num is {}, decrease all lr by 10'.format(epoch, ))
        for param_group in model.optimizer.param_groups:
            old_lr = param_group['lr']
            new_lr = old_lr * 0.1
            param_group['lr'] = new_lr
            logger.info('new lr {}'.format(new_lr))


def init_logging(log_file_path):
    logger = logging.getLogger(__name__)
    logger.setLevel(level=logging.DEBUG)

    formatter = logging.Formatter('%(asctime)s %(message)s')

    file_handler = logging.FileHandler(log_file_path)
    file_handler.setLevel(level=logging.INFO)
    file_handler.setFormatter(formatter)

    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.DEBUG)
    stream_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    return logger


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar', prefix='', ckpt=True,stage=''):
    logger = logging.getLogger(__name__)
    tries = 15
    # deal with unstable I/O. Usually not necessary.
    while tries:
        try:
            if ckpt:
                torch.save(state, prefix + filename)
            if is_best:
                torch.save(state, prefix + f'model_{stage}_best.pth.tar')
        except IOError as e:
            error = e
            tries -= 1
        else:
            break
        logger.info('model save {} failed, remaining {} trials'.format(filename, tries))
        if not tries:
            raise error

 
def UpdateMemoryBank_(data_loader, model, topK):
    memory_bank_path = model.opt.logger_path+f'/memory_bank_top{topK}.npy'

    if 'f30k' in data_loader.dataset.opt.data_name:
        bs = 1000
    else:
        bs = 400
        
    model.val_start()

    memory_bank = {
        'hard_i2t': torch.zeros((data_loader.dataset.img_length , 2*topK)).cuda(), # index sims_i2t
        'hard_t2i': torch.zeros((data_loader.dataset.length , 2*topK)).cuda() # index sims_t2i
    }

    print("compute embs")
    img_set =  data.Img_dataset(data_loader.dataset.images)
    cap_set =  data.Cap_dataset(data_loader.dataset.captions, data_loader.dataset.vocab)
    img_set_loader = torch.utils.data.DataLoader(dataset=img_set, batch_size=bs,
                                            shuffle=False,
                                            collate_fn=data.collate_fn_img,
                                            num_workers=10,
                                            drop_last=False)
    cap_set_loader = torch.utils.data.DataLoader(dataset=cap_set, batch_size=bs,
                                            shuffle=False,
                                            collate_fn=data.collate_fn_cap,
                                            num_workers=10,
                                            drop_last=False)

    img_embs = np.zeros((data_loader.dataset.img_length,1024))
    cap_embs = np.zeros((data_loader.dataset.length,1024))


    for i, data_i in enumerate(img_set_loader):
        images, image_lengths, img_ids = data_i
        with torch.no_grad():
            img_emb = model.forward_imgs(images, image_lengths)
        img_embs[img_ids] = img_emb.data.cpu()

    for i, data_i in enumerate(cap_set_loader):
        captions, caption_lengths, cap_ids = data_i
        with torch.no_grad():
            cap_emb = model.forward_caps(captions, caption_lengths)
        cap_embs[cap_ids] = cap_emb.data.cpu()

    shuffle_inx = data_loader.dataset.shuffle_inx
    i_label = torch.ones(img_embs.shape[0])
    t_label = torch.ones(cap_embs.shape[0])
    for i in range(img_embs.shape[0]):
        if shuffle_inx[i] == i:
            i_label[i] = 0

    for i in range(cap_embs.shape[0]):
        if shuffle_inx[i//5] == i//5:
            t_label[i] = 0

    i_label = i_label.cuda()
    t_label = t_label.cuda()


    print("i2t correlation")
    n_i = (img_embs.shape[0]-1) // bs +1
    n_t = (cap_embs.shape[0]-1) // bs +1
  
    for i in range(n_i):
        if i%5==0:
            print( f"image batch:  {i}")
        end =  i_label.size(0) if i == n_i-1 else (i+1)*bs
        sims = (torch.Tensor(img_embs[i * bs: end]).cuda()).mm( torch.Tensor(cap_embs).cuda().t()) * t_label * (i_label[ i * bs : end].view(-1,1))
        max = sims.topk(dim=1,k=topK)
        # print(max[0].size(),max[1].size())
        for j in range(i * bs, end):
            if i_label[j].data.item() == 1:
                memory_bank['hard_i2t'][j] = torch.cat([max[1][j-i * bs], max[0][j-i * bs]], dim=-1)
            else:
                memory_bank['hard_i2t'][j] = torch.Tensor(np.array([j*5+k for k in range(topK)]+[1 for _ in range(topK)])).cuda() 
        del sims
    print("t2i correlation")
    for i in range(n_t):
        if i%30==0:
            print( f"text batch:  {i}")
        end =  t_label.size(0) if i == n_t-1 else (i+1)*bs
        sims = (torch.Tensor(cap_embs[i * bs: end]).cuda()).mm( torch.Tensor(img_embs).cuda().t()) * i_label  * (t_label[ i * bs : end].view(-1,1))
        max = sims.topk(dim=1,k=topK)
        for j in range(i * bs, end):
            if t_label[j].data.item() == 1:
                memory_bank['hard_t2i'][j] = torch.cat([max[1][j-i * bs], max[0][j-i * bs]], dim=-1)
            else:
                memory_bank['hard_t2i'][j] = torch.Tensor(np.array([j//5 for k in range(topK)]+[1 for _ in range(topK)])).cuda() # else paired
        del sims
    memory_bank['hard_i2t'] = memory_bank['hard_i2t'].detach().cpu().numpy()
    memory_bank['hard_t2i'] = memory_bank['hard_t2i'].detach().cpu().numpy()
    
    del i_label,t_label,img_set_loader,cap_set_loader,img_set,cap_set
    torch.cuda.empty_cache()
    np.save(memory_bank_path, memory_bank)
    return memory_bank


def train(opt, train_loader, model, epoch, val_loader, best_rsum=0):
    # average meters to record the training statistics
    logger = logging.getLogger(__name__)
    batch_time = AverageMeter()
    data_time = AverageMeter()
    train_logger = LogCollector()

    num_loader_iter = len(train_loader.dataset) // train_loader.batch_size + 1
    end = time.time()
    logger.info("=======>Epoch: {0}".format(epoch))
    for i, train_data in enumerate(train_loader):
        model.train_start()
        data_time.update(time.time() - end)
        model.logger = train_logger

        # Update the model
        images, img_lengths, captions, cap_lengths, img_ids, ids, labels = train_data

        model.train_emb(images, captions, img_lengths, cap_lengths, img_ids, ids, labels, epoch=epoch)

        batch_time.update(time.time() - end)
        end = time.time()
        if model.step % opt.log_step == 0:
            logger.info( 
                'Epoch: [{0}][{1}/{2}] Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                'Data {data_time.val:.3f} ({data_time.avg:.3f})\t{e_log}'.format(epoch, i, num_loader_iter,
                                                                                  batch_time=batch_time,
                                                                                  data_time=data_time,
                                                                                  e_log=str(model.logger)))

        # Record logs in tensorboard
        tb_logger.log_value('epoch', epoch, step=model.step)
        tb_logger.log_value('step', i, step=model.step)
        tb_logger.log_value('batch_time', batch_time.val, step=model.step)
        tb_logger.log_value('data_time', data_time.val, step=model.step)
        model.logger.tb_log(tb_logger, step=model.step)


def validate(val_loader, model, mode='dev'):
    model.val_start()
    logger.info(f"=====>Mode: {mode}")
    npts = 0
    with torch.no_grad():
        img_embs, cap_embs = encode_data(model, val_loader)
        img_embs = np.array([img_embs[i] for i in range(0, len(img_embs), 5)])

    sims = cosine_similarity_matrix(img_embs, cap_embs)
    npts = img_embs.shape[0]
    (r1, r5, r10, medr, meanr) = i2t(npts, sims)
    logger.info("Image to text: %.1f, %.1f, %.1f, %.1f, %.1f" %
                (r1, r5, r10, medr, meanr))
    # image retrieval
    (r1i, r5i, r10i, medri, meanr) = t2i(npts, sims)
    logger.info("Text to image: %.1f, %.1f, %.1f, %.1f, %.1f" %
                (r1i, r5i, r10i, medri, meanr))
    currscore = r1 + r5 + r10 + r1i + r5i + r10i
    logger.info('rSum is {0}'.format(currscore))
 
    # record metrics in tensorboard
    if mode == 'test':
        tb_logger.log_value('t-r1', r1, step=model.step)
        tb_logger.log_value('t-r5', r5, step=model.step)
        tb_logger.log_value('t-r10', r10, step=model.step)
        tb_logger.log_value('t-medr', medr, step=model.step)
        tb_logger.log_value('t-meanr', meanr, step=model.step)
        tb_logger.log_value('t-r1i', r1i, step=model.step)
        tb_logger.log_value('t-r5i', r5i, step=model.step)
        tb_logger.log_value('t-r10i', r10i, step=model.step)
        tb_logger.log_value('t-medri', medri, step=model.step)
        tb_logger.log_value('t-meanr', meanr, step=model.step)
        tb_logger.log_value('t-rsum', currscore, step=model.step)
    else:
        tb_logger.log_value('r1', r1, step=model.step)
        tb_logger.log_value('r5', r5, step=model.step)
        tb_logger.log_value('r10', r10, step=model.step)
        tb_logger.log_value('medr', medr, step=model.step)
        tb_logger.log_value('meanr', meanr, step=model.step)
        tb_logger.log_value('r1i', r1i, step=model.step)
        tb_logger.log_value('r5i', r5i, step=model.step)
        tb_logger.log_value('r10i', r10i, step=model.step)
        tb_logger.log_value('medri', medri, step=model.step)
        tb_logger.log_value('meanr', meanr, step=model.step)
        tb_logger.log_value('rsum', currscore, step=model.step)
        
    return currscore
def com(memory_bank,th=0.5,shuffle_inx=None):
    logger = logging.getLogger(__name__)
    len_ = 0
    count_ = 0
    for i in range(memory_bank['hard_i2t'].shape[0]):
        if memory_bank['hard_i2t'][i][1] > th and shuffle_inx[i] != i:
            len_ += 1
            if memory_bank['hard_i2t'][i][0]//5 == i:
                count_ += 1
    logger.info(f"i2t hard matched:  {count_}, {len_}, {count_/(len_+1)}")
   
    len_ = 0
    count_ = 0
    for i in range(memory_bank['hard_t2i'].shape[0]):
        if memory_bank['hard_t2i'][i][1] > th and shuffle_inx[i//5] != i//5:
            len_ += 1
            if memory_bank['hard_t2i'][i][0] == i//5:
                count_ += 1
    logger.info(f"t2i hard matched:  {count_}, {len_}, {count_/(len_+1)}")


def UpdateMemoryBank(data_loader, model, time_u=0):
    memory_bank_path = model.opt.logger_path+f'/memory_bank_{time_u}.npy'
    # memory_bank_path ='/home/qinyang/windows/sda1/ProjectsOfQy/EvidenceTextImage/SemiVSE/mb/memory_bank.npy'
    if os.path.exists(memory_bank_path):
        memory_bank = np.load(memory_bank_path, allow_pickle= True).item()
        return memory_bank

    if 'f30k' in data_loader.dataset.opt.data_name:
        bs = 1000
    else:
        bs = 400
        
    model.val_start()
    if time_u == 0:
        memory_bank = {
            'hard_i2t': torch.zeros((data_loader.dataset.img_length , 2)).cuda(), # index sims_i2t
            'hard_t2i': torch.zeros((data_loader.dataset.old_length , 2)).cuda() # index sims_t2i
        }
    else:
        memory_bank = {
            'hard_i2t': torch.Tensor(model.memory_bank['hard_i2t']).cuda(), # index sims_i2t
            'hard_t2i': torch.Tensor(model.memory_bank['hard_t2i']).cuda() # index sims_t2i
        }
    print("compute embs")
    img_set =  data.Img_dataset(data_loader.dataset.images)
    cap_set =  data.Cap_dataset(data_loader.dataset.captions, data_loader.dataset.vocab)
    img_set_loader = torch.utils.data.DataLoader(dataset=img_set, batch_size=bs,
                                            shuffle=False,
                                            collate_fn=data.collate_fn_img,
                                            num_workers=10,
                                            drop_last=False)
    cap_set_loader = torch.utils.data.DataLoader(dataset=cap_set, batch_size=bs,
                                            shuffle=False,
                                            collate_fn=data.collate_fn_cap,
                                            num_workers=10,
                                            drop_last=False)

    img_embs = np.zeros((data_loader.dataset.img_length,1024))
    cap_embs = np.zeros((data_loader.dataset.old_length,1024))

    for i, data_i in enumerate(img_set_loader):
        images, image_lengths, img_ids = data_i
        with torch.no_grad():
            img_emb = model.forward_imgs(images, image_lengths)
        img_embs[img_ids] = img_emb.data.cpu()

    for i, data_i in enumerate(cap_set_loader):
        captions, caption_lengths, cap_ids = data_i
        with torch.no_grad():
            cap_emb = model.forward_caps(captions, caption_lengths)
        cap_embs[cap_ids] = cap_emb.data.cpu()

    shuffle_inx = data_loader.dataset.shuffle_inx
    i_label = torch.ones(img_embs.shape[0])
    t_label = torch.ones(cap_embs.shape[0])
    for i in range(img_embs.shape[0]):
        if shuffle_inx[i] == i:
            i_label[i] = 0

    for i in range(cap_embs.shape[0]):
        if shuffle_inx[i//5] == i//5:
            t_label[i] = 0

    i_label = i_label.cuda()
    t_label = t_label.cuda()


    print("i2t correlation")
    n_i = (img_embs.shape[0]-1) // bs +1
    n_t = (cap_embs.shape[0]-1) // bs +1

    for i in range(n_i):
        if i%5==0:
            print( f"image batch:  {i}")
        end =  i_label.size(0) if i == n_i-1 else (i+1)*bs
        sims = (torch.Tensor(img_embs[i * bs: end]).cuda()).mm( torch.Tensor(cap_embs).cuda().t()) 
        # sims = (torch.Tensor(img_embs[i * bs: end]).cuda()).mm( torch.Tensor(cap_embs).cuda().t()) 
     
        max = sims.max(dim=1)
        for j in range(i * bs, end):
            if i_label[j].data.item() == 1:
                if max[0][j-i * bs].data.item() > memory_bank['hard_i2t'][j][1] or time_u == 0:
                    memory_bank['hard_i2t'][j] = torch.Tensor(np.array([max[1][j-i * bs].data.item(), max[0][j-i * bs].data.item()])).cuda()
            else:
                memory_bank['hard_i2t'][j] = torch.Tensor(np.array([j*5, 1])).cuda() # else paired
        del sims
    print("t2i correlation")
    for i in range(n_t):
        if i%30==0:
            print( f"text batch:  {i}")

        end =  t_label.size(0) if i == n_t-1 else (i+1)*bs
        sims = (torch.Tensor(cap_embs[i * bs: end]).cuda()).mm( torch.Tensor(img_embs).cuda().t()) 
        # sims = (torch.Tensor(cap_embs[i * bs: end]).cuda()).mm( torch.Tensor(img_embs).cuda().t())
        max = sims.max(dim=1)
        for j in range(i * bs, end):
            if t_label[j].data.item() == 1:
                if max[0][j - i*bs].data.item() > memory_bank['hard_t2i'][j][1] or time_u == 0:
                    memory_bank['hard_t2i'][j] = torch.Tensor(np.array([max[1][j-i * bs].data.item(), max[0][j-i * bs].data.item()])).cuda()
            else:
                memory_bank['hard_t2i'][j] = torch.Tensor(np.array([j//5, 1])).cuda() # else paired
        del sims
    memory_bank['hard_i2t'] = memory_bank['hard_i2t'].detach().cpu().numpy()
    memory_bank['hard_t2i'] = memory_bank['hard_t2i'].detach().cpu().numpy()
    
    del i_label,t_label,img_set_loader,cap_set_loader,img_set,cap_set
    torch.cuda.empty_cache()
    np.save(memory_bank_path, memory_bank)
    return memory_bank


if __name__ == '__main__':
    parser = opts.get_argument_parser()
    opt = parser.parse_args()

    # Make dir
    if not os.path.isdir(opt.model_path):
        os.makedirs(opt.model_path)
    if not os.path.isdir(opt.logger_path):
        os.makedirs(opt.logger_path)
    # Save config
    save_config(opt, os.path.join(opt.logger_path, "config.json"))
    # logger initialization
    tb_logger.configure(opt.logger_path, flush_secs=5)
    logger = init_logging(opt.logger_path + '/log.txt')
    logger.info(f"===>PID:{os.getpid()}, GPU:[{opt.gpu}]")
    logger.info(opt)
    # Load Vocabulary

    v_path = os.path.join(opt.vocab_path, '%s_vocab.json' % opt.data_name)
    if opt.init_txt == 'glove': 
        vocab_or_tokenizer = deserialize_vocab_glove(v_path)
        word2idx = vocab_or_tokenizer.word2idx
    else:
        vocab_or_tokenizer = deserialize_vocab(v_path)
        word2idx = None
    opt.vocab_size = len(vocab_or_tokenizer)
    model = SVSE(opt,word2idx)
    if not model.parallel:
        model.make_data_parallel()

    # Get data loaders
    train_loader, val_loader, test_loader = data.get_loaders(opt.data_name, vocab_or_tokenizer, opt.batch_size,
                                                             opt.workers, opt)
        
    # Load checkpoint
    start_epoch = 0
    best_rsum = 0
    lr_schedules = [opt.lr_update, 2*opt.lr_update, 3*opt.lr_update]
    
    if opt.resume:
        if os.path.isfile(opt.resume):
            logger.info("=> loading checkpoint '{}'".format(opt.resume))
            checkpoint = torch.load(opt.resume)
            start_epoch = checkpoint['epoch'] + 1
            best_rsum = checkpoint['best_rsum']
            model.load_state_dict(checkpoint['model'])
            # step is used to show logs as the continuation of another training
            model.step = checkpoint['step']
            opt.stage = checkpoint['opt'].stage
            # opt.learning_rate *= 0.1 
            model.opt = opt
            if opt.stage == 'mining':
                model.memory_bank = checkpoint['memory_bank']
                memory_bank_path = model.opt.logger_path+f'/memory_bank_0.npy' 
                np.save(memory_bank_path,  model.memory_bank)
                
            model.reinit_optimizer()

            logger.info("=> loaded checkpoint '{}' (epoch {}, best_rsum {})"
                        .format(opt.resume, start_epoch-1, best_rsum))
            validate(val_loader, model, 'dev')
        else:
            logger.info("=> no checkpoint found at '{}'".format(opt.resume))
    #####
    # Train the Model
    logger.info("Logger path\t" + opt.logger_path)
    logger.info("Save path\t" + opt.model_path)
    if not os.path.exists(opt.model_path):
        os.makedirs(opt.model_path)

    # UpdateMemoryBank_(train_loader, model, topK=5)
    # exit()
    for epoch in range(start_epoch, opt.MaxEpoch):
        if epoch < opt.MineEpoch:
            logger.info(f"Learning, best_rsum:{best_rsum}") 
            model.opt.stage = 'learning' 
        else:
            model.opt.stage = 'mining' 
            logger.info(f"Mining, best_rsum:{best_rsum}") 

        train_loader.dataset.re_sort()
        train_loader.dataset.opt = model.opt 

        if epoch == opt.MineEpoch: 
            model.reinit_optimizer()  #keep
            
        if epoch >= opt.MineEpoch: 
            logger.info(f"Start mining") 
            memory_bank = UpdateMemoryBank(train_loader, model)
            model.memory_bank = memory_bank
            train_loader.dataset.memory_bank = model.memory_bank


        adjust_learning_rate(model, epoch, lr_schedules)
        train(opt, train_loader, model, epoch, val_loader, best_rsum)
        # # evaluate on validation set
        rsum = validate(val_loader, model, 'dev')
        validate(test_loader, model, 'test')

        # remember best R@ sum and save checkpoint
        is_best = rsum > best_rsum
        best_rsum = max(rsum, best_rsum)
        if epoch == (opt.MineEpoch-1):
            ckpt = True 
        else:
            ckpt = False
 
        save_checkpoint({
            'epoch': epoch,
            'model': model.state_dict(),
            'step': model.step,
            'best_rsum': best_rsum,
            'memory_bank': model.memory_bank,
            'opt': opt,
        }, is_best, filename='checkpoint_{}.pth.tar'.format(epoch), prefix=opt.model_path + '/',ckpt=ckpt,stage= model.opt.stage)
 
    logger.info(f"best_rsum:{best_rsum}")

    # Get data loader
    
    checkpoint = torch.load(opt.model_path+'/model_mining_best.pth.tar')
    opt = checkpoint['opt']
    model = SVSE(opt,word2idx)
    if not model.parallel:
        model.make_data_parallel()
     
    model.load_state_dict(checkpoint['model'])

    if 'coco' in opt.data_name:
        test_loader = data.get_test_loader('testall', opt.data_name, vocab_or_tokenizer, opt.batch_size, opt.workers,
                                           opt)
        evalrank(test_loader, model, fold5=True, logger = logger)
        evalrank(test_loader, model, fold5=False, logger =logger)
    else:
        test_loader = data.get_test_loader('test', opt.data_name, vocab_or_tokenizer, 128, opt.workers,
                                           opt)
        evalrank(test_loader, model, fold5=False, logger =logger)

