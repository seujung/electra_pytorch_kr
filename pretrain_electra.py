import os
import argparse
import numpy as np
import json
import torch
from torch import nn
from torch.autograd import Variable
import random
import copy
import logging
from glob import glob

from data import PretrainDataset
from tokenizer import SentencepieceTokenizer
from model.utils import gen_attention_mask, get_lr
from model.modeling_electra import BertModel
from model.modeling_electra import ElectraForPretrain
from metrics import calc_mask_accuracy, calc_gan_accuracy
from optimizer import AdamW
from utils import mkdir_p

import apex
from apex import amp
from apex.parallel import DistributedDataParallel

from tensorboardX import SummaryWriter

parser = argparse.ArgumentParser()
# FOR DISTRIBUTED:  Parse for the local_rank argument, which will be supplied
# automatically by torch.distributed.launch.
parser.add_argument("--local_rank", default=0, type=int)
parser.add_argument("--batch_size", default=16, type=int)
parser.add_argument("--max_len", default=128, type=int)
parser.add_argument("--input_path", default='/home/dmig/work/test_container/dataset/news/news_split', type=str)
parser.add_argument("--tokenizer", default='/home/dmig/work/test_container/tokenizer/tokenizer.model', type=str)
parser.add_argument("--save_path", default='/home/dmig/work/test_container/binary/electra', type=str)
parser.add_argument("--warmup_step", default=10000, type=int)
parser.add_argument("--lr", default=5e-4, type=float)
parser.add_argument("--training_step", default=1000000, type=int)
parser.add_argument("--save_interval", default=1000, type=int)
parser.add_argument("--config", default='./model_config/bert_small_config.json', type=str)
parser.add_argument("--dis_weight", default=40, type=float)
parser.add_argument("--mask_prob", default=0.2, type=float)
parser.add_argument("--cls_prob", default=0.0, type=float)
parser.add_argument("--generation_type", default='bert', type=str)
parser.add_argument("--rezero", action="store_true")
parser.add_argument("--bert", action="store_true")

args = parser.parse_args()

print("REZERO :{}".format(args.rezero))
print("BERT :{}".format(args.bert))
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
stream_hander = logging.StreamHandler()
stream_hander.setFormatter(formatter)
logger.addHandler(stream_hander)

torch.cuda.set_device(args.local_rank)

# FOR DISTRIBUTED:  Initialize the backend.  torch.distributed.launch will provide
# environment variables, and requires that you use init_method=`env://`.
torch.distributed.init_process_group(backend='nccl', init_method='env://')

torch.backends.cudnn.benchmark = True

## Define Tokenizer
tokenizer = SentencepieceTokenizer(model_path = args.tokenizer)

## Define Pretrained model
if args.bert:
    from model.modeling_electra import BertModel
    from model.modeling_electra import ElectraForPretrain
    logger.info("load BERT Electra")
elif args.rezero:
    from model.modeling_electra import BertModel
    from model.modeling_electra import ElectraForPretrain
    logger.info("load reZERO Electra")
else:
    assert("please selete the model type")
model = ElectraForPretrain(args.config).cuda()

##Define loss function
gen_loss = nn.CrossEntropyLoss(ignore_index=-1, reduction='mean')
dis_loss = nn.BCEWithLogitsLoss(reduction='mean')


param_optimizer = list(model.named_parameters())

no_decay = ['bias', 'gamma', 'beta']
optimizer_grouped_parameters = [
    {'params': [p for n, p in param_optimizer if n not in no_decay],
        'weight_decay_rate': 0.01},
    {'params': [p for n, p in param_optimizer if n in no_decay], 'weight_decay_rate': 0.0}]

optimizer = AdamW(optimizer_grouped_parameters, lr=args.lr)
# optimizer = apex.optimizers.FusedLAMB(optimizer_grouped_parameters, lr=args.lr)

model, optimizer = amp.initialize(model, optimizer, opt_level="O1")

model = DistributedDataParallel(model)

log_dir = os.path.join(args.save_path, 'logs')
model_dir = os.path.join(args.save_path, 'models')

if not os.path.exists(log_dir):
    mkdir_p(log_dir)

if not os.path.exists(model_dir):
    mkdir_p(model_dir)
    
sw = SummaryWriter(log_dir=log_dir)


def train(data_loader, model, gen_loss, dis_loss, optimizer, params, step_num, sw):
    for i, data in enumerate(data_loader):
        (
            input_ids,
            origin_ids,
            valid_length,
            segment_ids,
            label_cls,
            mask_idx,
            mask_label,
            label_lm
        ) = data
        
        input_ids = input_ids.cuda()
        origin_ids = origin_ids.cuda()
        segment_ids = segment_ids.cuda()
        valid_length = valid_length.cuda()
        label_lm = label_lm.cuda()
#         print("token id shape:{}".format(token_ids.shape))
        
        if step_num < args.warmup_step:
            new_lr = args.lr * step_num / args.warmup_step
        else:
            offset = (step_num - args.warmup_step) * args.lr / (
                args.training_step - args.warmup_step)
            new_lr = args.lr - offset
        
        for param_group in optimizer.param_groups:
            param_group['lr'] = new_lr
    
    
        optimizer.zero_grad()
        gen_logits, is_replaced_label, dls_logits = model(input_ids, origin_ids, segment_ids, valid_length, label_lm)
        ##Calculate Generator Loss
        masked_lm_loss = gen_loss(gen_logits.transpose(1, 2), label_lm.long())

        ##Calculate Discriminator Loss
        attention_mask = gen_attention_mask(input_ids, valid_length)
        active_indices = attention_mask.view(-1) ==1
        active_logits = dls_logits.view(-1)[active_indices]
        active_labels = is_replaced_label.view(-1)[active_indices]
        loss_val_dis = dis_loss(active_logits, active_labels.float())

        is_replaced_loss = dis_loss(active_logits, active_labels.float())

        total_loss = masked_lm_loss + args.dis_weight * is_replaced_loss
        
        masked_lm_accuracy = calc_mask_accuracy(gen_logits.argmax(2), label_lm, mask_idx)
        sigmoid = nn.Sigmoid()
        active_logits_pred = sigmoid(active_logits)
        is_replaced_accuracy = calc_gan_accuracy(active_logits_pred, active_labels)
        
        if args.local_rank == 0 and step_num % 20 == 0:
            logger.info("step:{} gen_loss:{} dis_loss:{} masked_acc:{} is_replaced_acc:{} lr:{}".format(step_num, masked_lm_loss.item(), args.dis_weight * is_replaced_loss.item(), masked_lm_accuracy, is_replaced_accuracy, get_lr(optimizer), 3))
            sw.add_scalar('loss/masked_lm_loss', masked_lm_loss.item(), global_step=step_num)
            sw.add_scalar('loss/is_replaced_loss', args.dis_weight * is_replaced_loss.item(), global_step=step_num)
            sw.add_scalar('loss/total_loss', total_loss.item(), global_step=step_num)
            sw.add_scalar('acc/masked_lm_accuracy', masked_lm_accuracy, global_step=step_num)
            sw.add_scalar('acc/is_replaced_accuracy', is_replaced_accuracy, global_step=step_num)
            sw.add_scalar('learning_rate', get_lr(optimizer), global_step=step_num)
            
            if step_num % args.save_interval == 0:
                logger.info("save the current model")
                torch.save({
                        'step': step_num,
                        'optimizer_state_dict': optimizer.state_dict()
                        }, '{}/optimizer_step_{}_loss{}.pth'.format(model_dir, step_num, round(total_loss.item(), 3)))
                
                torch.save({
                        'step': step_num,
                        'model_state_dict': model.module.state_dict()
                        }, '{}/electra_step_{}_loss{}.pth'.format(model_dir, step_num, round(total_loss.item(), 3)))
                logger.info("finish save")

        with amp.scale_loss(total_loss, optimizer) as scaled_loss:
            scaled_loss.backward()
        optimizer.step()
        step_num += 1 
    return step_num


file_list = glob(os.path.join(args.input_path,'*'))
data_iter = iter(file_list)
logger.info("total {} files are loaded..".format(len(file_list)))

## Defie loop dataloader
e = 0
step_num = 0
data_iter = iter(file_list)

while True:
    try:
        file = next(data_iter)
        logger.info("step {} file list {}".format(step_num, file))

        dataset = PretrainDataset(file, tok=tokenizer, max_len=args.max_len, mask_prob=args.mask_prob,
                                  cls_prob=args.cls_prob, generation_type=args.generation_type)
        data_loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size)
        step_num = train(data_loader, model, gen_loss, dis_loss, optimizer, optimizer_grouped_parameters, step_num, sw)
        # for data in data_loader:
        #     continue
        if step_num == args.training_step:
            break
    except StopIteration:
        logger.info("update file list")
        data_iter = iter(file_list)
