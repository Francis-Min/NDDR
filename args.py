import argparse
import os
import pickle
import json

parser = argparse.ArgumentParser()
# # dataset and dataloader args
# parser.add_argument('--save_path', type=str, default='exp/epilepsy/test')
# parser.add_argument('--dataset', type=str, default='eeg')
# parser.add_argument('--data_path', type=str,
#                     default='data/EEG/')
# #parser.add_argument('--device', type=str, default='cuda')
# parser.add_argument('--device', type=str, default='cuda')
# # parser.add_argument('--train_batch_size', type=int, default=128)
# # parser.add_argument('--test_batch_size', type=int, default=128)
# parser.add_argument('--train_batch_size', type=int, default=32)
# parser.add_argument('--test_batch_size', type=int, default=32)
#
# # SiamMAE
# parser.add_argument('--class_num', type=int, default=40)
# parser.add_argument('--channel', type=int, default=128)
# parser.add_argument('--input_size', type=int, default=440)
# parser.add_argument('--d_model', type=int, default=1024)
# parser.add_argument('--pred_dim', type=int, default=512)
# parser.add_argument('--dropout', type=int, default=0.2)
# parser.add_argument('--batch_size', type=int, default=32)
# parser.add_argument('--num_heads_encoder', type=int, default=16)
# parser.add_argument('--num_heads_decoder', type=int, default=8)
# parser.add_argument('--eval_per_steps', type=int, default=5)
# parser.add_argument('--num_layers_encoder', type=int, default=8)
# parser.add_argument('--num_layers_decoder', type=int, default=8)
#
# parser.add_argument('--enable_res_parameter', type=int, default=1)
# parser.add_argument('--alpha', type=float, default=4.0)
# parser.add_argument('--beta', type=float, default=2.0)
#
# # knowledge distillation
# parser.add_argument('--distill_dim', type=int, default=1024)
#
# parser.add_argument('--momentum', type=float, default=0.99)
# parser.add_argument('--vocab_size', type=int,  default=660)
# parser.add_argument('--wave_length', type=int, default=4)
# parser.add_argument('--mask_ratio', type=float, default=0.75)
# parser.add_argument('--reg_layers', type=int, default=4)
#
# # train args
# parser.add_argument('--lr', type=float, default=1e-5)
# parser.add_argument('--lr_decay_rate', type=float, default=1.)
# parser.add_argument('--lr_decay_steps', type=int, default=100)
# parser.add_argument('--weight_decay', type=float, default=0.01)
# parser.add_argument('--num_epoch_pretrain', type=int, default=500)
# parser.add_argument('--num_epoch', type=int, default=400)
# parser.add_argument('--load_pretrained_model', type=int, default=1)


# Trainer
parser.add_argument('--device', type=str, default='cuda:0')
parser.add_argument('--save', type=int, default=1)

# DataSet
parser.add_argument('--class_num', type=int, default=3)
parser.add_argument('--subject_num', type=int, default=15)

# Model
parser.add_argument('--train_batch_size', type=int, default=256)
parser.add_argument('--test_batch_size', type=int, default=256)
parser.add_argument('--time_length', type=int, default=5)
parser.add_argument('--channel_size', type=int, default=62)
parser.add_argument('--d_model', type=int, default=64)
parser.add_argument('--d_model2', type=int, default=16)

parser.add_argument('--num_layers_FreqEncoder', type=int, default=1)
parser.add_argument('--num_heads_FreqEncoder', type=int, default=2)

parser.add_argument('--num_layers_encoder', type=int, default=2)
parser.add_argument('--num_heads_encoder', type=int, default=4)
parser.add_argument('--num_layers_decoder', type=int, default=2)
parser.add_argument('--num_heads_decoder', type=int, default=4)

parser.add_argument('--num_layers_for_subject', type=int, default=1)
parser.add_argument('--num_layers_no_subject', type=int, default=1)
parser.add_argument('--num_heads_for_subject', type=int, default=4)
parser.add_argument('--num_heads_no_subject', type=int, default=4)
parser.add_argument('--dropout', type=float, default=0.2)
# parser.add_argument('--num_layers_decoder', type=int, default=2)
# parser.add_argument('--num_heads_decoder', type=int, default=4)
parser.add_argument('--pred_dim', type=int, default=64)
parser.add_argument('--domain_jw_dim', type=int, default=128)
parser.add_argument('--mutual_jw_dim', type=int, default=128)
parser.add_argument('--self_expression_dim', type=int, default=16)


args = parser.parse_args()
