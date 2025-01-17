import torch
import torch.nn as nn
import math
import matplotlib.pyplot as plt
from torch.autograd import Function
from models.utils import *
from models.loss_function import MMDLoss


class NDDR(nn.Module):
    def __init__(self, args):
        super(NDDR, self).__init__()
        self.input_size = args.time_length
        self.channel_size = args.channel_size
        self.d_model = args.d_model
        self.d_model2 = args.d_model2

        self.num_layers_FreqEncoder = args.num_layers_FreqEncoder
        self.num_heads_FreqEncoder = args.num_heads_FreqEncoder
        self.num_layers_encoder = args.num_layers_encoder
        self.num_heads_encoder = args.num_heads_encoder

        self.num_layers_for_subject = args.num_layers_for_subject
        self.num_layers_no_subject = args.num_layers_no_subject
        self.num_heads_for_subject = args.num_heads_for_subject
        self.num_heads_no_subject = args.num_heads_no_subject

        self.dropout = args.dropout
        self.num_layers_decoder = args.num_layers_decoder
        self.num_heads_decoder = args.num_heads_decoder
        self.pred_dim = args.pred_dim
        self.self_expression_dim = args.self_expression_dim
        self.class_num = args.class_num
        self.domain_num = args.subject_num
        self.device = args.device

        self.bn = nn.BatchNorm1d(self.channel_size)

        self.freqEncoder = SelfAttentionNet(self.channel_size, self.channel_size, self.num_heads_FreqEncoder,
                                            self.num_layers_FreqEncoder, self.dropout)

        self.encoder = SelfAttentionNet(self.d_model, self.d_model, self.num_heads_encoder,
                                        self.num_layers_encoder, self.dropout, "relu")

        self.encoder = nn.Sequential(nn.Linear(self.input_size, self.d_model),
                                     nn.ReLU(),
                                     self.encoder,
                                     nn.BatchNorm1d(self.channel_size),
                                     nn.ReLU())

        self.InternalEncoder = nn.Sequential(nn.Linear(self.d_model*self.channel_size, 2*self.d_model*self.channel_size),
                                             nn.ReLU(),
                                             nn.Linear(2*self.d_model*self.channel_size, self.d_model2),
                                             )

        self.MutualEncoder = nn.Sequential(nn.Linear(self.d_model*self.channel_size, 2*self.d_model*self.channel_size),
                                           nn.ReLU(),
                                           nn.Linear(2*self.d_model*self.channel_size, self.d_model2),
                                           )

        self.predictor = nn.Sequential(nn.Linear(self.d_model2, self.pred_dim),
                                       nn.ReLU())

        self.domain_jw_encoder = nn.Linear(self.d_model*self.channel_size, args.domain_jw_dim)
        self.mutual_jw_encoder = nn.Linear(self.d_model*self.channel_size, args.mutual_jw_dim)

        self.decoder1 = nn.Sequential(nn.Linear(self.d_model2, self.d_model*self.channel_size),)
        self.decoder2 = nn.Sequential(nn.Linear(self.d_model2, self.d_model*self.channel_size))

        self.decoder = CrossAttentionNet(self.d_model, self.d_model, self.num_heads_decoder,
                                         self.num_layers_decoder, self.dropout, "relu")
        self.decoder_out = nn.Sequential(nn.Linear(self.d_model, self.input_size))

        self.ClassEncoder = nn.Sequential(nn.Linear(2 * self.d_model2, self.d_model2),
                                          nn.ReLU(),
                                          nn.Linear(self.d_model2, 2 * self.d_model2))

        self.domain_discriminator = nn.Sequential(nn.Linear(self.d_model2, self.domain_num))

        self.class_discriminator = nn.Sequential(nn.Linear(2*self.d_model2, self.class_num))

    def reset_parameters(self):
        for module in self.children():
            if hasattr(module, 'reset_parameters'):
                module.reset_parameters()

    def forward(self, x, y=None, type="s"):
        x = self.bn(x)
        x = torch.transpose(x, 1, 2)
        x = self.freqEncoder(x)
        x = torch.transpose(x, 1, 2).contiguous()
        high_embedding = self.encoder(x)

        high_embedding = high_embedding.reshape(high_embedding.size(0), -1)

        domain_embedding = self.InternalEncoder(high_embedding)
        mutual_embedding = self.MutualEncoder(high_embedding)
        rec_embedding1 = self.decoder1(domain_embedding)
        rec_embedding1 = rec_embedding1.reshape(rec_embedding1.size(0), self.channel_size, self.d_model)
        rec_embedding2 = self.decoder2(mutual_embedding)
        rec_embedding2 = rec_embedding2.reshape(rec_embedding2.size(0), self.channel_size, self.d_model)
        rec_embedding = self.decoder(rec_embedding1, rec_embedding2)
        rec_embedding = self.decoder_out(rec_embedding)

        domain_predictor = self.predictor(domain_embedding)
        mutual_predictor = self.predictor(mutual_embedding)

        class_embedding = torch.cat((domain_embedding, mutual_embedding), dim=1)
        # class_embedding = self.ClassEncoder(class_embedding)

        domain_cls = self.domain_discriminator(domain_embedding)
        class_cls = self.class_discriminator(class_embedding)
        # class_cls = self.class_discriminator(class_embedding.reshape(class_embedding.shape[0], -1))

        loss_pre = self.loss_predictor(domain_predictor, mutual_predictor)
        # dw_domain = self.domain_jw_encoder(domain_embedding.reshape([domain_embedding.shape[0], -1]))
        # dw_mutual = self.domain_jw_encoder(mutual_embedding.reshape([mutual_embedding.shape[0], -1]))
        # loss_IIF = self.loss_class_align(dw_domain, y)
        # loss_IIF = self.loss_class_align(domain_embedding.reshape([domain_embedding.shape[0], -1]), y)
        loss_IIF = self.loss_class_align(domain_embedding, y)
        loss_MIF = self.loss_class_align(mutual_embedding, y)
        loss_class = self.loss_class_align(class_embedding, y)
        loss_rec = self.loss_reconstruction(x, rec_embedding)

        return ([class_embedding, mutual_embedding, domain_embedding, rec_embedding], [class_cls, domain_cls],
                [loss_pre, loss_IIF, loss_MIF, loss_rec, loss_class])

    def evaluate(self, x):
        x = self.bn(x)
        x = torch.transpose(x, 1, 2)
        x = self.freqEncoder(x)
        x = torch.transpose(x, 1, 2).contiguous()
        high_embedding = self.encoder(x)

        high_embedding = high_embedding.reshape(high_embedding.size(0), -1)

        domain_embedding = self.InternalEncoder(high_embedding)
        mutual_embedding = self.MutualEncoder(high_embedding)

        class_embedding = torch.cat((domain_embedding, mutual_embedding), dim=1)
        # class_embedding = self.ClassEncoder(class_embedding)
        domain_cls = self.domain_discriminator(domain_embedding)
        class_cls = self.class_discriminator(class_embedding)
        return class_embedding, class_cls, domain_embedding, domain_cls

    def recon(self, x):
        x = self.bn(x)
        x = torch.transpose(x, 1, 2)
        x = self.freqEncoder(x)
        x = torch.transpose(x, 1, 2).contiguous()
        high_embedding = self.encoder(x)

        high_embedding = high_embedding.reshape(high_embedding.size(0), -1)

        domain_embedding = self.InternalEncoder(high_embedding)
        mutual_embedding = self.MutualEncoder(high_embedding)
        rec_embedding1 = self.decoder1(domain_embedding)
        rec_embedding1 = rec_embedding1.reshape(rec_embedding1.size(0), self.channel_size, self.d_model)
        rec_embedding2 = self.decoder2(mutual_embedding)
        rec_embedding2 = rec_embedding2.reshape(rec_embedding2.size(0), self.channel_size, self.d_model)
        rec_embedding = self.decoder(rec_embedding1, rec_embedding2)
        rec_embedding = self.decoder_out(rec_embedding)

        loss_rec = self.loss_reconstruction(x, rec_embedding)
        return rec_embedding, loss_rec

    def loss_predictor(self, class_predictor, domain_predictor):
        loss = nn.CosineSimilarity(dim=1)
        return torch.abs(loss(class_predictor, domain_predictor).mean())

    def loss_reconstruction(self, original, predicted):
        loss = nn.MSELoss()
        return loss(original, predicted)

    def loss_class_align(self, feature, label, margin=1.5):
        loss = 0
        label = label.squeeze()
        unique_labels = label.unique()
        grouped_features = {lbl.item(): feature[label == lbl] for lbl in unique_labels}

        for i in grouped_features:
            f = grouped_features[i]
            center = f.mean(dim=0)

            # 计算每个点到中心的距离
            distances_to_center = torch.abs(f - center).sum(dim=1)  # L1 距离
            mean_distance = distances_to_center.mean()
            loss += torch.clamp(mean_distance - margin, min=0)
        return loss

