import torch
import torch.nn as nn
from torch.nn import functional as F
from peach.nn_utils.general import exp_mask, zero_mask, masked_pool, act_name2fn
from torch.nn.init import xavier_normal_

from transformers import BertPreTrainedModel
from transformers import BertModel
from transformers import RobertaConfig, RobertaModel, ROBERTA_PRETRAINED_MODEL_ARCHIVE_MAP
import numpy as np


def get_pos_gain(num_labels, tensor_labels, pos_weight, neg_weight=1., float_dtype=torch.float32):
    if num_labels == 1:
        neg_tensor = torch.full_like(tensor_labels, neg_weight, dtype=float_dtype)
        pos_tensor = torch.full_like(tensor_labels, pos_weight, dtype=float_dtype)
        loss_weights = torch.where(tensor_labels.eq(torch.ones_like(tensor_labels)), pos_tensor, neg_tensor)
    elif num_labels == 2:
        loss_weights = torch.tensor([neg_weight, pos_weight], dtype=float_dtype, device=tensor_labels.device)
    else:
        raise KeyError(num_labels)
    return loss_weights


class BertForPairCls(BertPreTrainedModel):
    def __init__(self, config):
        super(BertForPairCls, self).__init__(config)
        self.num_labels = self.config.num_labels

        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.proj = nn.Linear(config.hidden_size * 4, self.num_labels)  # 4个 rep_src, rep_tgt, src - tgt, src * tgt

    def encoder(self, input_ids, attention_mask=None, token_type_ids=None, position_ids=None):
        outputs = self.bert(input_ids,
                            attention_mask=attention_mask,
                            token_type_ids=token_type_ids,
                            position_ids=position_ids)
        pooled_output = outputs[1]
        return pooled_output

    def classifier(self, rep_src, rep_tgt):
        cls_feature = torch.cat(
            [rep_src, rep_tgt, rep_src - rep_tgt, rep_src * rep_tgt], dim=-1
        )
        cls_feature = self.dropout(cls_feature)
        logits = self.proj(cls_feature)

        if logits.shape[-1] == 1:
            logits = logits.squeeze(-1)
        return logits

    def forward(self, src_input_ids, tgt_input_ids,
                src_attention_mask=None, tgt_attention_mask=None,
                src_token_type_ids=None, tgt_token_type_ids=None,
                labels=None,  *args, **kwargs):
        rep_src = self.encoder(src_input_ids, attention_mask=src_attention_mask, token_type_ids=src_token_type_ids)
        rep_tgt = self.encoder(tgt_input_ids, attention_mask=tgt_attention_mask, token_type_ids=tgt_token_type_ids)
        logits = self.classifier(rep_src, rep_tgt)
        outputs = (logits,)

        if labels is not None:
            if hasattr(self.config, "pos_gain"):
                loss_weight = get_pos_gain(
                    self.num_labels, labels, pos_weight=getattr(self.config, "pos_gain"), neg_weight=1.,
                    float_dtype=logits.dtype)
            else:
                loss_weight = None

            if self.num_labels == 1:
                #  We are doing sigmoid instead of regression
                loss_fct = nn.BCEWithLogitsLoss(pos_weight=loss_weight)
                loss = loss_fct(logits.view(-1), labels.view(-1).to(logits.dtype))
            else:
                loss_fct = nn.CrossEntropyLoss(weight=loss_weight)
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            outputs = (loss,) + outputs
        return outputs


class RobertaForPairCls(BertPreTrainedModel):
    config_class = RobertaConfig
    pretrained_model_archive_map = ROBERTA_PRETRAINED_MODEL_ARCHIVE_MAP
    base_model_prefix = "roberta"
    def __init__(self, config):
        super(RobertaForPairCls, self).__init__(config)
        self.num_labels = self.config.num_labels
        self.roberta = RobertaModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.proj = nn.Linear(config.hidden_size * 4, self.num_labels)

    def encoder(self, input_ids, attention_mask=None, token_type_ids=None, position_ids=None):
        outputs = self.roberta(input_ids,
                               attention_mask=attention_mask,
                               token_type_ids=None,
                               position_ids=position_ids)
        pooled_output = outputs[1]
        return pooled_output

    def classifier(self, rep_src, rep_tgt, cat_type="1"):
        if cat_type == "1":
            cls_feature = torch.cat(
                [rep_src, rep_tgt, rep_src - rep_tgt, rep_src * rep_tgt], dim=-1
            )
        elif cat_type == "2":
            cls_feature = torch.cat(
                [rep_src, rep_tgt, rep_tgt - rep_src, rep_src * rep_tgt], dim=-1
            )
        elif cat_type == "3":
            cls_feature = torch.cat(
                [rep_src, rep_tgt, abs(rep_tgt - rep_src), rep_src * rep_tgt], dim=-1
            )
        else:
            raise KeyError(cat_type)
        cls_feature = self.dropout(cls_feature)
        logits = self.proj(cls_feature)
        return logits

    def forward(self, src_input_ids, tgt_input_ids,
                src_attention_mask=None, tgt_attention_mask=None,
                src_token_type_ids=None, tgt_token_type_ids=None,
                labels=None, *args, **kwargs):
        rep_src = self.encoder(src_input_ids, attention_mask=src_attention_mask, token_type_ids=src_token_type_ids)
        rep_tgt = self.encoder(tgt_input_ids, attention_mask=tgt_attention_mask, token_type_ids=tgt_token_type_ids)
        logits = self.classifier(rep_src, rep_tgt, cat_type="1")
        outputs = (logits,)


        if labels is not None:
            if hasattr(self.config, "pos_gain"):
                loss_weight = get_pos_gain(
                    self.num_labels, labels, pos_weight=getattr(self.config, "pos_gain"), neg_weight=1.,
                    float_dtype=logits.dtype)
            else:
                loss_weight = None

            if self.num_labels == 1:
                #  We are doing sigmoid instead of regression
                loss_fct = nn.BCEWithLogitsLoss(pos_weight=loss_weight)
                loss = loss_fct(logits.view(-1), labels.view(-1).to(logits.dtype))
            else:
                loss_fct = nn.CrossEntropyLoss(weight=loss_weight)
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            outputs = (loss,) + outputs
        return outputs


class BertForPairScoring(BertPreTrainedModel):

    @staticmethod
    def cos_distance(rep1, rep2):
        return - torch.cosine_similarity(rep1, rep2, dim=-1) + 1

    @staticmethod
    def euclidean_distance(rep1, rep2):
        distance = rep1 - rep2
        distance = torch.norm(distance, p=2, dim=-1)
        return distance

    def __init__(self, config):
        super(BertForPairScoring, self).__init__(config)
        self.num_labels = self.config.num_labels
        self.bert = BertModel(config)

        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.proj = nn.Linear(config.hidden_size * 4, self.num_labels)

        self.distance_metric = getattr(config, "distance_metric", "cosine")
        self.hinge_loss_margin = getattr(config, "hinge_loss_margin", 1.)
        self.pos_weight = getattr(config, "pos_weight", 1.)
        self.loss_weight = getattr(config, "loss_weight", 1.)

        if self.distance_metric == "cosine":
            self.distance_metric_fn = self.cos_distance
        elif self.distance_metric == "euclidean":
            self.distance_metric_fn = self.euclidean_distance
        elif self.distance_metric == "bilinear":
            self.dropout = nn.Dropout(config.hidden_dropout_prob)
            self.bilinear = nn.Bilinear(config.hidden_size, config.hidden_size, 1, False)
            self.distance_metric_fn = lambda rep1, rep2: self.bilinear(
                self.dropout(rep1), self.dropout(rep2)).squeeze(-1)
        elif self.distance_metric == "mlp":
            self.dropout = nn.Dropout(config.hidden_dropout_prob)
            self.proj = nn.Linear(config.hidden_size * 4, 1)
            self.distance_metric_fn = lambda rep_src, rep_tgt: self.proj(self.dropout(torch.cat(
                [rep_src, rep_tgt, rep_src - rep_tgt, rep_src * rep_tgt], dim=-1))).squeeze(-1)
        elif self.distance_metric == "l2norm":
            act_fn = act_name2fn("gelu")
            self.linear11 = nn.Sequential(
                nn.Dropout(config.hidden_dropout_prob),
                nn.Linear(config.hidden_size, 2 * config.hidden_size))
            self.linear12 = nn.Sequential(
                nn.Dropout(config.hidden_dropout_prob),
                nn.Linear(2 * config.hidden_size, config.hidden_size))
            self.linear21 = nn.Sequential(
                nn.Dropout(config.hidden_dropout_prob),
                nn.Linear(config.hidden_size, 2 * config.hidden_size))
            self.linear22 = nn.Sequential(
                nn.Dropout(config.hidden_dropout_prob),
                nn.Linear(2 * config.hidden_size, config.hidden_size))
            self.distance_metric_fn = lambda rep1, rep2: self.euclidean_distance(
                self.linear12(act_fn(self.linear11(rep1))),
                self.linear22(act_fn(self.linear21(rep1))),
            )
        elif self.distance_metric == "l2norm_ly1":
            self.linear1 = nn.Sequential(
                nn.Dropout(config.hidden_dropout_prob),
                nn.Linear(config.hidden_size, config.hidden_size))
            self.linear2 = nn.Sequential(
                nn.Dropout(config.hidden_dropout_prob),
                nn.Linear(config.hidden_size, config.hidden_size))
            self.distance_metric_fn = lambda rep1, rep2: self.euclidean_distance(
                self.linear1(rep1), self.linear2(rep1))
        else:
            raise KeyError(self.distance_metric)

    def classifier(self, rep_src, rep_tgt):
        cls_feature = torch.cat(
            [rep_src, rep_tgt, rep_src - rep_tgt, rep_src * rep_tgt], dim=-1
        )
        cls_feature = self.dropout(cls_feature)
        logits = self.proj(cls_feature)

        if logits.shape[-1] == 1:
            logits = logits.squeeze(-1)
        return logits

    def encoder(self, input_ids, attention_mask=None, token_type_ids=None, position_ids=None):
        outputs = self.bert(input_ids,
                            attention_mask=attention_mask,
                            token_type_ids=token_type_ids,
                            position_ids=position_ids)
        pooled_output = outputs[1]
        return pooled_output

    def forward(self, src_input_ids, tgt_input_ids,
                src_attention_mask=None, tgt_attention_mask=None,
                src_token_type_ids=None, tgt_token_type_ids=None,
                label_dict=None,
                *args, **kwargs):
        rep_src = self.encoder(src_input_ids, attention_mask=src_attention_mask, token_type_ids=src_token_type_ids)
        rep_tgt = self.encoder(tgt_input_ids, attention_mask=tgt_attention_mask, token_type_ids=tgt_token_type_ids)

        distances = self.distance_metric_fn(rep_src, rep_tgt)
        outputs = [distances, ]

        if label_dict is not None:
            # neg_src_input_ids : bz, neg_times, sl
            assert "neg_src_input_ids" in label_dict and "neg_src_attention_mask" in label_dict
            bs, nt, _ = list(label_dict["neg_src_input_ids"].shape)
            if "neg_src_token_type_ids" in label_dict:
                neg_src_token_type_ids = label_dict["neg_src_token_type_ids"].view(bs * nt, -1)
            else:
                neg_src_token_type_ids = None
            rep_neg_src = self.encoder(  # bs*nt,hn
                label_dict["neg_src_input_ids"].view(bs * nt, -1),
                attention_mask=label_dict["neg_src_attention_mask"].view(bs * nt, -1),
                token_type_ids=neg_src_token_type_ids)

            assert "neg_tgt_input_ids" in label_dict and "neg_tgt_attention_mask" in label_dict
            if "neg_tgt_token_type_ids" in label_dict:
                neg_tgt_token_type_ids = label_dict["neg_tgt_token_type_ids"].view(bs * nt, -1)
            else:
                neg_tgt_token_type_ids = None

            rep_neg_tgt = self.encoder(  # bs*nt,hn
                label_dict["neg_tgt_input_ids"].view(bs * nt, -1),
                attention_mask=label_dict["neg_tgt_attention_mask"].view(bs * nt, -1),
                token_type_ids=neg_tgt_token_type_ids)

            pos_distances = distances.unsqueeze(1).expand(-1, nt).reshape(bs * nt)  # bs*nt
            neg_distances = self.distance_metric_fn(rep_neg_src, rep_neg_tgt)  # bs*nt

            logits = self.classifier(  # [6bs,2]
                torch.cat([rep_src, rep_neg_src], dim=0),
                torch.cat([rep_tgt, rep_neg_tgt], dim=0),
            )

            # loss fn
            # 1. cls loss
            cls_loss_fn = nn.CrossEntropyLoss(
                weight=torch.tensor([1., self.pos_weight, ], dtype=logits.dtype, device=logits.device)
            )
            cls_loss = cls_loss_fn(
                logits,
                torch.cat([
                    logits.new_ones([rep_src.shape[0], ], dtype=torch.long),
                    logits.new_zeros([rep_tgt.shape[0], ], dtype=torch.long),
                    logits.new_zeros([rep_tgt.shape[0], ], dtype=torch.long),
                    logits.new_zeros([rep_tgt.shape[0], ], dtype=torch.long),
                    logits.new_zeros([rep_tgt.shape[0], ], dtype=torch.long),
                    logits.new_zeros([rep_tgt.shape[0], ], dtype=torch.long), ],

                    dim=0,
                )
            )
            # 2. ranking loss
            rk_losses = torch.relu(self.hinge_loss_margin + pos_distances - neg_distances)
            rk_loss = torch.mean(rk_losses)

            loss = cls_loss + self.loss_weight * rk_loss
            outputs.insert(0, loss)
        else:
            logits = self.classifier(  # [2bs,2]
                rep_src,
                rep_tgt,
            )

        return tuple(outputs)


class RobertaForPairScoring(BertPreTrainedModel):
    config_class = RobertaConfig
    pretrained_model_archive_map = ROBERTA_PRETRAINED_MODEL_ARCHIVE_MAP
    base_model_prefix = "roberta"

    @staticmethod
    def cos_distance(rep1, rep2):
        return - torch.cosine_similarity(rep1, rep2, dim=-1) + 1

    @staticmethod
    def euclidean_distance(rep1, rep2):
        distance = rep1 - rep2
        distance = torch.norm(distance, p=2, dim=-1)
        return distance


    def __init__(self, config):
        super(RobertaForPairScoring, self).__init__(config)
        ##################################################################################################
        # Tucker相关初始化
        # 如果获得实体数目
        self.ent_dim = 200
        self.ent_num = config.ent_num
        d1 = self.ent_dim
        self.E = torch.nn.Embedding(self.ent_num, d1)
        self.R = torch.nn.Embedding(self.ent_num, d1)
        self.W = torch.nn.Parameter(torch.tensor(np.random.uniform(-1, 1, (d1, d1, d1)),
                                                 dtype=torch.float, requires_grad=True))
        # 通过字典获得dropout值
        self.input_dropout = torch.nn.Dropout(0.3)
        self.hidden_dropout1 = torch.nn.Dropout(0.4)
        self.hidden_dropout2 = torch.nn.Dropout(0.5)
        self.loss = torch.nn.BCELoss()

        self.bn0 = torch.nn.BatchNorm1d(d1)
        self.bn1 = torch.nn.BatchNorm1d(d1)
        ##################################################################################################
        self.num_labels = self.config.num_labels
        self.roberta = RobertaModel(config)

        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.proj = nn.Linear(config.hidden_size * 4, self.num_labels)

        self.distance_metric = getattr(config, "distance_metric", "cosine")  # "cosine"为默认值
        self.hinge_loss_margin = getattr(config, "hinge_loss_margin", 1.)
        self.pos_weight = getattr(config, "pos_weight", 1.)
        self.loss_weight = getattr(config, "loss_weight", 1.)
        self.cls_loss_weight = getattr(config, "cls_loss_weight", 1.)

        if self.distance_metric == "cosine":
            self.distance_metric_fn = self.cos_distance
        elif self.distance_metric == "euclidean":
            self.distance_metric_fn = self.euclidean_distance
        elif self.distance_metric == "bilinear":
            self.dropout = nn.Dropout(config.hidden_dropout_prob)
            self.bilinear = nn.Bilinear(config.hidden_size, config.hidden_size, 1, False)
            self.distance_metric_fn = lambda rep1, rep2: self.bilinear(
                self.dropout(rep1), self.dropout(rep2)).squeeze(-1)
        elif self.distance_metric == "mlp":
            self.dropout = nn.Dropout(config.hidden_dropout_prob)
            self.proj_mlp = nn.Linear(config.hidden_size * 4, 1)
            self.distance_metric_fn = lambda rep_src, rep_tgt: self.proj_mlp(self.dropout(torch.cat(
            [rep_src, rep_tgt, rep_src - rep_tgt, rep_src * rep_tgt], dim=-1))).squeeze(-1)
        elif self.distance_metric == "l2norm":
            act_fn = act_name2fn("gelu")
            self.linear11 = nn.Sequential(
                nn.Dropout(config.hidden_dropout_prob),
                nn.Linear(config.hidden_size, 2*config.hidden_size))
            self.linear12 = nn.Sequential(
                nn.Dropout(config.hidden_dropout_prob),
                nn.Linear(2*config.hidden_size, config.hidden_size))
            self.linear21 = nn.Sequential(
                nn.Dropout(config.hidden_dropout_prob),
                nn.Linear(config.hidden_size, 2*config.hidden_size))
            self.linear22 = nn.Sequential(
                nn.Dropout(config.hidden_dropout_prob),
                nn.Linear(2*config.hidden_size, config.hidden_size))
            self.distance_metric_fn = lambda rep1, rep2: self.euclidean_distance(
                self.linear12(act_fn(self.linear11(rep1))),
                self.linear22(act_fn(self.linear21(rep1))),
            )
        elif self.distance_metric == "l2norm_ly1":
            self.linear1 = nn.Sequential(
                nn.Dropout(config.hidden_dropout_prob),
                nn.Linear(config.hidden_size, config.hidden_size))
            self.linear2 = nn.Sequential(
                nn.Dropout(config.hidden_dropout_prob),
                nn.Linear(config.hidden_size, config.hidden_size))
            self.distance_metric_fn = lambda rep1, rep2: self.euclidean_distance(
                self.linear1(rep1), self.linear2(rep1))
        else:
            raise KeyError(self.distance_metric)

    def encoder(self, input_ids, attention_mask=None, token_type_ids=None, position_ids=None):
        # token_type_ids = None
        # position_ids = None
        outputs = self.roberta(input_ids,
                            attention_mask=attention_mask,
                            token_type_ids=None,
                            position_ids=position_ids)
        pooled_output = outputs[1]
        return pooled_output

    def classifier(self, rep_src, rep_tgt):
        cls_feature = torch.cat(
            [rep_src, rep_tgt, rep_src - rep_tgt, rep_src * rep_tgt], dim=-1
        )
        cls_feature = self.dropout(cls_feature)
        logits = self.proj(cls_feature)

        if logits.shape[-1] == 1:
            logits = logits.squeeze(-1)
        return logits

    def structCal(self,  headIdx, relIdx, tailIdx):
        e1 = self.E(headIdx)
        x = self.bn0(e1)
        x = self.input_dropout(x)
        x = x.view(-1, 1, e1.size(1))

        r = self.R(relIdx)
        W_mat = torch.mm(r, self.W.view(r.size(1), -1))
        W_mat = W_mat.view(-1, e1.size(1), e1.size(1))
        W_mat = self.hidden_dropout1(W_mat)

        x = torch.bmm(x, W_mat)
        x = x.view(-1, e1.size(1))
        x = self.bn1(x)
        x = self.hidden_dropout2(x)
        x = torch.mm(x, self.E.weight.transpose(1, 0))
        # 是实体数目个维度，
        pred = torch.sigmoid(x)
        return pred

    def init(self):
        # 权重初始化方法
        xavier_normal_(self.E.weight.data)
        xavier_normal_(self.R.weight.data)

    def forward(self, src_input_ids, tgt_input_ids,
                src_attention_mask=None, tgt_attention_mask=None,
                src_token_type_ids=None, tgt_token_type_ids=None,
                label_dict=None,pos_ratio=None, neg_ratio=None,
                headIdx=None, relIdx=None, tailIdx=None,
                *args, **kwargs):
        ##############################################################################
        pred = self.structCal(headIdx, relIdx, tailIdx)
        targets = np.zeros((len(tailIdx), self.ent_num))
        # 应该是一个batch的
        for idx, item in enumerate(tailIdx):
            targets[idx, item] = 1
        # pdb.set_trace()
        targets = torch.tensor(targets, dtype=torch.float).to(pred.device)
        structLoss = self.loss(pred, targets)
        ##############################################################################
        rep_src = self.encoder(src_input_ids, attention_mask=src_attention_mask, token_type_ids=src_token_type_ids)
        rep_tgt = self.encoder(tgt_input_ids, attention_mask=tgt_attention_mask, token_type_ids=tgt_token_type_ids)

        distances = self.distance_metric_fn(rep_src, rep_tgt)
        outputs = [distances, ]

        if label_dict is not None:
            # neg_src_input_ids : bz, neg_times, sl
            assert "neg_src_input_ids" in label_dict and "neg_src_attention_mask" in label_dict
            bs, nt, _ = list(label_dict["neg_src_input_ids"].shape)
            if "neg_src_token_type_ids" in label_dict:
                neg_src_token_type_ids = label_dict["neg_src_token_type_ids"].view(bs*nt, -1)
            else:
                neg_src_token_type_ids = None
            rep_neg_src = self.encoder(  # bs*nt,hn
                label_dict["neg_src_input_ids"].view(bs*nt, -1),
                attention_mask=label_dict["neg_src_attention_mask"].view(bs*nt, -1),
                token_type_ids=neg_src_token_type_ids)

            assert "neg_tgt_input_ids" in label_dict and "neg_tgt_attention_mask" in label_dict
            if "neg_tgt_token_type_ids" in label_dict:
                neg_tgt_token_type_ids = label_dict["neg_tgt_token_type_ids"].view(bs*nt, -1)
            else:
                neg_tgt_token_type_ids = None

            rep_neg_tgt = self.encoder(  # bs*nt,hn
                label_dict["neg_tgt_input_ids"].view(bs*nt, -1),
                attention_mask=label_dict["neg_tgt_attention_mask"].view(bs*nt, -1),
                token_type_ids=neg_tgt_token_type_ids)

            # pos_distances = distances.unsqueeze(1).expand(-1, nt).reshape(bs*nt)  # bs*nt
            # neg_distances = self.distance_metric_fn(rep_neg_src, rep_neg_tgt)  # bs*nt


            logits = self.classifier(  # [6bs,2]
                torch.cat([rep_src, rep_neg_src], dim=0),
                torch.cat([rep_tgt, rep_neg_tgt], dim=0),
            )

            # loss fn
            # 1. cls loss
            cls_loss_fn = nn.CrossEntropyLoss(
                weight=torch.tensor([1., self.pos_weight, ], dtype=logits.dtype, device=logits.device)
            )
            cls_loss = cls_loss_fn(
                logits,
                torch.cat([
                    logits.new_ones([rep_src.shape[0], ], dtype=torch.long),
                    logits.new_zeros([rep_tgt.shape[0], ], dtype=torch.long),
                    logits.new_zeros([rep_tgt.shape[0], ], dtype=torch.long),
                    logits.new_zeros([rep_tgt.shape[0], ], dtype=torch.long),
                    logits.new_zeros([rep_tgt.shape[0], ], dtype=torch.long),
                    logits.new_zeros([rep_tgt.shape[0], ], dtype=torch.long),],

                    dim=0,
                )
            )
            # 2. ranking loss
            # rk_losses = torch.relu(self.hinge_loss_margin + pos_distances - neg_distances)
            # rk_loss = torch.mean(rk_losses)
            pos_ratio = torch.mean(pos_ratio)
            pos_ratio /= 2
            loss = self.cls_loss_weight * cls_loss + pos_ratio * structLoss
            outputs.insert(0, loss)
        else:
            logits = self.classifier(  # [2bs,2]
                rep_src,
                rep_tgt,
            )

        return tuple(outputs)




