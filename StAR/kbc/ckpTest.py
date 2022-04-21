import torch

from kbc.models import BertForPairScoring, RobertaForPairScoring
from transformers import BertConfig, BertTokenizer, RobertaConfig, RobertaTokenizer, logger


tokenizer = BertTokenizer.from_pretrained('../result/My_WN18RR_roberta-large/vocab.json')
model = RobertaForPairScoring.from_pretrained('../result/My_WN18RR_roberta-large/')