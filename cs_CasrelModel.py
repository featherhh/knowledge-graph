# coding:utf-8
import torch
import torch.nn as nn
from transformers import BertModel
from torch.optim import AdamW
from cs_config import *
from cs_dataloader import *
conf = Config()

class CasRel(nn.Module):
    def __init__(self, conf):
        super().__init__()
        # 定义预训练模型层
        self.bert = BertModel.from_pretrained(conf.bert_path)
        # 定义第一个全连接层：识别主实体的开始位置
        self.sub_heads_linear = nn.Linear(conf.bert_dim, 1)
        # 定义第二个全连接层：识别主实体的结束位置
        self.sub_tails_linear = nn.Linear(conf.bert_dim, 1)
        # 定义第三个全连接层：识别客实体的开始位置以及关系类型
        self.obj_heads_linear = nn.Linear(conf.bert_dim, conf.num_rel)
        # 定义第四个全连接层：识别客实体的尾部位置以及关系类型
        self.obj_tails_linear = nn.Linear(conf.bert_dim, conf.num_rel)
        # self.obj_tails = nn.Linear(conf.bert_dim, conf.num_rel)

    def get_subs(self, encoded_text):
        # 预测出主实体的开始位置
        pred_sub_heads = torch.sigmoid(self.sub_heads_linear(encoded_text))
        # 预测出主实体的结束位置
        pred_sub_tails = torch.sigmoid(self.sub_tails_linear(encoded_text))
        return pred_sub_heads, pred_sub_tails

    def get_encoded_text(self, input_ids, mask):
        bert_output = self.bert(input_ids=input_ids, attention_mask=mask)[0]
        # print(f'bert_output--》{bert_output.shape}')
        return bert_output

    def get_objs_for_specific_sub(self, sub_head2tail, sub_len, encoded_output):
        # sub_head2tail-shape-->[4, 1, 80];sub_len-->shape-->[4,1]:encoded_output-->shape-->[4, 80, 768]
        # 1. 将主实体的信息从encoded_output中筛选出来:sub-->[4, 1, 768]--》只筛选出了，一个批次4个样本，每个样本1个实体，对应的768维度向量
        sub = torch.matmul(sub_head2tail, encoded_output)
        # 2. 平均上述sub信息:[4,1,768]
        sub_len = sub_len.unsqueeze(dim=1)
        sub = sub / sub_len
        # 3. 融合原始的bert编码之后的结果：（已经突出了当前的sub信息）:encoded_text-->shape->[4, 80, 768]
        encoded_text = encoded_output + sub
        # 4. 预测出客实体的开始位置和对应关系
        pred_obj_heads = torch.sigmoid(self.obj_heads_linear(encoded_text))
        # 预测出客实体的开始位置和对应关系
        pred_obj_tails = torch.sigmoid(self.obj_tails_linear(encoded_text))
        return pred_obj_heads, pred_obj_tails


    def forward(self, input_ids, mask, sub_head2tail, sub_len):
        # 输入数据的形状：input_ids=mask=[batch_size, seq_len]=[4, 80]
        # 输入数据的形状：sub_head2tail=[batch_size, seq_len]=[4, 80]
        # 输入数据的形状: sub_len=[batch_size, 1] = [4, 1]
        # 1. 将原始文本进行bert编码:encoded_output-->[4, 80, 768]
        encoded_output = self.get_encoded_text(input_ids, mask)
        # 2. 将编码之后的结果送入get_subs函数，预测主实体开始和结束位置
        # 形状：pred_sub_heads--》[4, 80,1]; pred_sub_tails-->[4, 80, 1]
        pred_sub_heads, pred_sub_tails = self.get_subs(encoded_output)
        # print(f'pred_sub_heads--》{pred_sub_heads.shape}')
        # print(f'pred_sub_tails--》{pred_sub_tails.shape}')
        # 3. 将bert模型编码后的结果融合主实体的信息，进行客实体和对应关系的解码
        sub_head2tail = sub_head2tail.unsqueeze(1)
        pred_obj_heads, pred_obj_tails = self.get_objs_for_specific_sub(sub_head2tail, sub_len, encoded_output)
        # print(f'pred_obj_heads--》{pred_obj_heads.shape}')
        # print(f'pred_obj_tails--》{pred_obj_tails.shape}')
        result_dict = {'pred_sub_heads': pred_sub_heads,
                       'pred_sub_tails': pred_sub_tails,
                       'pred_obj_heads': pred_obj_heads,
                       'pred_obj_tails': pred_obj_tails,
                       'mask': mask}
        return result_dict

    def compute_loss(self,
                     pred_sub_heads, pred_sub_tails,
                     pred_obj_heads, pred_obj_tails,
                     mask,
                     sub_heads, sub_tails,
                     obj_heads, obj_tails):
        '''
        计算损失
        :param pred_sub_heads:[4, 80, 1]
        :param pred_sub_tails:[4, 80, 1]
        :param pred_obj_heads:[4, 80, 18]
        :param pred_obj_tails:[4, 80, 18]
        :param mask: shape-->[4, 80]
        :param sub_heads: shape-->[4, 80]
        :param sub_tails: shape-->[4, 80]
        :param obj_heads: shape-->[4, 80, 18]
        :param obj_tails: shape-->[4, 80, 18]
        :return:
        '''
        # 获取关系类别总数:rel_count=18
        rel_count = obj_heads.shape[-1]
        # 将mask进行升维
        rel_mask = mask.unsqueeze(-1).repeat(1, 1, rel_count)
        # 计算主实体开始位置的损失
        loss1 = self.loss(pred_sub_heads, sub_heads, mask)
        # 计算主实体结束位置的损失
        loss2 = self.loss(pred_sub_tails, sub_tails, mask)
        # 计算客实体开始位置及关系的损失
        loss3 = self.loss(pred_obj_heads, obj_heads, rel_mask)
        # 计算客实体结束位置及关系的损失
        loss4 = self.loss(pred_obj_tails, obj_tails, rel_mask)
        return loss1+loss2+loss3+loss4

    def loss(self, pred, gold, mask):
        pred = pred.squeeze(-1)
        my_loss = nn.BCELoss(reduction='none')(pred, gold)
        # print(f'my_loss--》{my_loss}')
        # print(f'my_loss--》{my_loss.shape}')
        last_loss = torch.sum(my_loss * mask) / torch.sum(mask)
        return last_loss


def load_model(conf):
    model = CasRel(conf).to(conf.device)
    # print(f'model-->{model}')
    # named_parameters()获取模型中的参数和参数名字
    param_optimzer = list(model.named_parameters())
    # print(f'param_optimzer--》{param_optimzer}')
    # print(f'param_optimzer--》{len(param_optimzer)}')
    # no_decay中存放不进行权重衰减的参数{因为bert官方代码对这三项免于正则化}
    no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
    # any()函数用于判断给定的可迭代参数iterable是否全部为False，则返回False，如果有一个为True，则返回True
    # 判断param_optimizer中所有的参数。如果不在no_decay中，则进行权重衰减;如果在no_decay中，则不进行权重衰减
    optimizer_grouped_parameters = [
        {"params": [p for n, p in param_optimzer if not any(nd in n for nd in no_decay)], "weight_decay": 0.01},
        {"params": [p for n, p in param_optimzer if any(nd in n for nd in no_decay)], "weight_decay": 0.0}]
    optimizer = AdamW(optimizer_grouped_parameters, lr=conf.learning_rate, eps=10e-8)
    sheduler = None
    return model, optimizer, sheduler, conf.device
if __name__ == '__main__':
    load_model(conf)


