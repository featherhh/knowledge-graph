# coding:utf-8
from cs_config import *
import torch
from random import choice
from collections import defaultdict
conf = Config()
from rich import print


def find_head_idx(source, target):
    # source:原始一个句子的id；target代表句子中实体（id）
    # print(f'source--》{source}')
    # print(f'target--》{target}')
    target_len = len(target)
    for i in range(len(source)):
        if source[i: i+target_len] == target:
            return i
    return -1


def create_label(inner_triples, inner_input_ids, seq_len):
    '''
    获取每个样本的：主实体长度、主实体开始和结束位置张量表示、客实体以及对应关系实现张量表示
    '''
    # print(f'inner_input_ids--》{inner_input_ids}')
    inner_sub_heads, inner_sub_tails = torch.zeros(seq_len), torch.zeros(seq_len)
    inner_obj_heads = torch.zeros((seq_len, conf.num_rel))
    inner_obj_tails = torch.zeros((seq_len, conf.num_rel))
    inner_sub_head2tail = torch.zeros(seq_len)
    # 随机抽取一个实体，从开头一个词到末尾词的索引赋值
    # 因为数据预处理代码还待优化,会有不存在关系三元组的情况，
    # 初始化一个主词的长度为1，即没有主词默认主词长度为1，
    # 防止零除报错,初始化任何非零数字都可以，没有主词分子是全零矩阵
    inner_sub_len = torch.tensor([1], dtype=torch.float)
    # 主词到谓词的映射(s2ro_map本身就是个字典)
    s2ro_map = defaultdict(list)
    for inner_triple in inner_triples:
        # 对一个样本中每一个spo三元组进行数字id的表示
        sub1 = conf.tokenizer(inner_triple["subject"], add_special_tokens=False)["input_ids"]
        obj1 = conf.tokenizer(inner_triple["object"], add_special_tokens=False)["input_ids"]
        rel1 = conf.rel_vocab.to_index(inner_triple["predicate"])
        inner_triple = (sub1, rel1, obj1)
        # print(f'编码之后的inner_triple--》{inner_triple[0]}')
        # print(f'编码之后的inner_triple[0]--》{len(inner_triple[0])}')
        # 分别获取主客实体的开始索引位置
        sub_head_idx = find_head_idx(inner_input_ids, inner_triple[0])
        obj_head_idx = find_head_idx(inner_input_ids, inner_triple[2])
        # print(f'sub_head_idx--》{sub_head_idx}')
        # print(f'obj_head_idx--》{obj_head_idx}')
        if sub_head_idx != -1 and obj_head_idx != -1:
            sub = (sub_head_idx, sub_head_idx+len(inner_triple[0])-1)
            s2ro_map[sub].append((obj_head_idx, obj_head_idx+len(inner_triple[2])-1, inner_triple[1]))
    # print(f's2ro_map--》{s2ro_map}')

    if s2ro_map:
        for s in s2ro_map:
            # s代表主实体
            inner_sub_heads[s[0]] = 1
            inner_sub_tails[s[1]] = 1
        # 随机选择其中的一个主体来进行
        sub_head_idx, sub_tail_idx = choice(list(s2ro_map.keys()))
        # print(f'sub_head_idx--》{sub_head_idx}')
        # print(f'sub_tail_idx--》{sub_tail_idx}')
        inner_sub_head2tail[sub_head_idx: sub_tail_idx+1] = 1
        inner_sub_len = torch.tensor([sub_tail_idx-sub_head_idx+1],  dtype=torch.float)
        for obr in s2ro_map.get((sub_head_idx, sub_tail_idx), []):
            inner_obj_heads[obr[0]][obr[2]] = 1
            inner_obj_tails[obr[1]][obr[2]] = 1
    return inner_sub_len, inner_sub_head2tail, inner_sub_heads, inner_sub_tails, inner_obj_heads, inner_obj_tails


def collate_fn(datas):
    text_list = [data[0] for data in datas]
    triple = [data[1] for data in datas]
    # 对一个批次的文本进行编码： padding=True,意思按照最长句子进行补齐
    inputs = conf.tokenizer.batch_encode_plus(text_list, padding=True)
    # print(f'inputs--》{inputs}')
    # 一个批次的样本个数
    batch_size = len(inputs["input_ids"])
    # 获取每个样本编码之后的长度
    seq_len = len(inputs["input_ids"][0])
    # 存放主实体开始位置信息
    sub_heads = []
    # 存放主实体结束位置信息
    sub_tails = []
    # 存放客实体开始位置信息（含关系）
    obj_heads = []
    # 存放客实体结束位置信息（含关系）
    obj_tails = []
    # 存储主实体长度
    sub_len = []
    sub_head2tail = []
    # 遍历每一个样本，进行实体信息的转换
    for batch_idx in range(batch_size):
        # 根据索引取出当前样本对应的编码后的结果（ids）
        inner_input_ids = inputs["input_ids"][batch_idx]
        # print(f'inner_input_ids--》{inner_input_ids}')
        # 根据索引取出当前样本对应的三元组spo——list
        inner_triples = triple[batch_idx]
        # print(f'inner_triples--》{inner_triples}')
        # 获取每个样本的：主实体长度、主实体开始和结束位置张量表示、客实体以及对应关系实现张量表示
        results = create_label(inner_triples, inner_input_ids, seq_len)
        sub_len.append(results[0])
        sub_head2tail.append(results[1])
        sub_heads.append(results[2])
        sub_tails.append(results[3])
        obj_heads.append(results[4])
        obj_tails.append(results[5])


    input_ids = torch.tensor(inputs["input_ids"]).to(conf.device)
    # print(f'input_ids--》{input_ids.shape}')
    attention_mask = torch.tensor(inputs["attention_mask"]).to(conf.device)
    # print(f'attention_mask--》{attention_mask.shape}')
    sub_heads = torch.stack(sub_heads).to(conf.device)
    # print(f'sub_heads--》{sub_heads.shape}')
    sub_tails = torch.stack(sub_tails).to(conf.device)
    # print(f'sub_tails--》{sub_tails.shape}')
    sub_head2tail = torch.stack(sub_head2tail).to(conf.device)
    # print(f'sub_head2tail--》{sub_head2tail.shape}')
    obj_heads = torch.stack(obj_heads).to(conf.device)
    # print(f'obj_heads--》{obj_heads.shape}')
    obj_tails = torch.stack(obj_tails).to(conf.device)
    # print(f'obj_tails-->{obj_tails.shape}')
    sub_len = torch.stack(sub_len).to(conf.device)
    # print(f'sub_len-->{sub_len.shape}')
    inputs = {
        'input_ids': input_ids,
        'mask': attention_mask,
        'sub_head2tail': sub_head2tail,
        'sub_len': sub_len
    }
    labels = {
        'sub_heads': sub_heads,
        'sub_tails': sub_tails,
        'obj_heads': obj_heads,
        'obj_tails': obj_tails
    }

    return inputs, labels


def convert_score_to_zero_one(tensor1):
    # 将预测的tensor1值，概率大于等于0.5的设置为1，否则为0
    tensor1[tensor1 >= 0.5] = 1
    tensor1[tensor1 < 0.5] = 0
    return tensor1


def extract_sub(pred_sub_heads, pred_sub_tails):
    '''
    :param pred_sub_heads: 模型预测出的主实体开头位置
    :param pred_sub_tails: 模型预测出的主实体尾部位置
    :return: subs列表里面对应的所有实体【head, tail】
    '''
    # print(f'pred_sub_heads-->{pred_sub_heads}')
    # print(f'pred_sub_tails-->{pred_sub_tails}')
    subs = [] # 存储所有的主实体（start, end）
    # 获取所有位置为1的索引
    heads = torch.arange(0, len(pred_sub_heads),device=conf.device)[pred_sub_heads==1]
    # print(f'heads--》{heads}')
    tails = torch.arange(0, len(pred_sub_tails),device=conf.device)[pred_sub_tails==1]
    # print(f'tails--》{tails}')
    for head, tail in zip(heads, tails):
        if tail >= head:
            subs.append((head.item(), tail.item()))

    return subs


def extract_obj_and_rel(obj_heads, obj_tails):
    '''
    :param obj_heads:  模型预测出的从实体开头位置以及关系类型：[seq_len, rel_count]
    :param obj_tails:  模型预测出的从实体尾部位置以及关系类型  [seq_len, rel_count]
    :return: obj_and_rels：元素形状：(rel_index, start_index, end_index)
    '''
    obj_heads = obj_heads.T
    # print(f'转置之后的结果obj_heads--》{obj_heads.shape}')
    obj_tails = obj_tails.T
    # print(f'转置之后的结果obj_tails--》{obj_tails.shape}')
    rel_count = obj_heads.shape[0] # 关系类别的总个数
    obj_and_rels = [] # 存储所有的客实体及关系的
    for rel_index in range(rel_count):
        obj_head = obj_heads[rel_index]
        obj_tail = obj_tails[rel_index]
        objs = extract_sub(obj_head, obj_tail) # [(start, end),(..)]
        if objs:
            for obj in objs:
                start_idx, end_idx = obj
                obj_and_rels.append((rel_index, start_idx, end_idx ))
    return obj_and_rels

