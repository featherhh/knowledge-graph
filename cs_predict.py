# coding:utf-8
import json

import torch

from cs_CasrelModel import *
from cs_process import *
conf = Config()

# 加载训练好的模型

def load_model():
    # 实例化模型
    model = CasRel(conf).to(conf.device)
    model.load_state_dict(torch.load("./save_model/last_model_old.pth", map_location='cpu'), strict=False)
    return model

# def load_stu_model():
#     # 实例化模型
#     model = CasRel(conf).to(conf.device)
#     model.load_state_dict(torch.load("./save_model/student_best_f1.pth", map_location='cpu'))
#     return model


def get_inputs(sample, model):
    inputs = conf.tokenizer(sample)
    # print(f'inputs--》{inputs}')
    # input_ids-->[batch_size, seq_len]-->[1, 13]
    input_ids = torch.tensor([inputs["input_ids"]]).to(conf.device)
    mask = torch.tensor([inputs["attention_mask"]]).to(conf.device)
    # 获取当前样本的句子长度
    seq_len = len(inputs["input_ids"])
    inner_sub_head2tail = torch.zeros(seq_len)
    inner_sub_len = torch.tensor([1], dtype=torch.float)
    # 获取模型预测的主实体位置信息
    model.eval()
    with torch.no_grad():
        # 获取bert编码之后的结果
        bert_encoded = model.get_encoded_text(input_ids, mask)
        # print(f'bert_encoded--》{bert_encoded.shape}')
        # 根据编码结果获取模型预测的主实体的开始位置分数和结束位置分数
        pred_sub_heads, pred_sub_tails = model.get_subs(bert_encoded)
        # print(f'pred_sub_heads--》{pred_sub_heads.shape}')
        # print(f'pred_sub_tails--》{pred_sub_tails.shape}')
        #
        pred_sub_heads = convert_score_to_zero_one(pred_sub_heads)
        pred_sub_tails = convert_score_to_zero_one(pred_sub_tails)
        pred_subs = extract_sub(pred_sub_heads.squeeze(), pred_sub_tails.squeeze())
        # print(f'pred_subs-->{pred_subs}')
        if len(pred_subs) != 0:
            # 这里我们只拿了第一个主实体来进行预测的，如果想要严谨的化，需要循环遍历所有主实体，然后去预测对应的客实体和关系
            # 但是，我们大部分样本主实体都只是一个；而且只要掌握一个主实体预测的思想，其他类同
            sub_head_idx = pred_subs[0][0]
            sub_tail_idx = pred_subs[0][1]
            # 获取主体长度以及对主体位置全部赋值为1
            inner_sub_head2tail[sub_head_idx: sub_tail_idx+1] = 1
            inner_sub_len = torch.tensor([sub_tail_idx-sub_head_idx+1], dtype=torch.float)

    sub_len = inner_sub_len.unsqueeze(0).to(conf.device)
    # print(f'sub_len--》{sub_len.shape}')
    sub_head2tail = inner_sub_head2tail.unsqueeze(0).to(conf.device)
    # print(f'sub_head2tail--》{sub_head2tail.shape}')

    inputs = {'input_ids': input_ids,
              'mask': mask,
              'sub_head2tail': sub_head2tail,
              'sub_len': sub_len}
    return inputs, model


def model2predict(sample, model):
    # 1.获取关系字典
    with open(conf.rel_dict_path, 'r', encoding='utf-8') as fr:
        rel_id2word = json.load(fr)
    # print(f'rel_id2word--》{rel_id2word}')
    # 2. 获取模型的输入
    inputs, model = get_inputs(sample, model)
    # print(f'inputs--》{inputs}')
    logist = model(**inputs)
    pred_sub_heads = convert_score_to_zero_one(logist['pred_sub_heads'])
    pred_sub_tails = convert_score_to_zero_one(logist['pred_sub_tails'])
    pred_obj_heads = convert_score_to_zero_one(logist['pred_obj_heads'])
    pred_obj_tails = convert_score_to_zero_one(logist['pred_obj_tails'])
    # print(f"logist['pred_sub_heads']-->{logist['pred_sub_heads'].shape}")
    # print(f"logist['pred_obj_heads']-->{logist['pred_obj_heads'].shape}")
    new_dict = {}
    spo_list = []

    ids = inputs["input_ids"][0]
    # print(f'ids-->{ids}')
    text_list = conf.tokenizer.convert_ids_to_tokens(ids)
    # print(f'text_list--》{text_list}')
    sentence = ''.join(text_list[1: -1])
    # print(f'sentence---》{sentence}')
    pred_subs = extract_sub(pred_sub_heads.squeeze(), pred_sub_tails.squeeze())
    # print(f'pred_subs--》{pred_subs}')
    pred_objs = extract_obj_and_rel(pred_obj_heads[0], pred_obj_tails[0])
    # print(f'pred_objs--》{pred_objs}')
    if len(pred_subs) == 0 or len(pred_objs) == 0:
        print('没有识别出结果')
        return {}
    if len(pred_objs) > len(pred_subs):
        pred_subs = pred_subs * len(pred_objs)

    for sub, rel_obj in zip(pred_subs, pred_objs):
        # print(f'sub--》{sub}')
        # print(f'rel_obj--》{rel_obj}')
        sub_spo = {}
        sub_head, sub_tail = sub
        sub = ''.join(text_list[sub_head: sub_tail + 1])
        # print(f'sub--》{sub}')
        if '[PAD]' in sub:
            continue
        sub_spo['subject'] = sub
        relation = rel_id2word[str(rel_obj[0])]
        # print(f'relation--》{relation}')
        obj_head, obj_tail = rel_obj[1], rel_obj[2]
        obj = ''.join(text_list[obj_head: obj_tail + 1])
        # print(f'obj--》{obj}')
        if '[PAD]' in obj:
            continue
        sub_spo["predicate"] = relation
        sub_spo["object"] = obj
        # print(f'sub_spo--》{sub_spo}')
        spo_list.append(sub_spo)

    new_dict["text"] = sentence
    new_dict["spo_list"] = spo_list

    return new_dict





if __name__ == '__main__':
    sample = '徐月焕，男，汉族，1961年7月生，浙江淳安人，无党派，1982年8月参加工作，中央党校大学学历'
    model = load_model()
    result = model2predict(sample, model)
    print(result)
