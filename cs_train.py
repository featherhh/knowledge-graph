# coding:utf-8
import os
import sys


import torch
import pandas as pd
from cs_CasrelModel import *
from cs_dataloader import *
from cs_process import *
from cs_config import *
from tqdm import tqdm
conf = Config()


# 定义模型训练函数
def model2train(model, train_iter, dev_iter, optimizer, conf):
    # 定义初始f1值为0
    best_triple_f1 = 0
    # 开始外部迭代
    for epoch in range(conf.epochs):
        best_triple_f1 = train_epoch(model, train_iter, dev_iter, optimizer, best_triple_f1, epoch)
    torch.save(model.state_dict(), './save_model/last_model.pth')

# 内部数据迭代函数
def train_epoch(model, train_iter, dev_iter, optimizer, best_triple_f1, epoch):
    for index, (inputs, labels) in enumerate(tqdm(train_iter, desc="Casrel训练")):
        model.train()
        # 将数据送入模型得到预测结果
        preds = model(**inputs)
        # 将计算预测结果和真实标签结果的损失
        loss = model.compute_loss(**preds, **labels)
        print(f'训练的损失--》{loss}')
        # 梯度清零
        # model.zero_grad()
        optimizer.zero_grad()
        # 反向传播
        loss.backward()
        # 梯度更新
        optimizer.step()
        # 每隔1500步进行模型的验证
        if index+1 % 500 == 0:
            torch.save(model.state_dict(), "./save_model/epoch_%s_step_%s.pth" % (epoch+1, index))
            results = model2dev(model, dev_iter)
            print(results[-1])
            if results[-2] > best_triple_f1:
                best_triple_f1 = results[-2]
                torch.save(model.state_dict(), './save_model/best_f1.pth')
                print('epoch:{},'
                      'step:{},'
                      'sub_precision:{:.4f}, '
                      'sub_recall:{:.4f}, '
                      'sub_f1:{:.4f}, '
                      'triple_precision:{:.4f}, '
                      'triple_recall:{:.4f}, '
                      'triple_f1:{:.4f},'
                      'train loss:{:.4f}'.format(epoch,
                                                 index,
                                                 results[0],
                                                 results[1],
                                                 results[2],
                                                 results[3],
                                                 results[4],
                                                 results[5],
                                                 loss.item()))
    return best_triple_f1

# 进行模型的
def model2dev(model, dev_iter):
    model.eval()
    # 创建一个dataframe对象：存储指标
    df = pd.DataFrame(columns=['TP', 'PRED', "REAL", 'p', 'r', 'f1'], index=['sub', 'triple'])
    df.fillna(0, inplace=True)
    # print(f'df-->{df}')
    # 迭代验证集
    for inputs, labels in tqdm(dev_iter, desc="Casrel验证"):
        logist = model(**inputs)
        pred_sub_heads = convert_score_to_zero_one(logist["pred_sub_heads"])
        pred_sub_tails = convert_score_to_zero_one(logist["pred_sub_tails"])
        sub_heads = convert_score_to_zero_one(labels["sub_heads"])
        sub_tails = convert_score_to_zero_one(labels["sub_tails"])
        obj_heads = convert_score_to_zero_one(labels['obj_heads'])
        obj_tails = convert_score_to_zero_one(labels['obj_tails'])
        pred_obj_heads = convert_score_to_zero_one(logist['pred_obj_heads'])
        pred_obj_tails = convert_score_to_zero_one(logist['pred_obj_tails'])
        batch_size = inputs['input_ids'].shape[0]
        # print(f'pred_obj_heads-->{pred_obj_heads.shape}')
        # print(f'obj_heads-->{obj_heads.shape}')
        for batch_idx in range(batch_size):
            #抽取预测的主实体
            pred_subs = extract_sub(pred_sub_heads[batch_idx].squeeze(),
                                    pred_sub_tails[batch_idx].squeeze())
            # print(f'pred_subs--》{pred_subs}')
            # 抽取真实的主实体
            true_subs = extract_sub(sub_heads[batch_idx].squeeze(),
                                    sub_tails[batch_idx].squeeze())
            # print(f'true_subs--》{true_subs}')
            # 抽取预测的客实体及对应关系
            pred_objs = extract_obj_and_rel(pred_obj_heads[batch_idx],
                                            pred_obj_tails[batch_idx])
            # print(f'pred_objs--》{pred_objs}')
            true_objs = extract_obj_and_rel(obj_heads[batch_idx],
                                            obj_tails[batch_idx])
            # print(f'true_objs--》{true_objs}')
            # 获取预测主实体的个数
            df.loc["sub", "PRED"] += len(pred_subs)
            # 获取真实的主实体的个数
            df.loc["sub", "REAL"] += len(true_subs)

            # 计算预测正确的主实体个数
            for true_sub in true_subs:
                if true_sub in pred_subs:
                    df.loc["sub", "TP"] += 1
            # 获取预测客实体及关系的个数
            df.loc["triple", "PRED"] += len(pred_objs)
            # 获取真实的客实体及关系的个数
            df.loc["triple", "REAL"] += len(true_objs)

            # 计算预测正确的客实体及关系的个数
            for true_obj in true_objs:
                if true_obj in pred_objs:
                    df.loc["triple", "TP"] += 1
    # 计算主实体指标
    # 计算精确率
    sub_precision = df.loc["sub", "TP"] / (df.loc["sub", "PRED"] + 1e-9)
    df.loc["sub", "p"] = sub_precision
    # 计算召回率
    sub_recall = df.loc["sub", "TP"] / (df.loc["sub", "REAL"] + 1e-9)
    df.loc["sub", "r"] = sub_recall
    #  f1值计算
    sub_f1 = 2*sub_precision*sub_recall / (sub_precision+sub_recall+1e-9)
    df.loc["sub", "f1"] = sub_f1

    # 计算客实体指标
    # 计算精确率
    obj_precision = df.loc["triple", "TP"] / (df.loc["triple", "PRED"] + 1e-9)
    df.loc["triple", "p"] = obj_precision
    # 计算召回率
    obj_recall = df.loc["triple", "TP"] / (df.loc["triple", "REAL"] + 1e-9)
    df.loc["triple", "r"] = obj_recall
    #  f1值计算
    obj_f1 = 2*obj_precision*obj_recall / (obj_precision+obj_recall+1e-9)
    df.loc["triple", "f1"] = obj_f1
    return sub_precision, sub_recall, sub_f1, obj_precision, obj_recall, obj_f1, df

if __name__ == '__main__':
    model, optimizer, sheduler, conf.device = load_model(conf)
    train_iter, dev_iter, _ = get_data()
    model2train(model, train_iter, dev_iter, optimizer, conf)