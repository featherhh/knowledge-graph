# coding:utf-8
import json

from torch.utils.data import DataLoader, Dataset
from cs_process import *
conf = Config()


class CsDataset(Dataset):
    def __init__(self,data_path):
        super().__init__()
        self.datas = [json.loads(line) for line in open(data_path, encoding='utf-8')]

    def __len__(self):
        return len(self.datas)

    def __getitem__(self, item):
        content = self.datas[item] # 根据索引取出一个样本，字典样式
        text = content["text"]
        spo_list = content["spo_list"]
        return text, spo_list


def get_data():
    train_dataset = CsDataset(data_path=conf.train_data_path)
    train_dataloader = DataLoader(dataset=train_dataset,
                                  shuffle=True,
                                  batch_size=conf.batch_size,
                                  collate_fn=collate_fn)

    dev_dataset = CsDataset(data_path=conf.dev_data_path)
    dev_dataloader = DataLoader(dataset=dev_dataset,
                                  shuffle=True,
                                  batch_size=conf.batch_size,
                                  collate_fn=collate_fn)

    test_dataset = CsDataset(data_path=conf.test_data_path)
    test_dataloader = DataLoader(dataset=test_dataset,
                                  shuffle=True,
                                  batch_size=conf.batch_size,
                                  collate_fn=collate_fn)
    return train_dataloader, dev_dataloader, test_dataloader

if __name__ == '__main__':
    train_dataloader, dev_dataloader, test_dataloader = get_data()
    for inputs, labels in train_dataloader:
        print(f'inputs-->{inputs}')
        print(f'labels-->{labels["obj_tails"].shape}')
        break

