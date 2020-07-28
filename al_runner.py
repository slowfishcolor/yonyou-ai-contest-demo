# !/usr/bin/env python
# coding: utf-8

import traceback
import argparse
import json
import sys
import yaml
import os
from algorithm import Algorithm


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("-p", "--path", default="竞赛用数据.csv", type=str, help="测试数据文件")
args = parser.parse_args()

algorithm_py = "algorithm.py"
algorithm_yaml = "algorithm.yaml"
model_yaml = "model.yaml"


class DataReader:
    def __init__(self, data):
        self.data = data
        self.len = len(self.data)
        self.offset = 0

    def __iter__(self):
        return self

    def __next__(self):
        if self.offset >= self.len:
            raise StopIteration
        try:
            raw_data = self.data[self.offset]
            data = raw_data.rstrip('\n').split(',')
            self.offset += 1
            return data
        except Exception as e:
            raise StopIteration

    @property
    def iterable(self):
        return True

if __name__ == '__main__':
    try:
        with open(algorithm_yaml, encoding="utf-8") as f:
            al_cfg = yaml.load(f, Loader=yaml.Loader)
        with open(model_yaml, encoding="utf-8") as f:
            md_cfg = yaml.load(f, Loader=yaml.Loader)

        print('读取训练数据')
        data = open(args.path).readlines()
        data_len = len(data)
        train_data_len = int(data_len * 0.7)
        train_reader = DataReader(data[1:train_data_len])
        val_reader = DataReader(data[train_data_len:])
        alg = Algorithm({})
        alg.train(train_reader, val_reader)
    except Exception as e:
        stack_info = traceback.format_exc()
        print("失败:\n{}".format(str(stack_info)))
