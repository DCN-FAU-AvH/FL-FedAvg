#!/usr/bin/env python
# -*- coding: utf-8 -*-
# basic implementation of the FL-FedAvg algorithm (https://arxiv.org/abs/1602.05629)
# adapted from https://github.com/shaoxiongji/federated-learning

from utils.arguments import args_parser
from models.server import FedAvg

if __name__ == "__main__":
    args = args_parser()
    FedAvg(args)
