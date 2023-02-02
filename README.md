# An implementation of the FedAvg algorithm

This repo is a basic PyTorch implementation of the Federated Learning [FedAvg](https://arxiv.org/abs/1602.05629) algorithm.

The code is adapted from [GitHub-FL](https://github.com/shaoxiongji/federated-learning).

## Run

It is easy to run with:

```bash
python fedavg.py --dataset mnist --iid 0 --model mlp1 --rounds 100
```

You can also check `./utils/arguments.py` to personalize different parameters.

## Example results

<img src="/results/results.jpg" alt="results" width=50%/>
