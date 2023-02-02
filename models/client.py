import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


class DatasetSplit(Dataset):
    """
    assign data to client
    """

    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = list(idxs)

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        return image, label


class LocalUpdate(object):
    """
    client's local update process
    """

    def __init__(self, args, dataset=None, idxs=None):
        self.args = args
        self.loss_func = nn.CrossEntropyLoss()
        self.selected_clients = []
        self.loader_train = DataLoader(
            DatasetSplit(dataset, idxs), batch_size=self.args.local_bs, shuffle=True
        )

    def train(self, model, round):
        model.train()
        lr_l = self.args.lr * pow(self.args.decay, round)
        if self.args.optimizer == "sgd":
            optimizer = torch.optim.SGD(
                model.parameters(), lr=lr_l, momentum=self.args.momentum
            )
        elif self.args.optimizer == "adam":
            optimizer = torch.optim.Adam(model.parameters(), lr=lr_l, weight_decay=1e-4)

        epoch_loss = []
        for iter in range(self.args.local_ep):
            batch_loss = []
            for images, labels in self.loader_train:
                images, labels = images.to(self.args.device), labels.to(
                    self.args.device
                )
                model.zero_grad()
                log_probs = model(images)
                loss = self.loss_func(log_probs, labels)
                loss.backward()
                optimizer.step()
                batch_loss.append(loss.item())
            epoch_loss.append(sum(batch_loss) / len(batch_loss))
        return model.state_dict(), sum(epoch_loss) / len(epoch_loss)


def cal_loss(args, model, dataset, dict_users):
    """
    calculate the loss
    """
    model.to(args.device)
    model.train()
    loss_all = []
    batch_size = int(len(dataset) / args.num_users)
    for idx in dict_users:
        loader_train = DataLoader(
            DatasetSplit(dataset, dict_users[idx]), batch_size=batch_size
        )
        for images, labels in loader_train:
            images, labels = images.to(args.device), labels.to(args.device)
            model.zero_grad()
            log_probs = model(images)
            loss_idx = F.cross_entropy(log_probs, labels).item()
            loss_all.append(loss_idx)
    loss_avg = sum(loss_all) / len(loss_all)
    return loss_avg
