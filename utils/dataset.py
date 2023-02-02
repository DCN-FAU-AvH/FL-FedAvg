import torch
import numpy as np
from torchvision import datasets, transforms
from utils.split import *
from models.networks import *


def load_dataset(args):
    """
    load and split dataset
    """
    if args.dataset == "mnist":
        trans_mnist = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
        )
        dataset_train = datasets.MNIST(
            "./data/mnist/", train=True, download=True, transform=trans_mnist
        )
        dataset_test = datasets.MNIST(
            "./data/mnist/", train=False, download=True, transform=trans_mnist
        )
        if args.iid == 1:
            dict_users = mnist_iid(dataset_train, args.num_users)
        elif args.iid == 0:
            dict_users = mnist_noniid(dataset_train, args.num_users)
        else:
            exit("Error: unrecognized split")
    else:
        exit("Error: unrecognized dataset")
    return dataset_train, dataset_test, dict_users


def load_model(args, dataset_train):
    """
    load different models
    """
    img_size = dataset_train[0][0].shape
    MLP_len_in = 1
    for x in img_size:
        MLP_len_in *= x

    if args.dataset == "mnist":
        if args.model == "cnn1":
            net_glob = CNN_M(args=args, dim_channels=5, dim_fc_hiddens=50).to(
                args.device
            )
        elif args.model == "cnn2":
            net_glob = CNN_M(args=args, dim_channels=32, dim_fc_hiddens=512).to(
                args.device
            )
        elif args.model == "mlp1":
            net_glob = MLP_1NN(
                dim_in=MLP_len_in, dim_hidden=200, dim_out=args.num_classes
            ).to(args.device)
        elif args.model == "mlp2":
            net_glob = MLP_2NN(
                dim_in=MLP_len_in, dim_hidden=200, dim_out=args.num_classes
            ).to(args.device)
        elif args.model == "resnet1":
            net_glob = ResNet_M(args=args, dim_channels=5, dim_fc_hiddens=50).to(
                args.device
            )
        elif args.model == "resnet2":
            net_glob = ResNet_M(args=args, dim_channels=32, dim_fc_hiddens=512).to(
                args.device
            )
        else:
            exit("Error: unrecognized model")
    return net_glob


def save_data(args, results_all, net_glob, round):
    """
    save results and models
    """
    params = f"{args.dataset}_rounds-{round}_iid-{args.iid}_{args.model}"
    # save results
    np.save(f"./results/FedAvg_{params}.npy", np.array(results_all))
    # save models
    torch.save(net_glob, f"./results/FedAvg_{params}.pt")


def exp_details(args):
    """
    print params for better understanding
    """
    print("\nExperimental details:")
    print(f"  Dataset: {args.dataset}")
    if args.iid:
        print("  Datatype: IID")
    else:
        print("  Datatype: Non-IID")
    print(f"  Model: {args.model}")
    print(f"  Rounds: {args.rounds}")
    print(f"  L_rate: {args.lr}")
    print(f"  Total number of users: {args.num_users}")
    print(f"  Fraction of users: {args.frac}")
    print(f"  Local batch size: {args.local_bs}")
    print(f"  Local epochs: {args.local_ep}\n")
    print(f"Running...\n")
