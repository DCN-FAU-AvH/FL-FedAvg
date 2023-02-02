import argparse


def args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="mnist", help="name of dataset")
    parser.add_argument("--iid", type=int, default=0, help="0 for noniid and 1 for iid")
    parser.add_argument("--model", type=str, default="mlp1", help="model name")
    parser.add_argument("--rounds", type=int, default=500, help="rounds of training")
    parser.add_argument("--num_users", type=int, default=100, help="number of users: K")
    parser.add_argument(
        "--frac", type=float, default=0.1, help="the fraction of clients: C"
    )
    parser.add_argument(
        "--local_ep", type=int, default=5, help="the number of local epochs: E"
    )
    parser.add_argument("--local_bs", type=int, default=10, help="local batch size: B")
    parser.add_argument("--lr", type=float, default=0.01, help="learning rate")
    parser.add_argument(
        "--num_channels", type=int, default=1, help="number of channels of imges"
    )
    parser.add_argument("--channels", type=int, default=32, help="num of channels")
    parser.add_argument("--num_classes", type=int, default=10, help="number of classes")
    parser.add_argument(
        "--decay", type=float, default=1, help="learning rate decay per global round"
    )
    parser.add_argument("--bs", type=int, default=128, help="test batch size")
    parser.add_argument(
        "--selected_clients",
        type=int,
        default=0,
        help="aggregation over selected clients",
    )
    parser.add_argument(
        "--momentum", type=float, default=0.5, help="SGD momentum (default: 0.5)"
    )
    parser.add_argument(
        "--split",
        type=str,
        default="user",
        help="train-test split type, user or sample",
    )
    parser.add_argument(
        "--optimizer", type=str, default="sgd", help="type of optimizer"
    )
    parser.add_argument("--gpu", type=int, default=0, help="GPU ID, -1 for CPU")
    parser.add_argument("--verbose", type=int, default=0, help="verbose")
    parser.add_argument("--seed", type=int, default=1, help="random seed (default: 1)")
    args = parser.parse_args()
    return args
