import matplotlib.pyplot as plt
import numpy as np

data_rounds = 100
plot_rounds = data_rounds

cnn1_noniid = np.load('./results/FedAvg_mnist_rounds-' + str(data_rounds) + '_iid-0_cnn1.npy', allow_pickle=True).tolist()
mlp1_noniid = np.load('./results/FedAvg_mnist_rounds-' + str(data_rounds) + '_iid-0_mlp1.npy', allow_pickle=True).tolist()
resnet1_noniid = np.load('./results/FedAvg_mnist_rounds-' + str(data_rounds) + '_iid-0_resnet1.npy', allow_pickle=True).tolist()
cnn2_noniid = np.load('./results/FedAvg_mnist_rounds-' + str(data_rounds) + '_iid-0_cnn2.npy', allow_pickle=True).tolist()
mlp2_noniid = np.load('./results/FedAvg_mnist_rounds-' + str(data_rounds) + '_iid-0_mlp2.npy', allow_pickle=True).tolist()
resnet2_noniid = np.load('./results/FedAvg_mnist_rounds-' + str(data_rounds) + '_iid-0_resnet2.npy', allow_pickle=True).tolist()

# mlp1_iid = np.load('./results/FedAvg_mnist_rounds-' + str(data_rounds) + '_iid-1_mlp1.npy', allow_pickle=True).tolist()
# cnn1_iid = np.load('./results/FedAvg_mnist_rounds-' + str(data_rounds) + '_iid-1_cnn1.npy', allow_pickle=True).tolist()
# resnet1_iid = np.load('./results/FedAvg_mnist_rounds-' + str(data_rounds) + '_iid-1_resnet1.npy', allow_pickle=True).tolist()
# mlp2_iid = np.load('./results/FedAvg_mnist_rounds-' + str(data_rounds) + '_iid-1_mlp2.npy', allow_pickle=True).tolist()
# cnn2_iid = np.load('./results/FedAvg_mnist_rounds-' + str(data_rounds) + '_iid-1_cnn2.npy', allow_pickle=True).tolist()
# resnet2_iid = np.load('./results/FedAvg_mnist_rounds-' + str(data_rounds) + '_iid-1_resnet2.npy', allow_pickle=True).tolist()

if 'cnn1_noniid' in locals():
    # Test accuracy Non-IID
    plt.figure()
    plt.plot(range(1, plot_rounds + 1), mlp1_noniid[2][0: plot_rounds], '', label='FedAvg-MLP1')
    plt.plot(range(1, plot_rounds + 1), mlp2_noniid[2][0: plot_rounds], '', label='FedAvg-MLP2')
    plt.plot(range(1, plot_rounds + 1), cnn1_noniid[2][0: plot_rounds], '', label='FedAvg-CNN1')
    plt.plot(range(1, plot_rounds + 1), cnn2_noniid[2][0: plot_rounds], '', label='FedAvg-CNN2')
    plt.plot(range(1, plot_rounds + 1), resnet1_noniid[2][0: plot_rounds], '', label='FedAvg-ResNet1')
    plt.plot(range(1, plot_rounds + 1), resnet2_noniid[2][0: plot_rounds], '', label='FedAvg-ResNet2')
    # plt.legend(ncol=6)
    plt.legend()
    plt.xlabel('Rounds')
    plt.ylabel('Test accuracy')
    plt.title('Test accuracy vs training rounds for Non-IID data')

    # Train loss Non-IID
    plt.figure()
    plt.plot(range(1, plot_rounds + 1), mlp1_noniid[0][0: plot_rounds], '', label='FedAvg-MLP1')
    plt.plot(range(1, plot_rounds + 1), mlp2_noniid[0][0: plot_rounds], '', label='FedAvg-MLP2')
    plt.plot(range(1, plot_rounds + 1), cnn1_noniid[0][0: plot_rounds], '', label='FedAvg-CNN1')
    plt.plot(range(1, plot_rounds + 1), cnn2_noniid[0][0: plot_rounds], '', label='FedAvg-CNN2')
    plt.plot(range(1, plot_rounds + 1), resnet1_noniid[0][0: plot_rounds], '', label='FedAvg-ResNet1')
    plt.plot(range(1, plot_rounds + 1), resnet2_noniid[0][0: plot_rounds], '', label='FedAvg-ResNet2')
    plt.legend()
    plt.yscale('log')
    plt.xscale('log')
    plt.xlabel('Rounds')
    plt.ylabel('Train loss')
    plt.title('Train loss vs training rounds for Non-IID data')

    # test loss Non-IID
    plt.figure()
    plt.plot(range(1, plot_rounds + 1), mlp1_noniid[1][0: plot_rounds], '', label='FedAvg-MLP1')
    plt.plot(range(1, plot_rounds + 1), mlp2_noniid[1][0: plot_rounds], '', label='FedAvg-MLP2')
    plt.plot(range(1, plot_rounds + 1), cnn1_noniid[1][0: plot_rounds], '', label='FedAvg-CNN1')
    plt.plot(range(1, plot_rounds + 1), cnn2_noniid[1][0: plot_rounds], '', label='FedAvg-CNN2')
    plt.plot(range(1, plot_rounds + 1), resnet1_noniid[1][0: plot_rounds], '', label='FedAvg-ResNet1')
    plt.plot(range(1, plot_rounds + 1), resnet2_noniid[1][0: plot_rounds], '', label='FedAvg-ResNet2')
    plt.legend()
    plt.yscale('log')
    plt.xscale('log')
    plt.xlabel('Rounds')
    plt.ylabel('Test loss')
    plt.title('Test loss vs training rounds for Non-IID data')

if 'resnet2_iid' in locals():
    # Test accuracy IID
    plt.figure()
    plt.plot(range(1, plot_rounds + 1), mlp1_iid[2][0: plot_rounds], '', label='FedAvg-MLP1')
    plt.plot(range(1, plot_rounds + 1), mlp2_iid[2][0: plot_rounds], '', label='FedAvg-MLP2')
    plt.plot(range(1, plot_rounds + 1), cnn1_iid[2][0: plot_rounds], '', label='FedAvg-CNN1')
    plt.plot(range(1, plot_rounds + 1), cnn2_iid[2][0: plot_rounds], '', label='FedAvg-CNN2')
    plt.plot(range(1, plot_rounds + 1), resnet1_iid[2][0: plot_rounds], '', label='FedAvg-ResNet1')
    plt.plot(range(1, plot_rounds + 1), resnet2_iid[2][0: plot_rounds], '', label='FedAvg-ResNet2')
    plt.legend()
    plt.xlabel('Rounds')
    plt.ylabel('Test accuracy')
    plt.title('Test accuracy vs training rounds for IID data')

    # Train loss IID
    plt.figure()
    plt.plot(range(1, plot_rounds + 1), mlp1_iid[0][0: plot_rounds], '', label='FedAvg-MLP1')
    plt.plot(range(1, plot_rounds + 1), mlp2_iid[0][0: plot_rounds], '', label='FedAvg-MLP2')
    plt.plot(range(1, plot_rounds + 1), cnn1_iid[0][0: plot_rounds], '', label='FedAvg-CNN1')
    plt.plot(range(1, plot_rounds + 1), cnn2_iid[0][0: plot_rounds], '', label='FedAvg-CNN2')
    plt.plot(range(1, plot_rounds + 1), resnet1_iid[0][0: plot_rounds], '', label='FedAvg-ResNet1')
    plt.plot(range(1, plot_rounds + 1), resnet2_iid[0][0: plot_rounds], '', label='FedAvg-ResNet2')
    plt.legend()
    plt.yscale('log')
    plt.xscale('log')
    plt.xlabel('Rounds')
    plt.ylabel('Train loss')
    plt.title('Train loss vs training rounds for IID data')

    # test loss IID
    plt.figure()
    plt.plot(range(1, plot_rounds + 1), mlp1_iid[1][0: plot_rounds], '', label='FedAvg-MLP1')
    plt.plot(range(1, plot_rounds + 1), mlp2_iid[1][0: plot_rounds], '', label='FedAvg-MLP2')
    plt.plot(range(1, plot_rounds + 1), cnn1_iid[1][0: plot_rounds], '', label='FedAvg-CNN1')
    plt.plot(range(1, plot_rounds + 1), cnn2_iid[1][0: plot_rounds], '', label='FedAvg-CNN2')
    plt.plot(range(1, plot_rounds + 1), resnet1_iid[1][0: plot_rounds], '', label='FedAvg-ResNet1')
    plt.plot(range(1, plot_rounds + 1), resnet2_iid[1][0: plot_rounds], '', label='FedAvg-ResNet2')
    plt.legend()
    plt.yscale('log')
    plt.xscale('log')
    plt.xlabel('Rounds')
    plt.ylabel('Test loss')
    plt.title('Test loss vs training rounds for IID data')

plt.show()