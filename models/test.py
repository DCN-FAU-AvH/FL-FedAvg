import torch.nn.functional as F
from torch.utils.data import DataLoader


def test(model, args, dataset_test):
    """
    evaluate test accuracy
    """
    model.eval()
    test_loss = 0
    correct = 0
    data_loader = DataLoader(dataset_test, batch_size=len(dataset_test))
    for images, labels in data_loader:
        if args.gpu != -1:
            images, labels = images.to(args.device), labels.to(args.device)
        log_probs = model(images)
        test_loss += F.cross_entropy(log_probs, labels, reduction="sum").item()
        y_pred = log_probs.data.max(1, keepdim=True)[1]
        correct += y_pred.eq(labels.data.view_as(y_pred)).long().cpu().sum()
    test_loss /= len(data_loader.dataset)
    accuracy = correct / len(data_loader.dataset)
    return accuracy, test_loss
