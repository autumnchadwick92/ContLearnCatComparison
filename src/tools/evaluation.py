import torch
from torch.autograd import Variable
from tqdm import tqdm


def evaluate_model(model, dataset_loader):
    total = 0
    correct = 0
    x = torch.Tensor()
    y = torch.LongTensor()
    for inputs, labels, task in tqdm(dataset_loader):
        x.resize_(inputs.size()).copy_(inputs)
        y.resize_(labels.size()).copy_(labels)

        x = x.view(x.size(0), -1)
        preds = model(Variable(x))

        _, predicted = torch.max(preds.data, 1)

        total += labels.size(0)
        correct += (predicted == y).sum()

    return 100 * correct / total