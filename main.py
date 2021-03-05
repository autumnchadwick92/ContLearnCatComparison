import argparse

import os

import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.tensorboard

from torch.utils.tensorboard import SummaryWriter

import logging

from torch.autograd import Variable
from tqdm import tqdm

from src.data.PermutedMNIST import get_permuted_MNIST
from src.data.PermutedCIFAR import get_permuted_CIFAR10
from src.data.PermutatedCORE50 import get_permuted_CORE50

#model imports
from src.model.ProgressiveNeuralNetworks import PNN
from src.model.AR1wReplay import AR1wReplay

from src.tools.arg_parser_actions import LengthCheckAction
from src.tools.evaluation import evaluate_model
from torch.utils.data import DataLoader


writer = SummaryWriter()

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


def get_args():
    parser = argparse.ArgumentParser(description='Progressive Neural Networks')
    parser.add_argument('-path', default='~/Documents/ThesisWork/ProgressiveNeuralNetworks.pytorch-master/src/data', type=str, help='path to the data')
    parser.add_argument('-cuda', default=-1, type=int, help='Cuda device to use (-1 for none)')

    parser.add_argument('--layers', metavar='L', type=int, default=3, help='Number of layers per task')
    parser.add_argument('--sizes', dest='sizes', default=[784, 1024, 512, 10], nargs='+',
                        action=LengthCheckAction)

    parser.add_argument('--n_tasks', dest='n_tasks', type=int, default=5)
    parser.add_argument('--epochs', dest='epochs', type=int, default=10)
    parser.add_argument('--bs', dest='batch_size', type=int, default=128)
    parser.add_argument('--lr', dest='lr', type=float, default=1e-3, help='Optimizer learning rate')
    parser.add_argument('--wd', dest='wd', type=float, default=1e-4, help='Optimizer weight decay')
    parser.add_argument('--momentum', dest='momentum', type=float, default=1e-4, help='Optimizer momentum')

    parser.add_argument('--dataset', default='CORE50', type=str, help='MNIST , CIFAR10, CORE50')
    parser.add_argument('--model', default='AR1', type=str, help='PNN, AR1')

    args = parser.parse_known_args()
    return args[0]


def main(args):
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args['cuda'])

    if(args['model'] == 'PNN'):
        model = PNN(args['layers'])

    if(args['dataset'] == 'MNIST'):
        tasks_data = get_permuted_MNIST(args['path'], args['batch_size'],True)
        test_data = get_permuted_MNIST(args['path'], args['batch_size'],False)

    elif(args['dataset'] == 'CIFAR10'):
        tasks_data = get_permuted_CIFAR10(args['path'], args['batch_size'],True)
        test_data = get_permuted_CIFAR10(args['path'], args['batch_size'],False)


    elif(args['dataset'] == 'CORE50'):
        tasks_data = get_permuted_CORE50(args['path'], args['batch_size'],True)
        test_data = get_permuted_CORE50(args['path'], args['batch_size'],False)




    x = torch.Tensor()
    y = torch.LongTensor()

    if (args['model'] == 'AR1'):
        AR1wReplay(args['dataset'] ,tasks_data,test_data,batch_size=args['batch_size'])
    else:
        if args['cuda'] != -1:
            logger.info('Running with cuda (GPU n°{})'.format(args['cuda']))
            model.cuda()
            x = x.cuda()
            y = y.cuda()
        else:
            logger.warning('Running WITHOUT cuda')

        for task_id, (train_set) in enumerate(tasks_data):
            train_loader = DataLoader(train_set, batch_size=args['batch_size'], shuffle=True)

            model.freeze_columns()
            if(args['dataset'] == 'MNIST'):
                model.new_task(args['sizes'])
            elif(args['dataset'] == 'CIFAR10'):
                model.new_task([3072, 1024, 128, 10])
            elif (args['dataset'] == 'CORE50'):
                model.new_task([49152, 16384, 128, 10])


            optimizer = torch.optim.RMSprop(model.parameters(task_id), lr=args['lr'],
                                            weight_decay=args['wd'], momentum=args['momentum'])

            train_accs = []
            train_losses = []
            for epoch in range(args['epochs']):
                total_samples = 0
                total_loss = 0
                correct_samples = 0

                for inputs, labels, task in tqdm(train_loader):

                    print("Input Size")
                    print(inputs.size())

                    x.resize_(inputs.size()).copy_(inputs)
                    y.resize_(labels.size()).copy_(labels)

                    x = x.view(x.size(0), -1)
                    predictions = model(Variable(x))

                    _, predicted = torch.max(predictions.data, 1)
                    total_samples += y.size(0)
                    correct_samples += (predicted == y).sum()

                    indiv_loss = F.cross_entropy(predictions, Variable(y))
                    total_loss += indiv_loss.item()

                    optimizer.zero_grad()
                    indiv_loss.backward()
                    optimizer.step()

                train_accs.append(correct_samples / total_samples)
                train_losses.append(total_loss / total_samples)
                logger.info(
                    '[T{}][{}/{}] Loss={}, Acc= {}'.format(task_id, epoch, args['epochs'], train_losses[-1],
                                                           train_accs[-1]))
                writer.add_scalar("Loss/train", train_losses[-1], epoch)
                writer.add_scalar("Acc/train", train_accs[-1], epoch)
                writer.flush()

            logger.info('Evaluation after task {}:'.format(task_id))

            for task_id, (test_set) in enumerate(test_data):
                test_loader = DataLoader(test_set, batch_size=args['batch_size'], shuffle=False)
                test_perf = evaluate_model(model, test_loader)
                logger.info('\t test:{}%'.format(test_perf.item()))



if __name__ == '__main__':
    main(vars(get_args()))
