from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import os
from models.LeNet import  LeNet

'''
Attenzione, nel paper Dynamic Netowrk Surgery parlano di usare LeNet-5, utilizzando il prototxt di caffe. In realtà il protoxt di caffe descrive la rete qui sotto, che è un pochino diversa
Da quella descritta nel paper di Lecun. Un'implementazione in pytorch di quella rete è qui : https://github.com/bollakarthikeya/LeNet-5-PyTorch/blob/master/lenet5_gpu.py
Tra l'altro il numero di parametri (431K) riportato nel paper pe LeNet corrisponde a quello di questa rete, quindi direi che possiamo utililzzare questa


Caffe learning policy:
- inv: return base_lr * (1 + gamma * iter) ^ (- power)

'''


def adjust_learning_rate(optimizer, iter, args):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = args.lr * (1+args.gamma_lr *iter)**(-args.power_lr)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

#Classic LeNet as in caffe protoxt https://github.com/BVLC/caffe/blob/master/examples/mnist/lenet.prototxt


def train(args, model, device, train_loader, optimizer, epoch):
    model.train()

    for batch_idx, (data, target) in enumerate(train_loader):
        adjust_learning_rate(optimizer, batch_idx, args)

        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.item()))


def test(args, model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.6f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    return correct / len(test_loader.dataset)



def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=20, metavar='N',
                        help='number of epochs to train (default: 20)')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                        metavar='W', help='weight decay (default: 1e-4)',
                        dest='weight_decay')
    parser.add_argument("--gamma-lr", type=float, default=0.001,
                        help="lr decay:  (lr= lr*(1+gamma*iter)^-power)")
    parser.add_argument("--power-lr", type=float, default=1.0,
                        help="lr decay (lr=lr* (1+gamma*iter)^-power)")
    parser.add_argument("--save-dir", type=str,
                        help="Directory where to save pretrined models")


    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")

    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=args.batch_size, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=False, transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])),
        batch_size=args.test_batch_size, shuffle=True, **kwargs)

    model = LeNet().to(device)
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    acc = 0
    best_acc = -1
    best_epoch = -1
    for epoch in range(1, args.epochs + 1):
        train(args, model, device, train_loader, optimizer, epoch)
        acc = test(args, model, device, test_loader)
        if acc > best_acc:
            best_acc = acc
            best_epoch = epoch
            save_name = os.path.join(args.save_dir,"best_model.pth" )
            torch.save(model.state_dict(), save_name)

    print("Best epoch {} with accuracy {:.3f}". format(best_epoch, best_acc))


if __name__ == '__main__':
    main()
