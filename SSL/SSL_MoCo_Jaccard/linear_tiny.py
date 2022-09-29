import argparse
import os

import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from thop import profile, clever_format
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
from torchvision.datasets import CIFAR10, CIFAR100
from tqdm import tqdm

import utils,utils3
from model import Model

torch.manual_seed(9916)

class Net(nn.Module):
    def __init__(self, num_class, pretrained_path):
        super(Net, self).__init__()

        # encoder
        self.f = Model().f
        # classifier
        self.fc = nn.Linear(2048, num_class, bias=True)
        self.load_state_dict(torch.load(pretrained_path, map_location='cpu'), strict=False)

    def forward(self, x):
        x = self.f(x)
        feature = torch.flatten(x, start_dim=1)
        out = self.fc(feature)
        return out


# train or test for one epoch
def train_val(net, data_loader, train_optimizer):
    is_train = train_optimizer is not None
    net.train() if is_train else net.eval()

    total_loss, total_correct_1, total_correct_5, total_num, data_bar = 0.0, 0.0, 0.0, 0, tqdm(data_loader)
    with (torch.enable_grad() if is_train else torch.no_grad()):
        for data, target in data_bar:
            data, target = data.cuda(non_blocking=True), target.cuda(non_blocking=True)
            out = net(data)
            loss = loss_criterion(out, target)

            if is_train:
                train_optimizer.zero_grad()
                loss.backward()
                train_optimizer.step()

            total_num += data.size(0)
            total_loss += loss.item() * data.size(0)
            prediction = torch.argsort(out, dim=-1, descending=True)
            total_correct_1 += torch.sum((prediction[:, 0:1] == target.unsqueeze(dim=-1)).any(dim=-1).float()).item()
            total_correct_5 += torch.sum((prediction[:, 0:5] == target.unsqueeze(dim=-1)).any(dim=-1).float()).item()

            data_bar.set_description('{} Epoch: [{}/{}] Loss: {:.4f} ACC@1: {:.2f}% ACC@5: {:.2f}%'
                                     .format('Train' if is_train else 'Test', epoch, epochs, total_loss / total_num,
                                             total_correct_1 / total_num * 100, total_correct_5 / total_num * 100))

    return total_loss / total_num, total_correct_1 / total_num * 100, total_correct_5 / total_num * 100


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Linear Evaluation')
    parser.add_argument('--model_path', type=str, default='results_jaccard/128_4096_0.08_0.999_200_256_300_model_TINY_2kernels_2lr1e3_q.pth',
#                         help='The pretrained model path')
#     parser.add_argument('--model_path', type=str, default='results/128_0.1_200_512_500_model_cifa100.pth',
                        help='The pretrained model path')
    
    parser.add_argument('--batch_size', type=int, default=256, help='Number of images in each mini-batch')
    parser.add_argument('--epochs', type=int, default=100, help='Number of sweeps over the dataset to train')

    os.environ["CUDA_VISIBLE_DEVICES"] = '5,6,7'
    args = parser.parse_args()
#     folder = 'results_tau/'
#     model = '128_0.5_200_512_500_model.pth'
    
    model_path, batch_size, epochs = args.model_path, args.batch_size, args.epochs
    torch.set_num_threads(3)
    train_dir = 'tiny-imagenet-200/tiny-imagenet-200/train'
    TINY_TR = datasets.ImageFolder(train_dir)
    train_data = utils3.TINY200(TINY_TR, transform = utils3.train_transform)
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=16, pin_memory=True,
                              drop_last=True)
    test_dir = 'tiny-imagenet-200/tiny-imagenet-200/val/image'
    target_dir = 'tiny-imagenet-200/tiny-imagenet-200/val/annotation.txt'
    f = open(target_dir, "r")
    annotations = []
    for i in f.read().split(','):
        annotations.append(int(i))
    TINY_TS = datasets.ImageFolder(test_dir)
    test_data = utils3.TINY200(TINY_TS, train = False, targets = annotations, transform = utils3.test_transform)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=16, pin_memory=True)

    model = Net(num_class=200, pretrained_path=model_path)
    model = nn.DataParallel(model).cuda()
    for param in model.module.f.parameters():
        param.requires_grad = False

#     flops, params = profile(model, inputs=(torch.randn(1, 3, 32, 32).cuda(),))
#     flops, params = clever_format([flops, params])
#     print('# Model Params: {} FLOPs: {}'.format(params, flops))
    optimizer = optim.Adam(model.module.fc.parameters(), lr=1e-3, weight_decay=1e-6)
    loss_criterion = nn.CrossEntropyLoss()
    
    broken_point = 1
    best_acc = 0.0
    results = {'train_loss': [], 'train_acc@1': [], 'train_acc@5': [],
               'test_loss': [], 'test_acc@1': [], 'test_acc@5': []}

#     df = pd.read_csv('results_jaccard/linear_128_0.1_200_256_300_statistics_TINY_conventional_SSL.csv')
#     results['train_loss'] = df['train_loss'].to_list()[0:broken_point-1]
#     results['train_acc@1'] = df['test_acc@1'].to_list()[0:broken_point-1]
#     results['train_acc@5'] = df['test_acc@5'].to_list()[0:broken_point-1]
#     results['test_loss'] = df['train_loss'].to_list()[0:broken_point-1]
#     results['test_acc@1'] = df['test_acc@1'].to_list()[0:broken_point-1]
#     results['test_acc@5'] = df['test_acc@5'].to_list()[0:broken_point-1]
    
    
    
    for epoch in range(broken_point, epochs + 1):
        train_loss, train_acc_1, train_acc_5 = train_val(model, train_loader, optimizer)
        results['train_loss'].append(train_loss)
        results['train_acc@1'].append(train_acc_1)
        results['train_acc@5'].append(train_acc_5)
        test_loss, test_acc_1, test_acc_5 = train_val(model, test_loader, None)
        results['test_loss'].append(test_loss)
        results['test_acc@1'].append(test_acc_1)
        results['test_acc@5'].append(test_acc_5)
        # save statistics
        data_frame = pd.DataFrame(data=results, index=range(1, epoch + 1))
        data_frame.to_csv('results_jaccard/linear_128_4096_0.08_0.999_200_256_300_statistics_TINY_2kernels_2lr1e3.csv', index_label='epoch')
#         data_frame.to_csv('results_jaccard/linear_128_0.1_200_512_500_statistics_cifa100.pth', index_label='epoch')
        if test_acc_1 > best_acc:
            best_acc = test_acc_1
#             torch.save(model.state_dict(), 'results_jaccard/linear_128_0.1_200_512_500_model_cifa100.pth')
            torch.save(model.module.state_dict(), 'results_jaccard/linear_128_4096_0.08_0.999_200_256_300_model_TINY_2kernels_2lr1e3.pth')
