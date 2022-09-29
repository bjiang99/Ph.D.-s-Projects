import argparse
import os
import pandas as pd
import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
from tqdm import tqdm
import numpy as np
import utils3
from model_jaccard import Model


# train for one epoch to learn unique features
def train(encoder_q, encoder_k, data_loader, train_optimizer,temperature):
    global memory_queue_p, memory_queue_n
    encoder_q.train()
    total_loss, total_num, train_bar = 0.0, 0, tqdm(data_loader)
    for x_q, x_k, _ in train_bar:
        x_q, x_k = x_q.cuda(non_blocking=True), x_k.cuda(non_blocking=True)
        _, query_p, query_n = encoder_q(x_q)

        # shuffle BN
        idx = torch.randperm(x_k.size(0), device=x_k.device)
        _, key_p, key_n = encoder_k(x_k[idx])
        key_p = key_p[torch.argsort(idx)]
        key_n = key_n[torch.argsort(idx)]

        score_pos_p = torch.bmm(query_p.unsqueeze(dim=1), key_p.unsqueeze(dim=-1)).squeeze(dim=-1)
        score_neg_p = torch.mm(query_p, memory_queue_p.t().contiguous())
        score_pos_n = torch.bmm(query_n.unsqueeze(dim=1), key_n.unsqueeze(dim=-1)).squeeze(dim=-1)
        score_neg_n = torch.mm(query_n, memory_queue_n.t().contiguous())
        score_pos_s = score_pos_p + 1.0
        score_neg_s = score_neg_p + 1.0
        score_pos_ds = 2.0 - 2.0*score_pos_n
        score_neg_ds = 2.0 - 2.0*score_neg_n
        score_pos_j = score_pos_s/(score_pos_s+score_pos_ds)
        score_neg_j = score_neg_s/(score_neg_s+score_neg_ds)
        # [B, 1+M]
        out_p = torch.cat([score_pos_p, score_neg_p], dim=-1)
        out_n = torch.cat([score_pos_n, score_neg_n], dim=-1)
        out_j = torch.cat([score_pos_j, score_neg_j], dim=-1)
        # compute loss
        loss = 0.1*F.cross_entropy(out_p / temperature, torch.zeros(x_q.size(0), dtype=torch.long, device=x_q.device)) + 0.1*F.cross_entropy(out_n / temperature, torch.zeros(x_q.size(0), dtype=torch.long, device=x_q.device)) + 0.8*F.cross_entropy(out_j / temperature, torch.zeros(x_q.size(0), dtype=torch.long, device=x_q.device))

        train_optimizer.zero_grad()
        loss.backward()
        train_optimizer.step()

        # momentum update
        for parameter_q, parameter_k in zip(encoder_q.parameters(), encoder_k.parameters()):
            parameter_k.data.copy_(parameter_k.data * momentum + parameter_q.data * (1.0 - momentum))
        # update queue
        memory_queue_p = torch.cat((memory_queue_p, key_p), dim=0)[key_p.size(0):]
        memory_queue_n = torch.cat((memory_queue_n, key_n), dim=0)[key_n.size(0):]

        total_num += x_q.size(0)
        total_loss += loss.item() * x_q.size(0)
        train_bar.set_description('Train Epoch: [{}/{}] Loss: {:.4f}'.format(epoch, epochs, total_loss / total_num))

    return total_loss / total_num


# test for one epoch, use weighted knn to find the most similar images' label to assign the test image
def test(net, memory_data_loader, test_data_loader):
    net.eval()
    total_top1, total_top5, total_num, feature_bank = 0.0, 0.0, 0, []
    with torch.no_grad():
        # generate feature bank
        targets = torch.tensor([])
        for data, _, target in tqdm(memory_data_loader, desc='Feature extracting'):
            feature, _, _ = net(data.cuda(non_blocking=True))
            feature_bank.append(feature)
            targets = torch.cat((targets,target))
        # [D, N]
        feature_bank = torch.cat(feature_bank, dim=0).t().contiguous()
        feature_labels = torch.tensor(targets,dtype = torch.int64).to(device=feature_bank.device)
        # [N]
#         feature_labels = torch.tensor(memory_data_loader.dataset.targets, device=feature_bank.device)
        # loop test data to predict the label by weighted knn search
        test_bar = tqdm(test_data_loader)
        for data, _, target in test_bar:
            data, target = data.cuda(non_blocking=True), target.cuda(non_blocking=True)
            feature, _, _ = net(data)

            total_num += data.size(0)
            # compute cos similarity between each feature vector and feature bank ---> [B, N]
            sim_matrix = torch.mm(feature, feature_bank)
            # [B, K]
            sim_weight, sim_indices = sim_matrix.topk(k=k, dim=-1)
            # [B, K]
            sim_labels = torch.gather(feature_labels.expand(data.size(0), -1), dim=-1, index=sim_indices)
            sim_weight = (sim_weight / temperature).exp()

            # counts for each class
            one_hot_label = torch.zeros(data.size(0) * k, c, device=sim_labels.device)
            # [B*K, C]
            one_hot_label = one_hot_label.scatter(dim=-1, index=sim_labels.view(-1, 1), value=1.0)
            # weighted score ---> [B, C]
            pred_scores = torch.sum(one_hot_label.view(data.size(0), -1, c) * sim_weight.unsqueeze(dim=-1), dim=1)

            pred_labels = pred_scores.argsort(dim=-1, descending=True)
            total_top1 += torch.sum((pred_labels[:, :1] == target.unsqueeze(dim=-1)).any(dim=-1).float()).item()
            total_top5 += torch.sum((pred_labels[:, :5] == target.unsqueeze(dim=-1)).any(dim=-1).float()).item()
            test_bar.set_description('Test Epoch: [{}/{}] Acc@1:{:.2f}% Acc@5:{:.2f}%'
                                     .format(epoch, epochs, total_top1 / total_num * 100, total_top5 / total_num * 100))

    return total_top1 / total_num * 100, total_top5 / total_num * 100


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train MoCo')
    parser.add_argument('--feature_dim', default=128, type=int, help='Feature dim for each image')
    parser.add_argument('--m', default=4096, type=int, help='Negative sample number')
    parser.add_argument('--temperature', default=0.06, type=float, help='Temperature used in softmax')
    parser.add_argument('--momentum', default=0.999, type=float, help='Momentum used for the update of memory bank')
    parser.add_argument('--k', default=200, type=int, help='Top k most similar images used to predict the label')
    parser.add_argument('--batch_size', default=256, type=int, help='Number of images in each mini-batch')
    parser.add_argument('--epochs', default=300, type=int, help='Number of sweeps over the dataset to train')

    # args parse
    args = parser.parse_args()
    feature_dim, m, temperature, momentum = args.feature_dim, args.m, args.temperature, args.momentum
    k, batch_size, epochs = args.k, args.batch_size, args.epochs

    # data prepare
    torch.set_num_threads(3)
    train_dir = 'tiny-imagenet-200/tiny-imagenet-200/train'
    TINY_TR = datasets.ImageFolder(train_dir)
    train_data = utils3.TINY200Pair(TINY_TR, transform = utils3.train_transform)
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=16, pin_memory=True,
                              drop_last=True)
    memory_data = utils3.TINY200Pair(TINY_TR, transform = utils3.test_transform)
    memory_loader = DataLoader(memory_data, batch_size=batch_size, shuffle=False, num_workers=16, pin_memory=True)
    test_dir = 'tiny-imagenet-200/tiny-imagenet-200/val/image'
    target_dir = 'tiny-imagenet-200/tiny-imagenet-200/val/annotation.txt'
    f = open(target_dir, "r")
    annotations = []
    for i in f.read().split(','):
        annotations.append(int(i))
    TINY_TS = datasets.ImageFolder(test_dir)
    test_data = utils3.TINY200Pair(TINY_TS, train = False, targets = annotations, transform = utils3.test_transform)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=16, pin_memory=True)

    # model setup and optimizer config
    os.environ["CUDA_VISIBLE_DEVICES"] = '5,6,7'
    model_q = nn.DataParallel(Model(feature_dim)).cuda()
    model_k = nn.DataParallel(Model(feature_dim)).cuda()
    # initialize
    for param_q, param_k in zip(model_q.parameters(), model_k.parameters()):
        param_k.data.copy_(param_q.data)
        # not update by gradient
        param_k.requires_grad = False
    optimizer = optim.Adam(model_q.parameters(), lr=1e-3, weight_decay=1e-6)

    # c as num of train class
#     c = len(memory_data.classes)
    c = 200
    # init memory queue as unit random vector ---> [M, D]
    memory_queue_p = F.normalize(torch.randn(m, feature_dim).cuda(), dim=-1)
    memory_queue_n = F.normalize(torch.randn(m, feature_dim).cuda(), dim=-1)

    # training loop
    results = {'train_loss': [], 'test_acc@1': [], 'test_acc@5': [],'temperature':[]}
    save_name_pre = '{}_{}_{}_{}_{}_{}_{}'.format(feature_dim, m, temperature, momentum, k, batch_size, epochs)

#     df = pd.read_csv('results_jaccard/128_4096_0.1_0.999_200_512_300_results_TINY_2kernels_2lr1e3.csv')
#     results['train_loss'] = df['train_loss'].to_list()
#     results['test_acc@1'] = df['test_acc@1'].to_list()
#     results['test_acc@5'] = df['test_acc@5'].to_list()
#     results['temperature'] = df['temperature'].to_list()
    best_acc = 0.0
    broken_point = 1

    for epoch in range(broken_point, epochs + 1):
        results['temperature'].append(temperature)
        train_loss = train(model_q, model_k, train_loader, optimizer, temperature)
        results['train_loss'].append(train_loss)
        test_acc_1, test_acc_5 = test(model_q, memory_loader, test_loader)
        results['test_acc@1'].append(test_acc_1)
        results['test_acc@5'].append(test_acc_5)
        # save statistics
        if not os.path.exists('results_jaccard'):
            os.mkdir('results_jaccard')
        data_frame = pd.DataFrame(data=results, index=range(1, epoch + 1))
        data_frame.to_csv('results_jaccard/{}_results_TINY_2kernels_2lr1e3.csv'.format(save_name_pre), index_label='epoch')
        if test_acc_1 > best_acc:
            best_acc = test_acc_1
            torch.save(model_q.module.state_dict(), 'results_jaccard/{}_model_TINY_2kernels_2lr1e3_q.pth'.format(save_name_pre))
            torch.save(model_k.module.state_dict(), 'results_jaccard/{}_model_TINY_2kernels_2lr1e3_k.pth'.format(save_name_pre))
