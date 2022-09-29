import argparse
import os
import pickle
import pandas as pd
import torch
import torch.optim as optim
from thop import profile, clever_format
from torch.utils.data import DataLoader
from tqdm import tqdm
import torchvision.datasets as datasets
import torch.nn as nn
import utils3
from model_jaccard_2kernels import Model

torch.manual_seed(9916)

# train for one epoch to learn unique features
def train(net, data_loader, train_optimizer, temperature):
    net.train()
    total_loss, total_num, train_bar = 0.0, 0, tqdm(data_loader)
    for pos_1, pos_2, target in train_bar:
        pos_1, pos_2 = pos_1.cuda(non_blocking=True), pos_2.cuda(non_blocking=True)
        feature_1, out1_p, out1_n = net(pos_1)
        feature_2, out2_p, out2_n = net(pos_2)
        
        out_p = torch.cat([out1_p, out2_p], dim=0)
        out_n = torch.cat([out1_n, out2_n], dim=0)
        
        # [2*B, 2*B]
        sim_matrix_p = 1.0 + torch.mm(out_p, out_p.t().contiguous())
        sim_matrix_n = 2 - 2 * torch.mm(out_n, out_n.t().contiguous())
        sim_matrix_ns = 1.0 + torch.mm(out_n, out_n.t().contiguous())
        mask = (torch.ones_like(sim_matrix_p) - torch.eye(2 * batch_size, device=sim_matrix_p.device)).bool()
        
#         sim_matrix_n = torch.tensor([]).cuda()
#         for temp in out_n:
#             temp_dissim = torch.norm(temp.repeat(2*batch_size,1)-out_n, dim = -1).unsqueeze(0)
#             sim_matrix_n = torch.cat([sim_matrix_n,temp_dissim/12.0], dim = 0)
        # [2*B, 2*B-1]
        sim_matrix_p = sim_matrix_p.masked_select(mask).view(2 * batch_size, -1)
        sim_matrix_n = sim_matrix_n.masked_select(mask).view(2 * batch_size, -1)
        sim_matrix_ns = sim_matrix_ns.masked_select(mask).view(2 * batch_size, -1)
        sim_matrix = torch.exp(sim_matrix_p/(sim_matrix_p+sim_matrix_n)/temperature)
        sim_matrix_p = torch.exp(sim_matrix_p/temperature)
        sim_matrix_ns = torch.exp(sim_matrix_ns/temperature2)

        # compute similarity between positive keys
        pos_sim = 1.0 + torch.sum(out1_p * out2_p, dim=-1)
        pos_nsim = 1.0 + torch.sum(out1_n * out2_n, dim=-1)
        pos_dissim = 2 - 2 * torch.sum(out1_n * out2_n, dim=-1)
        
        # exponential
        pos = torch.exp(pos_sim/(pos_sim+pos_dissim)/temperature)
        pos_sim = torch.exp(pos_sim/temperature)
        pos_nsim = torch.exp(pos_nsim/temperature2)
        # [2*B]
        pos = torch.cat([pos, pos], dim=0)
        pos_sim = torch.cat([pos_sim, pos_sim], dim=0)
        pos_nsim = torch.cat([pos_nsim, pos_nsim], dim=0)
        
        #compute loss
        loss = 0.8*(- torch.log(pos / sim_matrix.sum(dim=-1))).mean() + 0.1*(- torch.log(pos_sim / sim_matrix_p.sum(dim=-1))).mean() + 0.1*(- torch.log(pos_nsim / sim_matrix_ns.sum(dim=-1))).mean()

        # back-propagation
        train_optimizer.zero_grad()
        loss.backward()
        train_optimizer.step()

        total_num += batch_size
        total_loss += loss.item() * batch_size
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
        # [N]
#         feature_labels = torch.tensor(memory_data_loader.dataset.targets, device=feature_bank.device)
        feature_labels = torch.tensor(targets,dtype = torch.int64).to(device=feature_bank.device)
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
    parser = argparse.ArgumentParser(description='Train SimCLR')
    parser.add_argument('--feature_dim', default=128, type=int, help='Feature dim for latent vector')
    parser.add_argument('--temperature', default=0.1, type=float, help='Temperature used in softmax')
    parser.add_argument('--k', default=200, type=int, help='Top k most similar images used to predict the label')
    parser.add_argument('--batch_size', default=256, type=int, help='Number of images in each mini-batch')
    parser.add_argument('--epochs', default=300, type=int, help='Number of sweeps over the dataset to train')

    # args parse
    args = parser.parse_args()
    feature_dim, temperature, k = args.feature_dim, args.temperature, args.k
    batch_size, epochs = args.batch_size, args.epochs
    temperature2 = 0.2

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
    model = nn.DataParallel(Model(feature_dim)).cuda()

#     flops, params = profile(model, inputs=(torch.randn(1, 3, 32, 32).cuda(),))
#     flops, params = clever_format([flops, params])
#     print('# Model Params: {} FLOPs: {}'.format(params, flops))
#     optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-6)
    optimizer = optim.Adam(
        [
                          {"params": model.module.f.parameters(),       "lr": 1e-3, "weight_decay": 1e-6},
                          {"params": model.module.gp.parameters(), "lr": 1e-3, "weight_decay": 1e-6},
                          {"params": model.module.gn.parameters(), "lr": 1e-3, "weight_decay": 1e-6},
        ],
        lr = 1e-4
    )
    c = 200
#     c = len(memory_data.classes)

    best_acc = 34.91
    broken_point = 132
    # training loop
    results = {'train_loss': [], 'test_acc@1': [], 'test_acc@5': [], 'temperature':[]}
    df = pd.read_csv('results_jaccard/128_0.1_200_256_300_statistics_TINY_jaccard_2kernels_2lr1e3_2tau0102.csv')
    results['train_loss'] = df['train_loss'].to_list()[0:broken_point-1]
    results['test_acc@1'] = df['test_acc@1'].to_list()[0:broken_point-1]
    results['test_acc@5'] = df['test_acc@5'].to_list()[0:broken_point-1]
    results['temperature'] = df['temperature'].to_list()[0:broken_point-1]
    save_name_pre = '{}_{}_{}_{}_{}'.format(feature_dim, temperature, k, batch_size, epochs)
    if not os.path.exists('results_jaccard'):
        os.mkdir('results_jaccard')
    
    
    
    for epoch in range(broken_point, epochs+1):
        results['temperature'].append(temperature)
        train_loss = train(model, train_loader, optimizer, temperature)
        results['train_loss'].append(train_loss)
        test_acc_1, test_acc_5 = test(model, memory_loader, test_loader)
        results['test_acc@1'].append(test_acc_1)
        results['test_acc@5'].append(test_acc_5)
        
        '''
        grad_f = []
        grad_gp = []
        grad_gn = []
        # record gradients
        for para in model.f.parameters():
            grad_f.append(torch.norm(para.grad.view(-1)))
        for para in model.gp.parameters():
            grad_gp.append(torch.norm(para.grad.view(-1)))
        for para in model.gn.parameters():
            grad_gn.append(torch.norm(para.grad.view(-1)))
        grads = [grad_f,grad_gp,grad_gn]
#         print(grads)
        '''
        
        # save statistics
        
        
        data_frame = pd.DataFrame(data=results, index=range(1, epoch + 1))
        data_frame.to_csv('results_jaccard/{}_statistics_TINY_jaccard_2kernels_2lr1e3_2tau0102.csv'.format(save_name_pre), index_label='epoch')
        
#         temp_feature, temp_hidden_p, temp_hidden_n = model(samples)
#         temp = [temp_feature,temp_hidden_p,temp_hidden_n,grads]
#         file = open("results_feature/t_0.1_samples_feature_SSL_2kernels_2lr1e4_samples.txt", "ab")   #Pickling
#         pickle.dump(temp, file)
#         file.close()
        
        if test_acc_1 > best_acc:
            print('test_acc_1:',test_acc_1,'best_acc:',best_acc)
            best_acc = test_acc_1
            torch.save(model.module.state_dict(), 'results_jaccard/{}_model_TINY_jaccard_2kernels_2lr1e3_2tau0102.pth'.format(save_name_pre))
