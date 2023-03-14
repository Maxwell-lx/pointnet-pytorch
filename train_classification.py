from __future__ import print_function
import argparse
import os
import random
import torch
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
from dataset import ShapeNetDataset
from model import PointNetCls, feature_transform_regularizer
import torch.nn.functional as F
from tqdm import tqdm
import utility as util

parser = argparse.ArgumentParser()
parser.add_argument('--batchSize', type=int, default=32, help='input batch size')
parser.add_argument('--workers', type=int, help='number of data loading workers', default=4)
parser.add_argument('--nepoch', type=int, default=2, help='number of epochs to train for')
parser.add_argument('--outf', type=str, default='cls', help='output folder')
parser.add_argument('--model', type=str, default='', help='model path')
parser.add_argument('--dataset', type=str, default='../shapenetcore_partanno_segmentation_benchmark_v0', help="dataset path")
parser.add_argument('--feature_transform', action='store_true', help="use feature transform")
opt = parser.parse_args()
print('\nopt:', opt)

blue = lambda x: '\033[94m' + x + '\033[0m'

opt.manualSeed = random.randint(1, 10000)  # fix seed for parallel training
print("Random Seed: ", opt.manualSeed)
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)

dataset = ShapeNetDataset(root=opt.dataset, dataset_type='classification', dataset_split='train')
test_dataset = ShapeNetDataset(root=opt.dataset, dataset_type='classification', dataset_split='test', data_augmentation=False)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=opt.batchSize, shuffle=True, num_workers=int(opt.workers))
testdataloader = torch.utils.data.DataLoader(test_dataset, batch_size=opt.batchSize, shuffle=True, num_workers=int(opt.workers))

num_classes = len(dataset.classes)
print('Total classes:', num_classes)

try:
    os.makedirs(opt.outf)
except OSError:
    pass

classifier = PointNetCls(k=num_classes, feature_transform=opt.feature_transform)
optimizer = optim.Adam(classifier.parameters(), lr=0.001, betas=(0.9, 0.999))
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
classifier.cuda()

num_batch = len(dataset) / opt.batchSize

for epoch in range(opt.nepoch):
    for i, data in enumerate(dataloader, 0):
        points, target = data
        target = target[:, 0]
        points = points.transpose(2, 1)
        points, target = points.cuda(), target.cuda()
        optimizer.zero_grad()
        classifier = classifier.train()
        pred, trans, trans_feat = classifier(points)
        loss = F.nll_loss(pred, target)
        if opt.feature_transform:
            loss += feature_transform_regularizer(trans_feat) * 0.001
        loss.backward()
        optimizer.step()
        pred_choice = pred.data.max(1)[1]
        correct = pred_choice.eq(target.data).cpu().sum()
        print('[%d: %d/%d] train loss: %f accuracy: %f' % (epoch, i, num_batch, loss.item(), correct.item() / float(opt.batchSize)))

        if i % 10 == 0:
            j, data = next(enumerate(testdataloader, 0))
            points, target = data
            target = target[:, 0]
            points = points.transpose(2, 1)
            points, target = points.cuda(), target.cuda()
            classifier = classifier.eval()
            pred, _, _ = classifier(points)
            loss = F.nll_loss(pred, target)
            pred_choice = pred.data.max(1)[1]
            correct = pred_choice.eq(target.data).cpu().sum()
            print('[%d: %d/%d] %s loss: %f accuracy: %f' % (epoch, i, num_batch, util.blue('test'), loss.item(), correct.item() / float(opt.batchSize)))

    scheduler.step()
    util.save_model('%s/cls_model_%d.pth' % (opt.outf, epoch), optimizer, classifier)

# benchmark
valid_dataset = ShapeNetDataset(root=opt.dataset, dataset_type='classification', class_choice=None, dataset_split='val', data_augmentation=False)
valid_dataloader = torch.utils.data.DataLoader(valid_dataset, batch_size=opt.batchSize, shuffle=True, num_workers=int(opt.workers))

total_correct = 0
total_validset = 0
for i, data in tqdm(enumerate(valid_dataloader, 0)):
    points, target = data
    target = target[:, 0]
    points = points.transpose(2, 1)
    points, target = points.cuda(), target.cuda()
    classifier = classifier.eval()
    pred, _, _ = classifier(points)
    pred_choice = pred.data.max(1)[1]
    correct = pred_choice.eq(target.data).cpu().sum()
    total_correct += correct.item()
    total_validset += points.size()[0]

print("final accuracy {}".format(total_correct / float(total_validset)))
