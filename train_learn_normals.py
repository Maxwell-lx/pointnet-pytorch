from __future__ import print_function
import argparse
import os
import random
import torch
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
from dataset import ShapeNetDataset
import model
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np
from torch import nn
import utility as util

parser = argparse.ArgumentParser()
parser.add_argument('--batchSize', type=int, default=32, help='input batch size')
parser.add_argument('--workers', type=int, help='number of data loading workers', default=4)
parser.add_argument('--nepoch', type=int, default=100, help='number of epochs to train for')
parser.add_argument('--outf', type=str, default='learn_normals', help='output folder')
parser.add_argument('--model', type=str, default='', help='model path')
parser.add_argument('--dataset', type=str, default='../shapenetcore_partanno_segmentation_benchmark_v0', help="dataset path")
parser.add_argument('--class_choice', type=str, default=None, help="class_choice")
parser.add_argument('--feature_transform', default=False, help="use feature transform")

opt = parser.parse_args()
print('opt:', opt)

opt.manualSeed = random.randint(1, 10000)  # fix seed
print("Random Seed: ", opt.manualSeed)
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)

dataset = ShapeNetDataset(root=opt.dataset, dataset_type='learn_normals', class_choice=['Chair'], data_augmentation=False)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=opt.batchSize, shuffle=True, num_workers=int(opt.workers))

test_dataset = ShapeNetDataset(root=opt.dataset, dataset_type='learn_normals', class_choice=['Chair'], dataset_split='test', data_augmentation=False)
testdataloader = torch.utils.data.DataLoader(test_dataset, batch_size=opt.batchSize, shuffle=True, num_workers=int(opt.workers))

try:
    os.makedirs(opt.outf)
except OSError:
    pass

classifier = model.PointNetLearn_normals(k=3, feature_transform=opt.feature_transform)
optimizer = optim.Adam(classifier.parameters(), lr=0.01, betas=(0.9, 0.999))
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
classifier.cuda()

num_batch = len(dataset) / opt.batchSize

for epoch in range(opt.nepoch):
    for i, data in enumerate(dataloader, 0):
        points, target = data
        points = points.transpose(2, 1)
        points, target = points.cuda(), target.cuda()
        optimizer.zero_grad()
        classifier = classifier.train()
        pred, trans, trans_feat = classifier(points)

        loss = util.learn_normals_loss(pred,target)
        loss_trans = model.feature_transform_regularizer(trans)*0.1
        loss += loss_trans
        if opt.feature_transform:
            loss += model.feature_transform_regularizer(trans_feat) * 0.001
        loss.backward()
        optimizer.step()
        print('[%d: %d/%d] train loss: %f trans loss: %f' % (epoch, i, num_batch, loss.item(),loss_trans))

        if i % 100 == 0:
            j, data = next(enumerate(testdataloader, 0))
            points, target = data
            points = points.transpose(2, 1)
            points, target = points.cuda(), target.cuda()
            classifier = classifier.eval()
            pred, trans, trans_feat = classifier(points)

            loss = util.learn_normals_loss(pred, target)
            loss_trans = model.feature_transform_regularizer(trans) * 0.1
            loss += loss_trans
            if opt.feature_transform:
                loss += model.feature_transform_regularizer(trans_feat) * 0.001

            print('[%d: %d/%d] %s loss: %f' % (epoch, i, num_batch, util.blue('test'), loss.item()))

    scheduler.step()
    util.save_model('%s/learn_normals_model_%s_%d.pth' % (opt.outf, opt.class_choice, epoch), optimizer, classifier)


# benchmark
valid_dataset = ShapeNetDataset(root=opt.dataset, dataset_type='learn_normals', class_choice=['Chair'], dataset_split='val', data_augmentation=False)
valid_dataloader = torch.utils.data.DataLoader(valid_dataset, batch_size=1, shuffle=True, num_workers=int(opt.workers))

total_loss = []

for _, data in tqdm(enumerate(valid_dataloader, 0)):
    points, target = data
    points = points.transpose(2, 1)
    points, target = points.cuda(), target.cuda()
    classifier = classifier.eval()
    pred, trans, trans_feat = classifier(points)

    loss = util.learn_normals_loss(pred, target)
    loss_trans = model.feature_transform_regularizer(trans) * 0.1
    loss += loss_trans
    if opt.feature_transform:
        loss += model.feature_transform_regularizer(trans_feat) * 0.001
    total_loss.append(loss.item())



print("learn normals total loss for class {}: {}".format(opt.class_choice, np.mean(total_loss)))
