from __future__ import print_function
import argparse
import os
import random
import torch
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
from dataset import ShapeNetDataset
from model import PointNetDenseCls, feature_transform_regularizer
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np
import utility as util

parser = argparse.ArgumentParser()
parser.add_argument('--batchSize', type=int, default=32, help='input batch size')
parser.add_argument('--workers', type=int, default=4, help='number of data loading workers')
parser.add_argument('--nepoch', type=int, default=1, help='number of epochs to train for')
parser.add_argument('--outf', type=str, default='seg', help='output folder')
parser.add_argument('--model', type=str, default='', help='model path')
parser.add_argument('--dataset', type=str, default='../shapenetcore_partanno_segmentation_benchmark_v0', help="dataset path")
parser.add_argument('--class_choice', type=str, default='Chair', help="class_choice")
parser.add_argument('--feature_transform', action='store_true', help="use feature transform")
opt = parser.parse_args()
print('\nopt:', opt)

opt.manualSeed = random.randint(1, 10000)  # fix seed for parallel training
print("Random Seed: ", opt.manualSeed)
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)

dataset = ShapeNetDataset(root=opt.dataset, dataset_type='segmentation', class_choice=[opt.class_choice], dataset_split='train', data_augmentation=True)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=opt.batchSize, shuffle=True, num_workers=int(opt.workers))

test_dataset = ShapeNetDataset(root=opt.dataset, dataset_type='segmentation', class_choice=[opt.class_choice], dataset_split='test', data_augmentation=False)
testdataloader = torch.utils.data.DataLoader(test_dataset, batch_size=opt.batchSize, shuffle=True, num_workers=int(opt.workers))

num_segs = dataset.num_seg_classes
print('num_segs:', num_segs)
try:
    os.makedirs(opt.outf)
except OSError:
    pass


classifier = PointNetDenseCls(k=num_segs, feature_transform=opt.feature_transform)

if opt.model != '':
    classifier.load_state_dict(torch.load(opt.model))

optimizer = optim.Adam(classifier.parameters(), lr=0.001, betas=(0.9, 0.999))
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
        pred = pred.view(-1, num_segs)
        target = target.view(-1, 1)[:, 0] - 1
        loss = F.nll_loss(pred, target)
        if opt.feature_transform:
            loss += feature_transform_regularizer(trans_feat) * 0.001
        loss.backward()
        optimizer.step()
        pred_choice = pred.data.max(1)[1]
        correct = pred_choice.eq(target.data).cpu().sum()
        print('[%d: %d/%d] train loss: %f accuracy: %f' % (epoch, i, num_batch, loss.item(), correct.item() / float(opt.batchSize * 2500)))

        if i % 10 == 0:
            j, data = next(enumerate(testdataloader, 0))
            points, target = data
            points = points.transpose(2, 1)
            points, target = points.cuda(), target.cuda()
            classifier = classifier.eval()
            pred, _, _ = classifier(points)
            pred = pred.view(-1, num_segs)
            target = target.view(-1, 1)[:, 0] - 1
            loss = F.nll_loss(pred, target)
            pred_choice = pred.data.max(1)[1]
            correct = pred_choice.eq(target.data).cpu().sum()
            print('[%d: %d/%d] %s loss: %f accuracy: %f' % (epoch, i, num_batch, util.blue('test'), loss.item(), correct.item() / float(opt.batchSize * 2500)))

    scheduler.step()
    # torch.save(classifier.state_dict(), '%s/seg_model_%s_%d.pth' % (opt.outf, opt.class_choice, epoch))
    util.save_model('%s/seg_model_%s_%d.pth' % (opt.outf, opt.class_choice, epoch), optimizer, classifier)

# benchmark
# classifier = PointNetDenseCls(k=num_segs, feature_transform=opt.feature_transform)
# optimizer = optim.Adam(classifier.parameters(), lr=0.001, betas=(0.9, 0.999))
# util.load_model('./seg/seg_model_Chair_24', optimizer, classifier)

valid_dataset = ShapeNetDataset(root=opt.dataset, dataset_type='segmentation', class_choice=[opt.class_choice], dataset_split='val', data_augmentation=False)
valid_dataloader = torch.utils.data.DataLoader(valid_dataset, batch_size=opt.batchSize, shuffle=True, num_workers=int(opt.workers))

shape_ious = []
for i, data in tqdm(enumerate(valid_dataloader, 0)):
    points, target = data
    points = points.transpose(2, 1)
    points, target = points.cuda(), target.cuda()
    classifier = classifier.eval()
    pred, _, _ = classifier(points)
    pred_choice = pred.data.max(2)[1]
    pred_np = pred_choice.cpu().data.numpy()
    target_np = target.cpu().data.numpy() - 1

    for shape_idx in range(target_np.shape[0]):
        parts = range(num_segs)  # np.unique(target_np[shape_idx])
        part_ious = []
        for part in parts:
            I = np.sum(np.logical_and(pred_np[shape_idx] == part, target_np[shape_idx] == part))
            U = np.sum(np.logical_or(pred_np[shape_idx] == part, target_np[shape_idx] == part))
            if U == 0:
                iou = 1  # If the union of groundtruth and prediction points is empty, then count part IoU as 1
            else:
                iou = I / float(U)
            part_ious.append(iou)
        shape_ious.append(np.mean(part_ious))

print("mIOU for class {}: {}".format(opt.class_choice, np.mean(shape_ious)))
