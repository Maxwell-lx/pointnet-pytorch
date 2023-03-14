from __future__ import print_function
import torch.utils.data
import os
import os.path
import torch
import numpy as np
import sys
from tqdm import tqdm
import json



class ShapeNetDataset(torch.utils.data.Dataset):
    def __init__(self,
                 root,
                 npoints=2500,
                 dataset_type='segmentation',
                 class_choice=[],
                 dataset_split='train',
                 data_augmentation=True):
        self.npoints = npoints
        self.root = root
        self.catfile = os.path.join(self.root, 'synsetoffset2category.txt')
        self.cat = {}
        self.data_augmentation = data_augmentation
        self.dataset_type = dataset_type
        self.seg_classes = {}

        print('\nShapeNetDataset: ', dataset_split)
        # segmentation only support single class
        if dataset_type == 'segmentation' and (len(class_choice) == 0 or len(class_choice) > 1):
            raise RuntimeError('Segmentation for this dataset only support training single class, if you want to train segmengtation, you need to modify point label to global label,'
                          'e.g.: you want to train Airplane;4 and Bag:2, then you need to relabel the points to "1,2,3,4" for Airplane and "5,6"for Bag')

        # self.cat self.id2cat
        with open(self.catfile, 'r') as f:
            for line in f:
                ls = line.strip().split()
                self.cat[ls[0]] = ls[1]

        print('Catagory:', self.cat)
        if not len(class_choice) == 0:
            self.cat = {k: v for k, v in self.cat.items() if k in class_choice}

        self.id2cat = {v: k for k, v in self.cat.items()}

        # self.meta self.datapath
        self.meta = {}
        splitfile = os.path.join(self.root, 'train_test_split', 'shuffled_{}_file_list.json'.format(dataset_split))
        filelist = json.load(open(splitfile, 'r'))
        for item in self.cat:
            self.meta[item] = []

        for file in filelist:
            _, category, uuid = file.split('/')
            if category in self.cat.values():
                self.meta[self.id2cat[category]].append((os.path.join(self.root, category, 'points', uuid + '.pts'),
                                                         os.path.join(self.root, category, 'points_label', uuid + '.seg'),
                                                         os.path.join(self.root, category, 'points_normals', uuid + '.pts')))
        self.datapath = []
        for item in self.cat:
            for file_paths in self.meta[item]:
                self.datapath.append((item, file_paths[0], file_paths[1], file_paths[2]))

        # self.classes
        self.classes = dict(zip(sorted(self.cat), range(len(self.cat))))
        if len(class_choice) == 0:
            print('Chosen classes: ALL')
        else:
            print('Chosen classes:', self.classes)

            # self.seg_classes
            with open(os.path.join(os.path.dirname(os.path.realpath(__file__)), 'num_seg_classes.txt'), 'r') as f:
                for line in f:
                    ls = line.strip().split()
                    self.seg_classes[ls[0]] = int(ls[1])
            print(self.seg_classes)
            self.num_seg_classes = self.seg_classes[class_choice[0]]
            print('Num of segs:', self.seg_classes)

    def __getitem__(self, index):
        file_paths = self.datapath[index]
        point_set = np.loadtxt(file_paths[1]).astype(np.float32)

        # normalize point cloud to a unit sphere，center locate at (0,0,0)
        point_set = point_set - np.expand_dims(np.mean(point_set, axis=0), 0)  # center
        dist = np.max(np.sqrt(np.sum(point_set ** 2, axis=1)), 0)
        point_set = point_set / dist

        if self.data_augmentation:
            theta = np.random.uniform(0, np.pi * 2)
            rotation_matrix = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
            point_set[:, [0, 2]] = point_set[:, [0, 2]].dot(rotation_matrix)  # random rotation
            point_set += np.random.normal(0, 0.02, size=point_set.shape)  # random jitter

        # Shuffle point cloud order
        choice = np.random.choice(len(point_set), self.npoints, replace=True)
        point_set = point_set[choice, :]
        # To tensor
        point_set = torch.from_numpy(point_set)

        if self.dataset_type == 'classification':
            cls = self.classes[self.datapath[index][0]]
            cls = torch.from_numpy(np.array([cls]).astype(np.int64))
            return point_set, cls
        elif self.dataset_type == 'segmentation':
            seg = np.loadtxt(file_paths[2]).astype(np.int64)
            seg = seg[choice]
            seg = torch.from_numpy(seg)
            return point_set, seg
        elif self.dataset_type == 'learn_normals':
            normals = np.loadtxt(file_paths[3]).astype((np.float32))
            normals = normals[choice]
            normals = torch.from_numpy(normals)
            return point_set, normals
        else:
            print('No such data type for dataloader!')

    def __len__(self):
        return len(self.datapath)


if __name__ == '__main__':
    dataset = ShapeNetDataset(root='../shapenetcore_partanno_segmentation_benchmark_v0', dataset_type='segmentation', class_choice='Motorbike')  # , class_choice=['Chair']
    print(dataset.seg_classes)
    print(len(dataset))
    points, supervisor = dataset[0]
    print(points.size(), points.type(), supervisor.size(), supervisor.type())
