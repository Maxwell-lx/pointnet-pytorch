import torch
import dataset


def save_model(save_path, optimizer, model):
    torch.save({'optimizer_dict': optimizer.state_dict(),
                'model_dict': model.state_dict()},
               save_path)


def load_model(save_name, optimizer, model):
    model_data = torch.load(save_name)
    model.load_state_dict(model_data['model_dict'])
    optimizer.load_state_dict(model_data['optimizer_dict'])


def blue(x):
    return '\033[94m' + x + '\033[0m'


# input: batch_size x num_points x dimension
# loss 向量夹角越接近0或180度，loss越小
def learn_normals_loss(pred, target):
    dimension = pred.size()[2]
    pred = pred.view(-1, dimension, 1)
    # normalize to unit vector
    pred_normalize = torch.bmm(pred, torch.rsqrt(torch.bmm(torch.transpose(pred, 1, 2), pred)))

    target = target.view(-1, dimension, 1)
    target_normalize = torch.bmm(target, torch.rsqrt(torch.bmm(torch.transpose(target, 1, 2), target)))
    target_normalize = torch.transpose(target_normalize, 1, 2)

    temp = torch.bmm(target_normalize, pred_normalize)
    temp = temp.view(-1)
    # temp = (1 - torch.abs(temp)) # L1 loss
    temp = (1 - torch.abs(temp)) * (1 - torch.abs(temp))  # L2 loss

    return torch.mean(temp)


if __name__ == '__main__':
    dataset = dataset.ShapeNetDataset(root='../shapenetcore_partanno_segmentation_benchmark_v0', dataset_type='learn_normals', class_choice=['Chair'], data_augmentation=False)
    points, target = dataset[2]
    target = target.view(1, -1, 3)
    print(target.shape)

    # loss of normals itself is 0
    print(learn_normals_loss(target, target))
    print(learn_normals_loss(target, -target))
    print(learn_normals_loss(-target, target))

    # loss of 2 rand vector is 0.5 in L1 and 0.33 in L2
    test1 = torch.rand(1, 2500, 3)
    test2 = torch.rand(1, 2500, 3)
    print(learn_normals_loss(test1 - 0.5, test2 - 0.5))

    test = torch.rand(1, 2500, 3) - 0.5
    jitter = (torch.rand(1, 2500, 3) - 0.5) * 0.3
    print(learn_normals_loss(test, test + jitter))
