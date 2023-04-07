# pointnet-pytorch

This is a pytorch version of [pointnet](https://github.com/charlesq34/pointnet), a classic framework for point cloud learning.

This project is forked from [pointnet.pytorch](https://github.com/fxia22/pointnet.pytorch) and adding a **learn-normals** test to testify the ability to integrate information from neighborhood, which is considered to be one of the most important features of CNN.

## learn-normals

![pointnet reconstract normals](https://github.com/Maxwell-lx/static/blob/main/images/pointnet%20reconstract%20normals.png)

This result comes from the original pointnet paper. The normals are calculated from mesh. 

You can also regenerate normals directly from point cloud through [open3d](http://www.open3d.org/docs/release/tutorial/geometry/pointcloud.html#Vertex-normal-estimation) or [python-pcl](https://pcl.readthedocs.io/projects/tutorials/en/master/normal_estimation.html#normal-estimation).

But none of them can get a proper direction for normals without a shooting position.

Loss function can ignore this by 
$$
L2\_loss = mean(\sum {{{\left( {{{\vec n}_1} \cdot {{\vec n}_2}} \right)}^2})}
$$

n1 and n2 are unit normals

if n1 and n2 are collinear, then loss = 0

if n1 and n2 are random vectors, then the expectation of L1 loss is 0.5 and the expectation of L2 loss is 0.333

## Disable STN3 and data augmentation

Although STN3 work well for classification and segmentation, but STN3 should be disabled in **learn-normals**.

Beacuse STN3 applies an affine transformation on raw data, the affine transformation can not keep the normal property of geometry, same to data augmentaion.

## 1.Download dataset

```bash
./download.sh
```

## 2.Generate normals

```
python gen_normals.py
```

## 3. train and evaluate 

```
python train_classificaton.py

python train_segmentation.py

python train_learn_normals.py
```

## Result of learn-normals

After about 100 epoch, L1 loss descend to 0.2, L2 loss descend to 0.09.

The experimental results have demonstrated the capability of PointNet to learn the normals of point clouds. 

This means that PointNet has the ability to integrate information from the local neighborhoods of points, even when they are presented in a disorderly manner.

## Possible improvements

Although the training loss has been reduced significantly, PointNet is expected to learn the basic ability to understand normal vectors in a short period of time. 

Therefore, I believe that there is still room for improvement in the code, such as disabling batch normalization. 

Another approach to improvement is to adjust the normal vectors synchronously with the adjustments made to STN, batch normalization, and data augmentation. Because when these operations adjust the original data, the normal vectors also change synchronously.
