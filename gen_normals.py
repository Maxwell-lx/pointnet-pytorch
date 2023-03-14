import open3d as o3d
import numpy as np
import os
from tqdm import tqdm

root = '../shapenetcore_partanno_segmentation_benchmark_v0'
pcd = o3d.geometry.PointCloud()

with open(os.path.join(root,'synsetoffset2category.txt'), 'r') as f:
    for line in f:
        ls = line.strip().split()
        uuid = ls[1]
        files = os.listdir(os.path.join(root, uuid, 'points'))
        print(uuid)
        for _,i in tqdm(enumerate(files,0)):
            path = os.path.join(root, uuid, 'points', i)
            xyz = np.loadtxt(path)
            pcd.points = o3d.utility.Vector3dVector(xyz)
            # o3d.visualization.draw_geometries([pcd], point_show_normal=True)
            pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.5, max_nn=10))
            # pcd.orient_normals_towards_camera_location()
            # o3d.visualization.draw_geometries([pcd], point_show_normal=True)
            normals = np.asarray(pcd.normals)
            normals = normals.tolist()

            if os.path.exists(os.path.join(root, uuid, 'points_normals')):
                pass
            else:
                os.mkdir(os.path.join(root, uuid, 'points_normals'))
            w_path = os.path.join(root, uuid, 'points_normals', i)
            with open(w_path, "w+") as fp:
                fp.writelines("\n".join([" ".join([str(format(i, '.5f')) for i in j]) for j in normals]))
