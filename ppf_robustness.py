import importlib
import os
import open3d
import torch
import numpy as np
from scipy.spatial.transform import Rotation as R
from geometric_registration.preparation import build_ppf_input
from geometric_registration.utils import get_keypts, get_pcd


def noise_Gaussian(points, std):
    noise = np.random.normal(0, std, points.shape)
    out = points + noise
    return out


def load_model():
    # dynamically load the model from snapshot
    module_file_path = f'models/model_conv1d.py'
    module_name = 'models'
    module_spec = importlib.util.spec_from_file_location(module_name, module_file_path)
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)

    model = module.PPFFoldNet(10, 1024)
    model.load_state_dict(torch.load(f'pretrained/sun3d_best.pkl'))
    model.eval()

    return model


def visualize(frag_pcd, keypts_pcd):
    visualizer = open3d.visualization.Visualizer()
    visualizer.create_window()
    keypts_pcd.paint_uniform_color(np.array([0.5, 0.5, 0.5]))
    visualizer.add_geometry(keypts_pcd)
    visualizer.add_geometry(frag_pcd)
    visualizer.run()
    visualizer.destroy_window()


def get_codeword(local_patches, model):
    input_ = torch.tensor(local_patches)
    input_ = input_.cuda()
    model = model.cuda()
    # cuda out of memry
    desc_list = []
    for i in range(50):
        step_size = int(5000 / 50)
        desc = model.encoder(input_[i * step_size: (i + 1) * step_size, :, :])
        desc_list.append(desc.detach().cpu().numpy())
        del desc
    desc = np.concatenate(desc_list, 0).reshape([5000, 512])

    return desc


def get_ppf_debug(frag_id):
    frag_name = f'cloud_bin_{frag_id}'
    frag_pcd = get_pcd(pcdpath, frag_name)
    frag = np.array(frag_pcd.points)
    print(frag.shape)

    keypts = get_keypts(keyptspath, frag_name)
    keypts_pcd = open3d.geometry.PointCloud()
    keypts_pcd.points = open3d.utility.Vector3dVector(keypts)
    print(keypts.shape) # [n, 3]

    # res = [(keypts[i] == frag).all(axis=1).any() for i in range(1000, 1100)]
    # print(res)

    visualize(frag_pcd, keypts_pcd)

    # F = { f (xr, x1) · · · f (xr, xi) · · · f (xr, xN) }, (num_points, N = 1024, 4)
    # 我们已经证明如果对f中四个元素进行指定处理（矩阵）确实是旋转不变，但是得看代码中PPF是怎么采用的（是四个元素的L2距离吗）
    local_patches = build_ppf_input(frag_pcd, keypts)
    print(local_patches.shape) # [n, 1024, 4]

    return local_patches


def get_ppf_default(frag_id, visualized=False):
    frag_name = f'cloud_bin_{frag_id}'
    frag_pcd = get_pcd(pcdpath, frag_name)
    # print(frag_pcd.has_normals())
    keypts = get_keypts(keyptspath, frag_name)
    keypts_pcd = open3d.geometry.PointCloud()
    keypts_pcd.points = open3d.utility.Vector3dVector(keypts)
    keypts = np.array(keypts_pcd.points)

    if visualized:
        visualize(frag_pcd, keypts_pcd)

    local_patches = build_ppf_input(frag_pcd, keypts, experiment=True)
    return local_patches


def get_ppf_translate(frag_id, T_mat=None, visualized=False):
    trans_mat = np.eye(4)
    if T_mat is None:
        T_mat = np.random.rand(3)
    trans_mat[:3, 3] = T_mat
    # print(trans_mat)

    frag_name = f'cloud_bin_{frag_id}'
    frag_pcd = get_pcd(pcdpath, frag_name)
    frag_pcd.transform(trans_mat)
    keypts = get_keypts(keyptspath, frag_name)
    keypts_pcd = open3d.geometry.PointCloud()
    keypts_pcd.points = open3d.utility.Vector3dVector(keypts)
    keypts_pcd.transform(trans_mat)
    keypts = np.array(keypts_pcd.points)

    if visualized:
        visualize(frag_pcd, keypts_pcd)

    local_patches = build_ppf_input(frag_pcd, keypts, True)
    return local_patches


def get_ppf_rotate(frag_id, R_mat=None, visualized=False):
    trans_mat = np.eye(4)
    if R_mat is None:
        R_mat = R.random().as_matrix()
    elif len(R_mat.shape) == 1 and R_mat.shape[0] == 3:
        R_mat = R.from_euler('xyz', R_mat).as_matrix()
    trans_mat[:3, :3] = R_mat
    # print(trans_mat)

    frag_name = f'cloud_bin_{frag_id}'
    frag_pcd = get_pcd(pcdpath, frag_name)
    frag_pcd.transform(trans_mat)
    keypts = get_keypts(keyptspath, frag_name)
    keypts_pcd = open3d.geometry.PointCloud()
    keypts_pcd.points = open3d.utility.Vector3dVector(keypts)
    keypts_pcd.transform(trans_mat)
    keypts = np.array(keypts_pcd.points)

    if visualized:
        visualize(frag_pcd, keypts_pcd)

    local_patches = build_ppf_input(frag_pcd, keypts, True)
    return local_patches


def get_ppf_remove(frag_id, remove_prop=0.8, visualized=False):
    frag_name = f'cloud_bin_{frag_id}'
    frag_pcd = get_pcd(pcdpath, frag_name)
    frag = np.array(frag_pcd.points)
    # print(frag.shape)
    prop = np.random.rand(frag.shape[0])
    prop = (prop < remove_prop)
    frag_new = []
    for i in range(len(prop)):
        if prop[i]:
            frag_new.append(frag[i])
    frag_new = np.concatenate(frag_new).reshape(-1, 3)
    # print(frag_new.shape)
    frag_pcd = open3d.geometry.PointCloud()
    frag_pcd.points = open3d.utility.Vector3dVector(frag_new)

    keypts = get_keypts(keyptspath, frag_name)
    keypts_pcd = open3d.geometry.PointCloud()
    keypts_pcd.points = open3d.utility.Vector3dVector(keypts)
    keypts = np.array(keypts_pcd.points)

    if visualized:
        visualize(frag_pcd, keypts_pcd)

    local_patches = build_ppf_input(frag_pcd, keypts, True)
    return local_patches


def get_ppf_perturb(frag_id, std=0.05, visualized=False):
    frag_name = f'cloud_bin_{frag_id}'
    frag_pcd = get_pcd(pcdpath, frag_name)
    frag = np.array(frag_pcd.points)
    # print(frag.shape)
    frag_new = noise_Gaussian(frag, std)
    # print(frag_new.shape)
    frag_pcd = open3d.geometry.PointCloud()
    frag_pcd.points = open3d.utility.Vector3dVector(frag_new)

    keypts = get_keypts(keyptspath, frag_name)
    keypts_pcd = open3d.geometry.PointCloud()
    keypts_pcd.points = open3d.utility.Vector3dVector(keypts)
    keypts = np.array(keypts_pcd.points)

    if visualized:
        visualize(frag_pcd, keypts_pcd)

    local_patches = build_ppf_input(frag_pcd, keypts, True)
    return local_patches


if __name__ == "__main__":
    scene_list = [
        # '7-scenes-redkitchen',
        # 'sun3d-home_at-home_at_scan1_2013_jan_1',
        # 'sun3d-home_md-home_md_scan9_2012_sep_30',
        # 'sun3d-hotel_uc-scan3',
        # 'sun3d-hotel_umd-maryland_hotel1',
        'sun3d-hotel_umd-maryland_hotel3',
        # 'sun3d-mit_76_studyroom-76-1studyroom2',
        # 'sun3d-mit_lab_hj-lab_hj_tea_nov_2_2012_scan1_erika'
    ]
    model = load_model()
    for scene in scene_list:
        pcdpath = f"data/3DMatch/fragments/{scene}/"
        interpath = f"data/3DMatch/intermediate-files-real/{scene}/"
        keyptspath = interpath

        all_frags = os.listdir(pcdpath)
        num_frags = len(os.listdir(pcdpath))
        frag_id = 0

        frag_ppf = get_ppf_default(frag_id)
        print(frag_ppf[7])
        dd = frag_ppf
        frag_codeword = get_codeword(frag_ppf, model)

        # print('Default')
        # frag_ppf = get_ppf_default(frag_id)
        # frag_codeword_default = get_codeword(frag_ppf, model)
        #
        # l2 = np.linalg.norm(frag_codeword_default - frag_codeword, ord=2, axis=1)
        # l2_avg = np.average(l2)
        # print(l2_avg)

        print('Translate')
        frag_ppf = get_ppf_translate(frag_id)
        print(frag_ppf[7])
        frag_codeword_translate = get_codeword(frag_ppf, model)

        l2 = np.linalg.norm(frag_codeword_translate - frag_codeword, ord=2, axis=1)
        l2_avg = np.average(l2)
        print(l2_avg)

        # print('Rotate')
        # frag_ppf = get_ppf_rotate(frag_id)
        # print(frag_ppf[0])
        # frag_codeword_rotate = get_codeword(frag_ppf, model)
        #
        # l2 = np.linalg.norm(frag_codeword_rotate - frag_codeword, ord=2, axis=1)
        # l2_avg = np.average(l2)
        # print(l2_avg)
        #
        # print('Remove')
        # frag_ppf = get_ppf_remove(frag_id, 0.4)
        # frag_codeword_remove = get_codeword(frag_ppf, model)
        #
        # l2 = np.linalg.norm(frag_codeword_remove - frag_codeword, ord=2, axis=1)
        # l2_avg = np.average(l2)
        # print(l2_avg)
        #
        # print('Perturb')
        # frag_ppf = get_ppf_perturb(frag_id, 0.01)
        # frag_codeword_perturb = get_codeword(frag_ppf, model)
        #
        # l2 = np.linalg.norm(frag_codeword_perturb - frag_codeword, ord=2, axis=1)
        # l2_avg = np.average(l2)
        # print(l2_avg)