# Modified by Huiming Yang
import argparse
import copy
import os
import sys
import mmcv
import numpy as np
import torch
from mmcv import Config
from mmcv.parallel import MMDistributedDataParallel, MMDataParallel
from mmcv.runner import load_checkpoint
from mmdet.datasets import build_dataloader as build_mmdet_dataloader
from mmdet3d.apis import init_random_seed
from mmdet3d.core import LiDARInstance3DBoxes
from  vis_utils import visualize_camera, visualize_lidar, visualize_map
from mmdet3d.datasets import  build_dataset
from mmdet3d.models import build_model
from mmcv.image.photometric import imdenormalize
from mmdet.apis import set_random_seed
from mmcv.runner import get_dist_info, init_dist
import torch.distributed as dist
import random
from typing import List, Optional, Tuple
import copy
import os
from typing import List, Optional, Tuple
import cv2
import mmcv
import numpy as np
from matplotlib import pyplot as plt
from mmdet3d.core import LiDARInstance3DBoxes



import importlib
sys.path.append( os.getcwd() )


def recursive_eval(obj, globals=None):
    if globals is None:
        globals = copy.deepcopy(obj)

    if isinstance(obj, dict):
        for key in obj:
            obj[key] = recursive_eval(obj[key], globals)
    elif isinstance(obj, list):
        for k, val in enumerate(obj):
            obj[k] = recursive_eval(val, globals)
    elif isinstance(obj, str) and obj.startswith("${") and obj.endswith("}"):
        obj = eval(obj[2:-1], globals)
        obj = recursive_eval(obj, globals)

    return obj


# cd7b7f9917364acc9e25e8d21e4dc297

mean=np.array([103.530, 116.280, 123.675]) # copy from config
std=np.array([57.375, 57.120, 58.395])

def main() -> None:
    importlib.import_module("projects.mmdet3d_plugin")

    seed = init_random_seed(0)
    set_random_seed(seed)

    parser = argparse.ArgumentParser()
    parser.add_argument("config", metavar="FILE")
    parser.add_argument("--mode", type=str, default="gt", choices=["gt", "pred"])  # model 
    parser.add_argument("--mode_details", type=str, default="pipeline_gt", choices=["pipeline_gt", "original_gt"])
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--split", type=str, default="train", choices=["train", "val"]) # pipeline
    parser.add_argument("--bbox-classes", nargs="+", type=int, default=None)
    parser.add_argument("--bbox-score", type=float, default=0.35)
    parser.add_argument("--map-score", type=float, default=0.5)
    parser.add_argument("--out-dir", type=str, default="vis/pictures")
    parser.add_argument("--max_img_show_num", type=int, default=20)

    args = parser.parse_args()

    cfg = Config.fromfile(args.config)

    # set cudnn_benchmark
    # torch.backends.cudnn.benchmark = cfg.cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True

    # if args.gpus is not None:
    cfg.gpu_ids = range(1)
    cfg.seed = init_random_seed(0)
    cfg.data.samples_per_gpu = 1
    cfg.data.workers_per_gpu = 2
    
    dataset = build_dataset(cfg.data[args.split])
    data_loader = build_mmdet_dataloader(
        dataset,
        cfg.data.samples_per_gpu,
        cfg.data.workers_per_gpu,
        num_gpus=len(cfg.gpu_ids),
        dist=False,
        seed=cfg.seed,
        shuffle=False,
        runner_type='EpochBasedRunner',
        persistent_workers=cfg.data.get('persistent_workers', False))

    model = build_model(cfg.model)
    load_checkpoint(model, args.checkpoint, map_location="cpu")

    model = MMDataParallel(
        model.cuda(cfg.gpu_ids[0]), device_ids=cfg.gpu_ids)
    model.eval()

    i = 0    
    for data in data_loader:
        metas = data["img_metas"].data[0][0] if args.split =="train" else data["img_metas"][0].data[0][0] 
        name = "img_{}".format( str(i))
        i = i + 1

        # training format to test format
        data_test = {
            'points': [data['points']],
            'img_metas': [data['img_metas']],
                     }

        with torch.no_grad():
            outputs = model(return_loss=False, **data_test)

        bboxes_gt = data["gt_bboxes_3d"].data[0][0].tensor.numpy()
        labels_gt = data["gt_labels_3d"].data[0][0].numpy()
        # indices = np.isin(labels_gt, args.bbox_classes)
        # bboxes_gt = bboxes_gt[indices]
        # labels_gt = labels_gt[indices]
        bboxes_gt = LiDARInstance3DBoxes(bboxes_gt, box_dim=9)

        bboxes_pred = outputs[0]['pts_bbox']["boxes_3d"].tensor.numpy()
        scores_pred = outputs[0]['pts_bbox']["scores_3d"].numpy()
        labels_pred = outputs[0]['pts_bbox']["labels_3d"].numpy()


            # if args.bbox_classes is not None:
            #     indices = np.isin(labels, args.bbox_classes)
            #     bboxes = bboxes[indices]
            #     scores = scores[indices]
            #     labels = labels[indices]

        if args.bbox_score is not None:
            indices = scores_pred >= args.bbox_score
            bboxes_pred = bboxes_pred[indices]
            scores_pred = scores_pred[indices]
            labels_pred = labels_pred[indices]
            bboxes_pred = LiDARInstance3DBoxes(bboxes_pred, box_dim=9)
        else:
            bboxes = None
            labels = None

        if "points" in data:
            lidar = data["points"].data[0][0].numpy() if args.split =="train" else data["points"][0].data[0][0].numpy()

            visualize_lidar(
                os.path.join(args.out_dir, "lidar", f"{name}.png"),
                lidar,
                bboxes_gt = bboxes_gt,
                labels_gt = labels_gt,
                bboxes_pred = bboxes_pred,
                labels_pred = labels_pred,
                xlim=[cfg.point_cloud_range[d] for d in [0, 3]],
                ylim=[cfg.point_cloud_range[d] for d in [1, 4]],
                # classes=cfg.object_classes,
                classes = cfg.class_names,
            )






def visualize_lidar(
    fpath: str,
    lidar: Optional[np.ndarray] = None,
    *,
    bboxes_gt: Optional[LiDARInstance3DBoxes] = None,
    labels_gt: Optional[np.ndarray] = None,
    bboxes_pred: Optional[LiDARInstance3DBoxes] = None,
    labels_pred: Optional[np.ndarray] = None,
    classes: Optional[List[str]] = None,
    xlim: Tuple[float, float] = (-50, 50),
    ylim: Tuple[float, float] = (-50, 50),
    color: Optional[Tuple[int, int, int]] = None,
    radius: float = 1,
    thickness: float = 10,
) -> None:
    
    fig = plt.figure(figsize=(xlim[1] - xlim[0], ylim[1] - ylim[0]))
    ax = plt.gca()
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_aspect(1)
    ax.set_axis_off()

    if lidar is not None:
        plt.scatter(
            lidar[:, 0],
            lidar[:, 1],
            s=radius,
            c="white",
        )

    if bboxes_gt is not None and len(bboxes_gt) > 0:
        coords = bboxes_gt.corners[:, [0, 3, 7, 4, 0], :2]
        for index in range(coords.shape[0]):
            name = classes[labels_gt[index]]
            plt.plot(
                coords[index, :, 0],
                coords[index, :, 1],
                linewidth=thickness,
                color=(1, 0, 0),
            )
    
    
    if bboxes_pred is not None and len(bboxes_pred) > 0:
        coords = bboxes_pred.corners[:, [0, 3, 7, 4, 0], :2]
        for index in range(coords.shape[0]):
            name = classes[labels_pred[index]]
            plt.plot(
                coords[index, :, 0],
                coords[index, :, 1],
                linewidth=thickness,
                color=(0, 1, 0),
            )

    mmcv.mkdir_or_exist(os.path.dirname(fpath))
    fig.savefig(
        fpath,
        dpi=10,
        facecolor="black",
        format="png",
        bbox_inches="tight",
        pad_inches=0,
    )
    plt.close()


def init_random_seed(seed=None, device='cuda'):
    """Initialize random seed.

    If the seed is not set, the seed will be automatically randomized,
    and then broadcast to all processes to prevent some potential bugs.

    Args:
        seed (int, Optional): The seed. Default to None.
        device (str): The device where the seed will be put on.
            Default to 'cuda'.

    Returns:
        int: Seed to be used.
    """
    if seed is not None:
        return seed

    # Make sure all ranks share the same random seed to prevent
    # some potential bugs. Please refer to
    # https://github.com/open-mmlab/mmdetection/issues/6339
    rank, world_size = get_dist_info()
    seed = np.random.randint(2**31)
    if world_size == 1:
        return seed

    if rank == 0:
        random_num = torch.tensor(seed, dtype=torch.int32, device=device)
    else:
        random_num = torch.tensor(0, dtype=torch.int32, device=device)
    dist.broadcast(random_num, src=0)
    return random_num.item()


def set_random_seed(seed, deterministic=False):
    """Set random seed.

    Args:
        seed (int): Seed to be used.
        deterministic (bool): Whether to set the deterministic option for
            CUDNN backend, i.e., set `torch.backends.cudnn.deterministic`
            to True and `torch.backends.cudnn.benchmark` to False.
            Default: False.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


__all__ = ["visualize_camera", "visualize_lidar", "visualize_map"]


OBJECT_PALETTE = {
    "car": (255, 158, 0),
    "truck": (255, 99, 71),
    "construction_vehicle": (233, 150, 70),
    "bus": (255, 69, 0),
    "trailer": (255, 140, 0),
    "barrier": (112, 128, 144),
    "motorcycle": (255, 61, 99),
    "bicycle": (220, 20, 60),
    "pedestrian": (0, 0, 230),
    "traffic_cone": (47, 79, 79),
}

MAP_PALETTE = {
    "drivable_area": (166, 206, 227),
    "road_segment": (31, 120, 180),
    "road_block": (178, 223, 138),
    "lane": (51, 160, 44),
    "ped_crossing": (251, 154, 153),
    "walkway": (227, 26, 28),
    "stop_line": (253, 191, 111),
    "carpark_area": (255, 127, 0),
    "road_divider": (202, 178, 214),
    "lane_divider": (106, 61, 154),
    "divider": (106, 61, 154),
}

LINE_INDICES = ((0, 1), (0, 3), (0, 4), (1, 2), (1, 5), (3, 2), (3, 7),
                    (4, 5), (4, 7), (2, 6), (5, 6), (6, 7))

if __name__ == "__main__":
    main()



