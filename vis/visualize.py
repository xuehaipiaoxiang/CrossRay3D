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



import importlib
sys.path.append( os.getcwd() )



################ slot

# from nuscenes.nuscenes import NuScenes
# nusc = NuScenes(version='v1.0-mini', dataroot='/data/sets/nuscenes', verbose=True)
# # NuScenes()方法
# nusc.render_annotation()

###############  slot 

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
    parser.add_argument("--mode", type=str, default="gt", choices=["gt", "pred"])
    parser.add_argument("--mode_details", type=str, default="pipeline_gt", choices=["pipeline_gt", "original_gt"])
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--split", type=str, default="train", choices=["train", "val"])
    parser.add_argument("--bbox-classes", nargs="+", type=int, default=None)
    parser.add_argument("--bbox-score", type=float, default=0.25)
    parser.add_argument("--map-score", type=float, default=0.5)
    parser.add_argument("--out-dir", type=str, default="vis/pictures")
    parser.add_argument("--max_img_show_num", type=int, default=20)

    args = parser.parse_args()

    # configs.load(args.config, recursive=True)
    # configs.update(opts)

    # cfg = Config(recursive_eval(configs), filename=args.config)
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
    
    # build the dataloader
    dataset = build_dataset(cfg.data[args.split])
    # datasets = [build_dataset(cfg.data.train)]

    data_loader = build_mmdet_dataloader(
        dataset,
        cfg.data.samples_per_gpu,
        cfg.data.workers_per_gpu,
        # `num_gpus` will be ignored if distributed
        num_gpus=len(cfg.gpu_ids),
        dist=False,
        seed=cfg.seed,
        shuffle=False,
        runner_type='EpochBasedRunner',
        persistent_workers=cfg.data.get('persistent_workers', False))

    # build the model and load checkpoint
    if args.mode == "pred":
        model = build_model(cfg.model)
        load_checkpoint(model, args.checkpoint, map_location="cpu")

        model = MMDataParallel(
            model.cuda(cfg.gpu_ids[0]), device_ids=cfg.gpu_ids)
        model.eval()

    # max_img_show_num = args.max_img_show_num
    # max_img_show_num = 20
    i = 0    
    for data in data_loader:
        metas = data["img_metas"].data[0][0] if args.split =="train" else data["img_metas"][0].data[0][0] 
        # if 'cd7b7f9917364acc9e25e8d21e4dc297' != metas['sample_idx']:
        #     continue

        # if i <= 100:
        #     i+=1
        #     continue
        name = "img_{}".format( str(i))
        i = i + 1
        if args.mode == "pred":
            with torch.no_grad():
                outputs = model(return_loss=False, **data)

        if args.mode == "gt" and "gt_bboxes_3d" in data:
            bboxes = data["gt_bboxes_3d"].data[0][0].tensor.numpy()
            labels = data["gt_labels_3d"].data[0][0].numpy()

            if args.bbox_classes is not None:
                indices = np.isin(labels, args.bbox_classes)
                bboxes = bboxes[indices]
                labels = labels[indices]

            bboxes = LiDARInstance3DBoxes(bboxes, box_dim=9)
        elif args.mode == "pred" and "boxes_3d" in outputs[0]['pts_bbox']:
            bboxes = outputs[0]['pts_bbox']["boxes_3d"].tensor.numpy()
            scores = outputs[0]['pts_bbox']["scores_3d"].numpy()
            labels = outputs[0]['pts_bbox']["labels_3d"].numpy()

            # np.set_printoptions(threshold = np.inf)
            # print(scores)
            # np.set_printoptions(threshold = None)

            if args.bbox_classes is not None:
                indices = np.isin(labels, args.bbox_classes)
                bboxes = bboxes[indices]
                scores = scores[indices]
                labels = labels[indices]

            if args.bbox_score is not None:
                indices = scores >= args.bbox_score
                bboxes = bboxes[indices]
                scores = scores[indices]
                labels = labels[indices]
            # bboxes[..., 2] -= bboxes[..., 5] / 2
            bboxes = LiDARInstance3DBoxes(bboxes, box_dim=9)
        else:
            bboxes = None
            labels = None

        if args.mode == "gt" and "gt_masks_bev" in data:
            masks = data["gt_masks_bev"].data[0].numpy()
            masks = masks.astype(np.bool)
        elif args.mode == "pred" and "masks_bev" in outputs[0]:
            masks = outputs[0]["masks_bev"].numpy()
            masks = masks >= args.map_score
        else:
            masks = None

        if "img" in data:
            for k, image_path in enumerate(metas["filename"]):
                
                if args.mode_details == "pipeline_gt":
                    image = data["img"].data[0][0][k] if args.split =="train" else data["img"][0].data[0][0][k]
                    image = image.cpu().data.numpy().transpose(1, 2, 0)
                    image = imdenormalize(image, mean, std,to_bgr = False)
                    image = image.astype(np.uint8)
                else :
                    image = mmcv.imread(image_path)
                visualize_camera(
                    os.path.join(args.out_dir, f"camera-{k}", f"{name}.png"),
                    image,
                    bboxes=bboxes,
                    labels=labels,
                    transform=metas["lidar2img"][k],
                    classes=cfg.class_names,
                )

        if "points" in data:
            # lidar = data["points"].data[0][0].numpy()
            lidar = data["points"].data[0][0].numpy() if args.split =="train" else data["points"][0].data[0][0].numpy()

            visualize_lidar(
                os.path.join(args.out_dir, "lidar", f"{name}.png"),
                lidar,
                bboxes=bboxes,
                labels=labels,
                xlim=[cfg.point_cloud_range[d] for d in [0, 3]],
                ylim=[cfg.point_cloud_range[d] for d in [1, 4]],
                # classes=cfg.object_classes,
                classes = cfg.class_names,
            )

        if masks is not None:
            visualize_map(
                os.path.join(args.out_dir, "map", f"{name}.png"),
                masks,
                classes=cfg.map_classes,
            )



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

if __name__ == "__main__":
    main()
    # from mmdet3d.core.visualizer.image_vis import (draw_camera_bbox3d_on_img, 
#                                                draw_depth_bbox3d_on_img,
#                                                 draw_lidar_bbox3d_on_img,
#                                                 plot_rect3d_on_img,
#                                                 project_pts_on_img,
#                                                 )


