import numpy as np
import torch
import mmcv
import mmengine
from mmengine.registry import TRANSFORMS
from PIL import Image
import random
from mmengine.registry import init_default_scope
from mmengine.dataset import Compose
from copy import deepcopy

@TRANSFORMS.register_module()
class CropResizeFlipImage(object):
    """Fixed Crop and then randim resize and flip the image. Note the flip requires to flip the feature in the network   
    ida_aug_conf = {
        "reisze": [576, 608, 640, 672, 704]  # stride of 32 based on 640 (0.9, 1.1)
        "reisze": [512, 544, 576, 608, 640, 672, 704, 736, 768]  #  (0.8, 1.2)
        "reisze": [512, 544, 576, 608, 640, 672, 704, 736, 768]  #  (0.8, 1.2)
        "reisze": [448, 480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800, 832]  #  (0.7, 1.3)
        "crop": (0, 260, 1600, 900), 
        "H": 900,
        "W": 1600,
        "rand_flip": True,
}
    Args:
        size (tuple, optional): Fixed padding size.
    """

    def __init__(self, data_aug_conf=None, training=True, debug=False):
        self.data_aug_conf = data_aug_conf
        self.training = training
        self.debug = debug

    def __call__(self, results):
        """Call function to pad images, masks, semantic segmentation maps.
        Args:
            results (dict): Result dict from loading pipeline.
        Returns:
            dict: Updated result dict.
        """
        if not 'aug_param' in results.keys():
            results['aug_param'] = {}
        imgs = results["img"]
        N = len(imgs)
        new_imgs = []
        resize, resize_dims, crop, flip = self._sample_augmentation(results)

        if self.debug:
            # unique id per img
            from uuid import uuid4
            uid = uuid4()
            # lidar is RFU in nuscenes
            lidar_pts = np.array([
                [10, 30, -2, 1],
                [-10, 30, -2, 1],
                [5, 15, -2, 1],
                [-5, 15, -2, 1],
                [30, 0, -2, 1],
                [-30, 0, -2, 1],
                [10, -30, -2, 1],
                [-10, -30, -2, 1]
            ], dtype=np.float32).T

        for i in range(N):
            img = Image.fromarray(np.uint8(imgs[i]))

            if self.debug:
                pts_to_img_pre_aug = results['lidar2img'][i] @ lidar_pts
                pts_to_img_pre_aug = pts_to_img_pre_aug / pts_to_img_pre_aug[2:3,
                                                          :]  # div by the depth component in homogenous vector

                img_copy = Image.fromarray(np.uint8(imgs[i]))
                for j in range(pts_to_img_pre_aug.shape[1]):
                    x, y = int(pts_to_img_pre_aug[0, j]), int(pts_to_img_pre_aug[1, j])
                    if (0 < x < img_copy.width) and (0 < y < img_copy.height):
                        img_copy.putpixel((x - 1, y - 1), (255, 0, 0))
                        img_copy.putpixel((x - 1, y), (255, 0, 0))
                        img_copy.putpixel((x - 1, y + 1), (255, 0, 0))
                        img_copy.putpixel((x, y - 1), (0, 255, 0))
                        img_copy.putpixel((x, y), (0, 255, 0))
                        img_copy.putpixel((x, y + 1), (0, 255, 0))
                        img_copy.putpixel((x + 1, y - 1), (0, 0, 255))
                        img_copy.putpixel((x + 1, y), (0, 0, 255))
                        img_copy.putpixel((x + 1, y + 1), (0, 0, 255))
                img_copy.save(f'pre_aug_{uid}_{i}.png')

            # augmentation (resize, crop, horizontal flip, rotate)
            # resize, resize_dims, crop, flip, rotate = self._sample_augmentation()  ###different view use different aug (BEV Det)
            img, ida_mat = self._img_transform(
                img,
                resize=resize,
                resize_dims=resize_dims,
                crop=crop,
                flip=flip,
            )
            new_imgs.append(np.array(img).astype(np.float32))
            results['cam2img'][i][:3, :3] = np.matmul(ida_mat, results['cam2img'][i][:3, :3])

            if self.debug:
                pts_to_img_post_aug = np.matmul(results['cam2img'][i], results['lidar2cam'][i]) @ lidar_pts
                pts_to_img_post_aug = pts_to_img_post_aug / pts_to_img_post_aug[2:3,
                                                            :]  # div by the depth component in homogenous vector
                for j in range(pts_to_img_post_aug.shape[1]):
                    x, y = int(pts_to_img_post_aug[0, j]), int(pts_to_img_post_aug[1, j])
                    if (0 < x < img.width) and (0 < y < img.height):
                        img.putpixel((x - 1, y - 1), (255, 0, 0))
                        img.putpixel((x - 1, y), (255, 0, 0))
                        img.putpixel((x - 1, y + 1), (255, 0, 0))
                        img.putpixel((x, y - 1), (0, 255, 0))
                        img.putpixel((x, y), (0, 255, 0))
                        img.putpixel((x, y + 1), (0, 255, 0))
                        img.putpixel((x + 1, y - 1), (0, 0, 255))
                        img.putpixel((x + 1, y), (0, 0, 255))
                        img.putpixel((x + 1, y + 1), (0, 0, 255))
                img.save(f'post_aug_{uid}_{i}.png')

            if 'mono_ann_idx' in results.keys():
                # apply transform to dd3d intrinsics
                if i in results['mono_ann_idx'].data:
                    mono_index = results['mono_ann_idx'].data.index(i)
                    intrinsics = results['mono_input_dict'][mono_index]['intrinsics']
                    if torch.is_tensor(intrinsics):
                        intrinsics = intrinsics.numpy().reshape(3, 3).astype(np.float32)
                    elif isinstance(intrinsics, np.ndarray):
                        intrinsics = intrinsics.reshape(3, 3).astype(np.float32)
                    else:
                        intrinsics = np.array(intrinsics, dtype=np.float32).reshape(3, 3)
                    results['mono_input_dict'][mono_index]['intrinsics'] = np.matmul(ida_mat, intrinsics)
                    results['mono_input_dict'][mono_index]['height'] = img.size[1]
                    results['mono_input_dict'][mono_index]['width'] = img.size[0]

                    # apply transform to dd3d box
                    for ann in results['mono_input_dict'][mono_index]['annotations']:
                        # bbox_mode = BoxMode.XYXY_ABS
                        box = self._box_transform(ann['bbox'], resize, crop, flip, img.size[0])[0]
                        box = box.clip(min=0)
                        box = np.minimum(box, list(img.size + img.size))
                        ann["bbox"] = box

        results["img"] = new_imgs
        results['lidar2img'] = [np.matmul(results['cam2img'][i], results['lidar2cam'][i]) for i in
                                range(len(results['lidar2cam']))]

        return results

    def _box_transform(self, box, resize, crop, flip, img_width):
        box = np.array([box])
        idxs = np.array([(0, 1), (2, 1), (0, 3), (2, 3)]).flatten()
        coords = np.asarray(box).reshape(-1, 4)[:, idxs].reshape(-1, 2)

        # crop
        coords[:, 0] -= crop[0]
        coords[:, 1] -= crop[1]

        # resize
        coords[:, 0] = coords[:, 0] * resize
        coords[:, 1] = coords[:, 1] * resize

        coords = coords.reshape((-1, 4, 2))
        minxy = coords.min(axis=1)
        maxxy = coords.max(axis=1)
        trans_box = np.concatenate((minxy, maxxy), axis=1)

        return trans_box

    def _img_transform(self, img, resize, resize_dims, crop, flip):
        ida_rot = np.eye(2)
        ida_tran = np.zeros(2)
        # adjust image
        img = img.crop(crop)
        img = img.resize(resize_dims)
        if flip:
            img = img.transpose(method=Image.FLIP_LEFT_RIGHT)

        # post-homography transformation
        ida_rot *= resize
        ida_tran -= np.array(crop[:2]) * resize
        ida_mat = np.eye(3)
        ida_mat[:2, :2] = ida_rot
        ida_mat[:2, 2] = ida_tran
        return img, ida_mat

    def _sample_augmentation(self, results):
        if 'CropResizeFlipImage_param' in results['aug_param'].keys():
            return results['aug_param']['CropResizeFlipImage_param']
        crop = self.data_aug_conf["crop"]

        if self.training:
            resized_h = random.choice(self.data_aug_conf["reisze"])
            resized_w = resized_h / (crop[3] - crop[1]) * (crop[2] - crop[0])
            resize = resized_h / (crop[3] - crop[1])
            resize_dims = (int(resized_w), int(resized_h))
            flip = False
            if self.data_aug_conf["rand_flip"] and np.random.choice([0, 1]):
                flip = True
        else:
            resized_h = random.choice(self.data_aug_conf["reisze"])
            assert len(self.data_aug_conf["reisze"]) == 1
            resized_w = resized_h / (crop[3] - crop[1]) * (crop[2] - crop[0])
            resize = resized_h / (crop[3] - crop[1])
            resize_dims = (int(resized_w), int(resized_h))
            flip = False
        results['aug_param']['CropResizeFlipImage_param'] = (resize, resize_dims, crop, flip)

        return resize, resize_dims, crop, flip


@TRANSFORMS.register_module()
class GlobalRotScaleTransImage(object):
    """Random resize, Crop and flip the image
    Args:
        size (tuple, optional): Fixed padding size.
    """

    def __init__(
            self,
            rot_range=[-0.3925, 0.3925],
            scale_ratio_range=[0.95, 1.05],
            translation_std=[0, 0, 0],
            reverse_angle=False,
            training=True,
            flip_dx_ratio=0.5,
            flip_dy_ratio=0.5,
            only_gt=False,
    ):

        self.rot_range = rot_range
        self.scale_ratio_range = scale_ratio_range
        self.translation_std = translation_std

        self.reverse_angle = reverse_angle
        self.training = training

        self.flip_dx_ratio = flip_dx_ratio
        self.flip_dy_ratio = flip_dy_ratio
        self.only_gt = only_gt

    def __call__(self, results):
        """Call function to pad images, masks, semantic segmentation maps.
        Args:
            results (dict): Result dict from loading pipeline.
        Returns:
            dict: Updated result dict.
        """
        if not 'aug_param' in results.keys():
            results['aug_param'] = {}

        rot_angle, scale_ratio, flip_dx, flip_dy, _, _ = self._sample_augmentation(results)

        # random rotate
        if not self.only_gt:
            self.rotate_bev_along_z(results, rot_angle)
        if self.reverse_angle:
            rot_angle *= -1
        results["gt_bboxes_3d"].rotate(
            np.array(rot_angle)
        )

        # random scale
        if not self.only_gt:
            self.scale_xyz(results, scale_ratio)
        results["gt_bboxes_3d"].scale(scale_ratio)

        # random flip
        if flip_dx:
            if not self.only_gt:
                self.flip_along_x(results)
            results["gt_bboxes_3d"].flip(bev_direction='vertical')
        if flip_dy:
            if not self.only_gt:
                self.flip_along_y(results)
            results["gt_bboxes_3d"].flip(bev_direction='horizontal')

        # TODO: support translation
        return results

    def _sample_augmentation(self, results):
        if 'GlobalRotScaleTransImage_param' in results['aug_param'].keys():
            return results['aug_param']['GlobalRotScaleTransImage_param']
        else:
            rot_angle = np.random.uniform(*self.rot_range) / 180 * np.pi
            scale_ratio = np.random.uniform(*self.scale_ratio_range)
            flip_dx = np.random.uniform() < self.flip_dx_ratio
            flip_dy = np.random.uniform() < self.flip_dy_ratio
        # generate bda_mat 

        rot_sin = torch.sin(torch.tensor(rot_angle))
        rot_cos = torch.cos(torch.tensor(rot_angle))
        rot_mat = torch.Tensor([[rot_cos, -rot_sin, 0], [rot_sin, rot_cos, 0],
                                [0, 0, 1]])
        scale_mat = torch.Tensor([[scale_ratio, 0, 0], [0, scale_ratio, 0],
                                  [0, 0, scale_ratio]])
        flip_mat = torch.Tensor([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        if flip_dx:
            flip_mat = flip_mat @ torch.Tensor([[-1, 0, 0], [0, 1, 0],
                                                [0, 0, 1]])
        if flip_dy:
            flip_mat = flip_mat @ torch.Tensor([[1, 0, 0], [0, -1, 0],
                                                [0, 0, 1]])
        bda_mat = flip_mat @ (scale_mat @ rot_mat)
        bda_mat = torch.inverse(bda_mat)
        results['aug_param']['GlobalRotScaleTransImage_param'] = (
        rot_angle, scale_ratio, flip_dx, flip_dy, bda_mat, self.only_gt)

        return rot_angle, scale_ratio, flip_dx, flip_dy, bda_mat, self.only_gt

    def rotate_bev_along_z(self, results, angle):
        rot_cos = np.cos(angle)
        rot_sin = np.sin(angle)

        rot_mat = np.array([[rot_cos, -rot_sin, 0, 0], [rot_sin, rot_cos, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
        rot_mat_inv = np.linalg.inv(rot_mat)

        num_view = len(results["lidar2img"])
        for view in range(num_view):
            results["lidar2img"][view] = np.matmul(results["lidar2img"][view], rot_mat_inv)
            results['lidar2cam'][view] = np.matmul(results['lidar2cam'][view], rot_mat_inv)

        return

    def scale_xyz(self, results, scale_ratio):
        scale_mat = np.array(
            [
                [scale_ratio, 0, 0, 0],
                [0, scale_ratio, 0, 0],
                [0, 0, scale_ratio, 0],
                [0, 0, 0, 1],
            ]
        )

        scale_mat_inv = np.linalg.inv(scale_mat)

        num_view = len(results["lidar2img"])
        for view in range(num_view):
            results["lidar2img"][view] = np.matmul(results["lidar2img"][view], scale_mat_inv)
            results['lidar2cam'][view] = np.matmul(results['lidar2cam'][view], scale_mat_inv)
        return

    def flip_along_x(self, results):
        flip_mat = np.array(
            [
                [-1, 0, 0, 0],
                [0, 1, 0, 0],
                [0, 0, 1, 0],
                [0, 0, 0, 1],
            ]
        ).astype(np.float32)

        flip_mat_inv = np.linalg.inv(flip_mat)

        num_view = len(results["lidar2img"])
        for view in range(num_view):
            results["lidar2img"][view] = np.matmul(results["lidar2img"][view], flip_mat_inv)
            results['lidar2cam'][view] = np.matmul(results['lidar2cam'][view], flip_mat_inv)
        return

    def flip_along_y(self, results):
        flip_mat = np.array(
            [
                [1, 0, 0, 0],
                [0, -1, 0, 0],
                [0, 0, 1, 0],
                [0, 0, 0, 1],
            ]
        ).astype(np.float32)

        flip_mat_inv = np.linalg.inv(flip_mat)

        num_view = len(results["lidar2img"])
        for view in range(num_view):
            results["lidar2img"][view] = np.matmul(results["lidar2img"][view], flip_mat_inv)
            results['lidar2cam'][view] = np.matmul(results['lidar2cam'][view], flip_mat_inv)
        return


@TRANSFORMS.register_module()
class MultiScaleFlipAug3D(object):
    """Test-time augmentation with multiple scales and flipping.

    Args:
        transforms (list[dict]): Transforms to apply in each augmentation.
        img_scale (tuple | list[tuple]: Images scales for resizing.
        pts_scale_ratio (float | list[float]): Points scale ratios for
            resizing.
        flip (bool): Whether apply flip augmentation. Defaults to False.
        flip_direction (str | list[str]): Flip augmentation directions
            for images, options are "horizontal" and "vertical".
            If flip_direction is list, multiple flip augmentations will
            be applied. It has no effect when ``flip == False``.
            Defaults to "horizontal".
        pcd_horizontal_flip (bool): Whether apply horizontal flip augmentation
            to point cloud. Defaults to True. Note that it works only when
            'flip' is turned on.
        pcd_vertical_flip (bool): Whether apply vertical flip augmentation
            to point cloud. Defaults to True. Note that it works only when
            'flip' is turned on.
    """

    def __init__(self,
                 transforms,
                 img_scale,
                 pts_scale_ratio,
                 flip=False,
                 flip_direction='horizontal',
                 pcd_horizontal_flip=False,
                 pcd_vertical_flip=False):
        self.transforms = Compose(transforms)
        self.img_scale = img_scale if isinstance(img_scale,
                                                 list) else [img_scale]
        self.pts_scale_ratio = pts_scale_ratio \
            if isinstance(pts_scale_ratio, list) else[float(pts_scale_ratio)]

        assert mmengine.is_list_of(self.img_scale, tuple)
        assert mmengine.is_list_of(self.pts_scale_ratio, float)

        self.flip = flip
        self.pcd_horizontal_flip = pcd_horizontal_flip
        self.pcd_vertical_flip = pcd_vertical_flip

        self.flip_direction = flip_direction if isinstance(
            flip_direction, list) else [flip_direction]
        assert mmengine.is_list_of(self.flip_direction, str)
        if not self.flip and self.flip_direction != ['horizontal']:
            warnings.warn(
                'flip_direction has no effect when flip is set to False')
        if (self.flip and not any([(t['type'] == 'RandomFlip3D'
                                    or t['type'] == 'RandomFlip')
                                   for t in transforms])):
            warnings.warn(
                'flip has no effect when RandomFlip is not in transforms')

    def __call__(self, results):
        """Call function to augment common fields in results.

        Args:
            results (dict): Result dict contains the data to augment.

        Returns:
            dict: The result dict contains the data that is augmented with \
                different scales and flips.
        """
        aug_data = []

        # modified from `flip_aug = [False, True] if self.flip else [False]`
        # to reduce unnecessary scenes when using double flip augmentation
        # during test time
        flip_aug = [True] if self.flip else [False]
        pcd_horizontal_flip_aug = [False, True] \
            if self.flip and self.pcd_horizontal_flip else [False]
        pcd_vertical_flip_aug = [False, True] \
            if self.flip and self.pcd_vertical_flip else [False]
        for scale in self.img_scale:
            for pts_scale_ratio in self.pts_scale_ratio:
                for flip in flip_aug:
                    for pcd_horizontal_flip in pcd_horizontal_flip_aug:
                        for pcd_vertical_flip in pcd_vertical_flip_aug:
                            for direction in self.flip_direction:
                                # results.copy will cause bug
                                # since it is shallow copy
                                _results = deepcopy(results)
                                _results['scale'] = scale
                                _results['flip'] = flip
                                _results['pcd_scale_factor'] = \
                                    pts_scale_ratio
                                _results['flip_direction'] = direction
                                _results['pcd_horizontal_flip'] = \
                                    pcd_horizontal_flip
                                _results['pcd_vertical_flip'] = \
                                    pcd_vertical_flip
                                data = self.transforms(_results)
                                aug_data.append(data)
        # list of dict to dict of list
        aug_data_dict = {key: [] for key in aug_data[0]}
        for data in aug_data:
            for key, val in data.items():
                aug_data_dict[key].append(val)
        return aug_data_dict

    def __repr__(self):
        """str: Return a string that describes the module."""
        repr_str = self.__class__.__name__
        repr_str += f'(transforms={self.transforms}, '
        repr_str += f'img_scale={self.img_scale}, flip={self.flip}, '
        repr_str += f'pts_scale_ratio={self.pts_scale_ratio}, '
        repr_str += f'flip_direction={self.flip_direction})'
        return repr_str
