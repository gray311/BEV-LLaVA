import copy
import json
import numpy as np
import mmcv
import gc
from os import path as osp
from mmengine import MODELS
import torch
import numpy as np
from nuscenes.eval.common.utils import quaternion_yaw, Quaternion
import random
from .custom_nuscenes_dataset import NuScenesDataset
from tqdm import tqdm
import random

random.seed(233)

bev_qa_instructions = [
    "Please answer the following questions related to a self-driving scenario. The views of the ego vehicle are provided by the front, front left, front right, back, back left, and back right cameras.<bev>\n{question}",
    "For the given self-driving car scenario, analyze the Bird's Eye View images. These images include perspectives from the front, rear, front-left, front-right, rear-left, and rear-right cameras of the ego vehicle. Based on these images, please provide answers to the following questions.<bev>\n{question}",
    "Using the provided Bird's Eye View (BEV) images from an autonomous vehicle's camera (front, rear, front left, front-right, rear-left, and rear-right), thoroughly answer the questions related to this ego car situation.<bev>\n{question}",
    "You are presented with a set of Bird's Eye View images capturing various cameras (front, back, front-left, front-right, back-left, back-right) of an autonomous vehicle. Please utilize these images to answer the questions about the self-driving scenario accurately.<bev>\n{question}",
    "In this task, examine the Bird's Eye View images from an ego vehicle, which include views from six different camera angles: front, rear, front-left, front-right, rear-left, and rear-right. Utilize these images to answer questions related to the autonomous driving context.<bev>\n{question}",
    "Analyze the provided BEV images from an autonomous vehicle and answer the following questions.<bev>\n{question}",
    "Review the Bird's Eye View images from an autonomous vehicle, and provide answers to the questions about this autonomous driving scenario.<bev>\n{question}",
    "Answer the autonomous driving-related question based on the given BEV images.<bev>\n{question}",
    "Analyze the provided BEV images from an autonomous vehicle and answer the following questions.<bev>\n{question}",
    "Review the Bird's Eye View images from an autonomous vehicle, and provide answers to the questions about this autonomous driving scenario.<bev>\n{question}",
    "Answer the autonomous driving-related question based on the given BEV images.<bev>\n{question}",
    "Analyze the provided BEV images from an autonomous vehicle and answer the following questions.<bev>\n{question}",
    "Review the Bird's Eye View images from an autonomous vehicle, and provide answers to the questions about this autonomous driving scenario.<bev>\n{question}",
    "Answer the autonomous driving-related question based on the given BEV images.<bev>\n{question}",
]

point_qa_instructions = [
    "Please indicate object locations with [cx,cy].",
    "You can indicate object locations with [cx,cy] format.",
    "Remember to label the center points of objects involved in the form of [cx,cy].",
    "Include the center points of any relevant objects in the image [cx,cy].",
    "Also, mark the center points of objects within the image [cx,cy].",
    "Please label any involved objects' center points [cx,cy].",
    "Indicate any mentioned objects' center points [cx,cy].",
    "Label the center points of relevant objects if necessary [cx,cy].",
    "Mark the center points of any related objects as [cx,cy]."
]


@MODELS.register_module()
class CustomNuScenesDataset(NuScenesDataset):
    r"""NuScenes Dataset.

    This datset only add camera intrinsics and extrinsics to the results.
    """

    def __init__(self, queue_length=4, bev_size=(200, 200), overlap_test=False, nuscenes_qa_file=None, drivelm_qa_file=None, dataset_length=None, *args, **kwargs):
        self.samples = []
        if nuscenes_qa_file is not None:
            self.nuscenes_qa = json.load(open(nuscenes_qa_file, "r"))['questions']
            self.samples.extend(self.nuscenes_qa)

        if drivelm_qa_file is not None:
            self.drivelm_qa = json.load(open(drivelm_qa_file, "r"))
            self.samples.extend(self.drivelm_qa)

        if dataset_length is not None:
            self.samples = self.samples[:dataset_length]

        super().__init__(*args, **kwargs)
        self.queue_length = queue_length
        self.overlap_test = overlap_test
        self.bev_size = bev_size
        self.token2index = {}
        self.index2image = {}
        for i in tqdm(range(len(self.data_infos))):
            self.token2index[self.data_infos[i]['token']] =  i




    def __len__(self):
        return len(self.samples)

    def prepare_test_data(self, idx):
        """
        Training data preparation.
        Args:
            index (int): Index for accessing the target data.
        Returns:
            dict: Training data dict of the corresponding index.
        """
        sample = self.samples[idx]
        if 'version' in sample.keys():
            if sample['version'] == 'qa':
                sample['instruction'] = bev_qa_instructions[0] + " " + point_qa_instructions[0]
            else:
                sample['instruction'] = bev_qa_instructions[0]
        else:
            sample['instruction'] = bev_qa_instructions[0]
        index = self.token2index[sample['sample_token']]
        queue = []
        index_list = list(range(index-self.queue_length + 1, index))
        index_list.append(index)
        for i in index_list:
            i = max(0, i)
            input_dict = self.get_data_info(i)
            if input_dict is None:
                return None
            if i in self.index2infos.keys():
                example = copy.deepcopy(self.index2infos[i])
            else:
                self.pre_pipeline(input_dict)
                example = self.pipeline(input_dict)
                example['img'] = example['img'].to(torch.float16)
                self.index2infos[i] = copy.deepcopy(example)
            # example = {key: value[0] for key, value in example.items()}
            queue.append(example)

        return self.union2one(queue, **sample)

    def prepare_train_data(self, idx):
        """
        Training data preparation.
        Args:
            index (int): Index for accessing the target data.
        Returns:
            dict: Training data dict of the corresponding index.
        """
        sample = self.samples[idx]
        if 'version' in sample.keys():
            if sample['version'] == 'qa':
                sample['instruction'] = random.sample(bev_qa_instructions, 1)[0] + " " + random.sample(point_qa_instructions, 1)[0]
            else:
                sample['instruction'] = random.sample(bev_qa_instructions, 1)[0]
        else:
            sample['instruction'] = random.sample(bev_qa_instructions, 1)[0]
        index = self.token2index[sample['sample_token']]
        queue = []
        index_list = list(range(index - self.queue_length, index))
        random.shuffle(index_list)
        index_list = sorted(index_list[1:])
        index_list.append(index)
        for i in index_list:
            i = max(0, i)
            input_dict = self.get_data_info(i)
            if input_dict is None:
                return None
            # if i in self.index2image.keys():
            #     input_dict['img'] = self.index2image[i].astype(np.float32)
            self.pre_pipeline(input_dict)
            example = self.pipeline(input_dict)
            # example = {key: value[0] for key, value in example.items()}
            queue.append(example)

        return self.union2one(queue, **sample)


    def union2one(self, queue, **kwargs):
        imgs_list = [each['img'].data for each in queue]
        metas_map = {}
        prev_scene_token = None
        prev_pos = None
        prev_angle = None
        for i, each in enumerate(queue):
            metas_map[i] = each['img_metas']
            if metas_map[i]['scene_token'] != prev_scene_token:
                metas_map[i]['prev_bev_exists'] = False
                prev_scene_token = metas_map[i]['scene_token']
                prev_pos = copy.deepcopy(metas_map[i]['can_bus'][:3])
                prev_angle = copy.deepcopy(metas_map[i]['can_bus'][-1])
                metas_map[i]['can_bus'][:3] = 0
                metas_map[i]['can_bus'][-1] = 0
            else:
                metas_map[i]['prev_bev_exists'] = True
                tmp_pos = copy.deepcopy(metas_map[i]['can_bus'][:3])
                tmp_angle = copy.deepcopy(metas_map[i]['can_bus'][-1])
                metas_map[i]['can_bus'][:3] -= prev_pos
                metas_map[i]['can_bus'][-1] -= prev_angle
                prev_pos = copy.deepcopy(tmp_pos)
                prev_angle = copy.deepcopy(tmp_angle)
        queue[-1]['img'] = torch.stack(imgs_list)
        queue[-1]['img_metas'] = metas_map
        queue[-1].update(
            instruction=kwargs['instruction'],
            question=kwargs['question'],
            answer=kwargs['answer'],
            sample_token=kwargs['sample_token'],
        )
        queue = queue[-1]

        return queue

    def get_data_info(self, index):
        """Get data info according to the given index.

        Args:
            index (int): Index of the sample data to get.

        Returns:
            dict: Data information that will be passed to the data \
                preprocessing pipelines. It includes the following keys:

                - sample_idx (str): Sample index.
                - pts_filename (str): Filename of point clouds.
                - sweeps (list[dict]): Infos of sweeps.
                - timestamp (float): Sample timestamp.
                - img_filename (str, optional): Image filename.
                - lidar2img (list[np.ndarray], optional): Transformations \
                    from lidar to different cameras.
                - ann_info (dict): Annotation info.
        """
        info = self.data_infos[index]
        # standard protocal modified from SECOND.Pytorch
        input_dict = dict(
            sample_idx=info['token'],
            pts_filename=info['lidar_path'],
            sweeps=info['sweeps'],
            ego2global_translation=info['ego2global_translation'],
            ego2global_rotation=info['ego2global_rotation'],
            prev_idx=info['prev'],
            next_idx=info['next'],
            scene_token=info['scene_token'],
            can_bus=info['can_bus'],
            frame_idx=info['frame_idx'],
            timestamp=info['timestamp'] / 1e6,
        )

        if self.modality['use_camera']:
            image_paths = []
            lidar2img_rts = []
            lidar2cam_rts = []
            cam_intrinsics = []
            for cam_type, cam_info in info['cams'].items():
                image_paths.append(cam_info['data_path'])
                # obtain lidar to image transformation matrix
                lidar2cam_r = np.linalg.inv(cam_info['sensor2lidar_rotation'])
                lidar2cam_t = cam_info[
                    'sensor2lidar_translation'] @ lidar2cam_r.T
                lidar2cam_rt = np.eye(4)
                lidar2cam_rt[:3, :3] = lidar2cam_r.T
                lidar2cam_rt[3, :3] = -lidar2cam_t
                intrinsic = cam_info['cam_intrinsic']
                viewpad = np.eye(4)
                viewpad[:intrinsic.shape[0], :intrinsic.shape[1]] = intrinsic
                lidar2img_rt = (viewpad @ lidar2cam_rt.T)
                lidar2img_rts.append(lidar2img_rt)

                cam_intrinsics.append(viewpad)
                lidar2cam_rts.append(lidar2cam_rt.T)

            input_dict.update(
                dict(
                    img_filename=image_paths,
                    lidar2img=lidar2img_rts,
                    cam_intrinsic=cam_intrinsics,
                    lidar2cam=lidar2cam_rts,
                ))

        if not self.test_mode:
            annos = self.get_ann_info(index)
            input_dict['ann_info'] = annos

        rotation = Quaternion(input_dict['ego2global_rotation'])
        translation = input_dict['ego2global_translation']
        can_bus = input_dict['can_bus']
        can_bus[:3] = translation
        can_bus[3:7] = rotation
        patch_angle = quaternion_yaw(rotation) / np.pi * 180
        if patch_angle < 0:
            patch_angle += 360
        can_bus[-2] = patch_angle / 180 * np.pi
        can_bus[-1] = patch_angle

        return input_dict

    def __getitem__(self, idx):
        """Get item from infos according to the given index.
        Returns:
            dict: Data dictionary of the corresponding index.
        """
        if self.test_mode:
            sample = self.prepare_test_data(idx)
            return sample
        while True:
            sample = self.prepare_train_data(idx)
            if sample is None:
                idx = self._rand_another(idx)
                continue
            return sample