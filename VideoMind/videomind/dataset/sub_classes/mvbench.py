# Copyright (c) 2025 Ye Liu. Licensed under the BSD-3-Clause License.

import nncore
from torch.utils.data import Dataset

from videomind.dataset.hybrid import DATASETS
from videomind.utils.parser import parse_query, parse_question


@DATASETS.register(name='mvbench')
class MVBenchDataset(Dataset):

    META_DATA = [('Episodic Reasoning', 'episodic_reasoning.json', 'tvqa/frames_fps3_hq', 'frame'),
                 ('Action Sequence', 'action_sequence.json', 'star/Charades_v1_480', 'video'),
                 ('Action Prediction', 'action_prediction.json', 'star/Charades_v1_480', 'video'),
                 ('Action Antonym', 'action_antonym.json', 'ssv2_video', 'video'),
                 ('Fine-grained Action', 'fine_grained_action.json', 'Moments_in_Time_Raw/videos', 'video'),
                 ('Unexpected Action', 'unexpected_action.json', 'FunQA_test/test', 'video'),
                 ('Object Existence', 'object_existence.json', 'clevrer/video_validation', 'video'),
                 ('Object Interaction', 'object_interaction.json', 'star/Charades_v1_480', 'video'),
                 ('Object Shuffle', 'object_shuffle.json', 'perception/videos', 'video'),
                 ('Moving Direction', 'moving_direction.json', 'clevrer/video_validation', 'video'),
                 ('Action Localization', 'action_localization.json', 'sta/sta_video', 'video'),
                 ('Scene Transition', 'scene_transition.json', 'scene_qa/video', 'video'),
                 ('Action Count', 'action_count.json', 'perception/videos', 'video'),
                 ('Moving Count', 'moving_count.json', 'clevrer/video_validation', 'video'),
                 ('Moving Attribute', 'moving_attribute.json', 'clevrer/video_validation', 'video'),
                 ('State Change', 'state_change.json', 'perception/videos', 'video'),
                 ('Fine-grained Pose', 'fine_grained_pose.json', 'nturgbd', 'video'),
                 ('Character Order', 'character_order.json', 'perception/videos', 'video'),
                 ('Egocentric Navigation', 'egocentric_navigation.json', 'vlnqa', 'video'),
                 ('Counterfactual Inference', 'counterfactual_inference.json', 'clevrer/video_validation', 'video')]

    DATA_ROOT = 'data/mvbench'

    MIN_LEN = 64

    @classmethod
    def load_annos(self, split='test', sample_frames=32):
        assert split == 'test'

        annos = []
        for meta in self.META_DATA:
            raw_annos = nncore.load(nncore.join(self.DATA_ROOT, 'json', meta[1]))

            for raw_anno in raw_annos:
                video_name = nncore.join(meta[2], raw_anno['video'])
                video_path = nncore.join(self.DATA_ROOT, 'video', video_name)

                if meta[3] == 'frame':
                    num_frames = len(nncore.ls(video_path, ext='.jpg'))
                    video_path = [
                        nncore.join(video_path, f'{i:0>5}.jpg')
                        for i in range(1, num_frames + 1, num_frames // (sample_frames - 1))
                    ][:sample_frames]

                options = raw_anno['candidates']
                answer = raw_anno['answer']
                ans = chr(ord('A') + options.index(answer))

                anno = dict(
                    source='mvbench',
                    data_type='multimodal',
                    video_path=video_path,
                    query=parse_query(raw_anno['question']),
                    question=parse_question(raw_anno['question']),
                    options=options,
                    answer=answer,
                    ans=ans,
                    task=meta[0])

                annos.append(anno)

        return annos
