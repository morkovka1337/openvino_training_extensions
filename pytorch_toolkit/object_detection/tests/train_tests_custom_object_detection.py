# Copyright (C) 2020 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions
# and limitations under the License.

import json
import os
import unittest

import torch
import yaml

from ote.tests.test_case import (
    skip_if_cpu_is_not_supported,
    skip_non_instantiated_template_if_its_allowed,
    get_dependencies,
    download_snapshot_if_not_yet
)

from ote.tests.utils import (
    collect_ap,
    run_through_shell
)

def create_custom_object_detection_test_case(model_name):

    problem_name = 'custom-object-detection'
    domain_name = 'object_detection'
    metric_keys = ['bbox']

    ann_file=os.path.dirname(__file__) + '/../../../data/airport/annotation_example_train.json'
    img_root=os.path.dirname(__file__) + '/../../../data/airport/train'

    class TestCase(unittest.TestCase):

        domain = domain_name
        problem = problem_name
        model = model_name
        metrics = metric_keys
        topic = 'train'

        @classmethod
        def setUpClass(cls):
            cls.templates_folder = os.environ['MODEL_TEMPLATES']
            cls.template_folder = os.path.join(cls.templates_folder, domain_name, problem_name, model_name)
            skip_non_instantiated_template_if_its_allowed(cls.template_folder, problem_name, model_name)
            cls.template_file = os.path.join(cls.template_folder, 'template.yaml')
            cls.ann_file = ann_file
            cls.img_root = img_root
            cls.dependencies = get_dependencies(cls.template_file)
            cls.total_epochs = 1

            download_snapshot_if_not_yet(cls.template_file, cls.template_folder)

            run_through_shell(
                f'cd {cls.template_folder};'
                f'pip install -r requirements.txt;'
            )

        def setUp(self):
            self.output_folder = os.path.join(self.template_folder, f'output_{self.id()}')
            os.makedirs(self.output_folder, exist_ok=True)

        def do_finetuning(self, with_classes, on_gpu):
            log_file = os.path.join(self.output_folder, 'test_finetuning.log')
            initial_command = 'export CUDA_VISIBLE_DEVICES=;' if not on_gpu else ''
            classes = ''
            if with_classes:
                classes = 'person,vehicle,non-vehicle'
            run_through_shell(
                f'{initial_command}'
                f'cd {self.template_folder};'
                f'python train.py'
                f' --train-ann-files {self.ann_file}'
                f' --train-data-roots {self.img_root}'
                f' --val-ann-files {self.ann_file}'
                f' --val-data-roots {self.img_root}'
                f' --resume-from snapshot.pth'
                f' --save-checkpoints-to {self.output_folder}'
                f' --gpu-num 1'
                f' --batch-size 1'
                f' --classes "{classes}"'
                f' --epochs {self.total_epochs}'
                f' | tee {log_file}')

            self.assertTrue(os.path.exists(os.path.join(self.output_folder, 'latest.pth')))

        def do_evaluation(self, with_classes, on_gpu):
            initial_command = 'export CUDA_VISIBLE_DEVICES=;' if not on_gpu else ''
            metrics_path = os.path.join(self.output_folder, "metrics.yaml")
            classes = ''
            if with_classes:
                classes = 'person,vehicle,non-vehicle'
            run_through_shell(
                f'{initial_command}'
                f'cd {self.template_folder};'
                f'python eval.py'
                f' --test-ann-files {self.ann_file}'
                f' --test-data-roots {self.img_root}'
                f' --save-metrics-to {metrics_path}'
                f' --classes "{classes}"'
                f' --load-weights {os.path.join(self.output_folder, "latest.pth")}'
            )

            with open(metrics_path) as read_file:
                content = yaml.safe_load(read_file)

            for metric_key in metric_keys:
                value = [metrics['value'] for metrics in content['metrics'] if metrics['key'] == metric_key][0]
                self.assertGreaterEqual(value, 0.0)

        def do_export(self, with_classes, on_gpu):
            initial_command = 'export CUDA_VISIBLE_DEVICES=;' if not on_gpu else ''
            classes = ''
            if with_classes:
                classes = 'person,vehicle,non-vehicle'
            run_through_shell(
                f'{initial_command}'
                f'cd {os.path.dirname(self.template_file)};'
                f'pip install -r requirements.txt;'
                f'python export.py'
                f' --load-weights {os.path.join(self.output_folder, "latest.pth")}'
                f' --classes "{classes}"'
                f' --save-model-to {self.output_folder}'
            )

        def do_evaluation_of_exported_model(self, with_classes):
            metrics_path = os.path.join(self.output_folder, "metrics_exported.yaml")
            classes = ''
            if with_classes:
                classes = 'person,vehicle,non-vehicle'
            run_through_shell(
                f'cd {os.path.dirname(self.template_file)};'
                f'python eval.py'
                f' --test-ann-files {self.ann_file}'
                f' --test-data-roots {self.img_root}'
                f' --load-weights {os.path.join(self.output_folder, "model.bin")}'
                f' --classes "{classes}"'
                f' --save-metrics-to {metrics_path}'
            )

            with open(metrics_path) as read_file:
                content = yaml.safe_load(read_file)

            for metric_key in self.metrics:
                value = [metrics['value'] for metrics in content['metrics'] if metrics['key'] == metric_key][0]
                self.assertGreaterEqual(value, 0.0)

        @unittest.skipUnless(torch.cuda.is_available(), 'No GPU found')
        def test_e2e_on_gpu(self):
            skip_if_cuda_not_available()
            self.do_finetuning(with_classes=False, on_gpu=True)
            self.do_evaluation(with_classes=False, on_gpu=True)
            self.do_export(with_classes=False, on_gpu=True)
            self.do_evaluation_of_exported_model(with_classes=False)

        def test_e2e_on_cpu_with_classes(self):
            skip_if_cpu_is_not_supported(self.template_file)
            self.do_finetuning(with_classes=True, on_gpu=False)
            self.do_evaluation(with_classes=True, on_gpu=False)
            self.do_export(with_classes=True, on_gpu=False)
            self.do_evaluation_of_exported_model(with_classes=True)

    return TestCase


class MobilenetV2_2S_SSD_256x256_TestCase(
        create_custom_object_detection_test_case(
            model_name='mobilenet_v2-2s_ssd-256x256',
        )
):
    """ Test case for mobilenet_v2-2s_ssd-256x256 model. """


class MobilenetV2_2S_SSD_384x384_TestCase(
        create_custom_object_detection_test_case(
            model_name='mobilenet_v2-2s_ssd-384x384',
        )
):
    """ Test case for mobilenet_v2-2s_ssd-384x384 model. """


class MobilenetV2_2S_SSD_512x512_TestCase(
        create_custom_object_detection_test_case(
            model_name='mobilenet_v2-2s_ssd-512x512',
        )
):
    """ Test case for mobilenet_v2-2s_ssd-512x512 model. """
