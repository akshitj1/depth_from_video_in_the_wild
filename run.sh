# Copyright 2020 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

#!/bin/bash
set -e
set -x

# source env/bin/activate
# euroc
DATASET=euroc
CKPT_DIR=checkpoints/depth_from_video_in_the_wild_euroc_ckpt_MachineHallAll
# vicon room
#IN_IMAGE=/Volumes/external/workspace/datasets/V2_01_easy/mav0/cam0/data/1413393300955760384.png
# machine hall
IN_IMAGE=/Volumes/external/workspace/datasets/MH_01_easy/1403636737313555456.png

# kitti
# CKPT_DIR=checkpoints/cityscapes_kitti_learned_intrinsics
# IN_IMAGE=/Users/akshitjain/ext/workspace/datasets/kitti_2012/2011_09_26/2011_09_26_drive_0035_sync/image_02/data/0000000000.png
# DATASET=kitti

# kitti
python -m depth_inference \
  --dataset=$DATASET \
  --input_image_path=$IN_IMAGE \
  --checkpoint_dir=$CKPT_DIR \
  --depth_image_dir=data