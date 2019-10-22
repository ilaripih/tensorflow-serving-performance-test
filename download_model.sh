#!/bin/bash

set -x
set -e

mkdir -p models
cd models
wget http://download.tensorflow.org/models/object_detection/ssd_resnet50_v1_fpn_shared_box_predictor_640x640_coco14_sync_2018_07_03.tar.gz
tar xvzf ssd_resnet50_v1_fpn_shared_box_predictor_640x640_coco14_sync_2018_07_03.tar.gz
rm ssd_resnet50_v1_fpn_shared_box_predictor_640x640_coco14_sync_2018_07_03.tar.gz
mkdir -p retinanet/1
cp -avr ./ssd_resnet50_v1_fpn_shared_box_predictor_640x640_coco14_sync_2018_07_03/saved_model/* ./retinanet/1
cd ..