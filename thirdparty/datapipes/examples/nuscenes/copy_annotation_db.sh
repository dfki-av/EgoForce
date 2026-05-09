#!/bin/bash

cd $HOME/Desktop/nuScenes_micro_tars || exit
mkdir v1.0-mini
cd v1.0-mini || exit
cp /ds-av/public_datasets/nuscenes/raw/v1.0-mini/attribute.json attribute.json
cp /ds-av/public_datasets/nuscenes/raw/v1.0-mini/calibrated_sensor.json calibrated_sensor.json
cp /ds-av/public_datasets/nuscenes/raw/v1.0-mini/category.json category.json
cp /ds-av/public_datasets/nuscenes/raw/v1.0-mini/ego_pose.json ego_pose.json
cp /ds-av/public_datasets/nuscenes/raw/v1.0-mini/instance.json instance.json
cp /ds-av/public_datasets/nuscenes/raw/v1.0-mini/lidarseg.json lidarseg.json
cp /ds-av/public_datasets/nuscenes/raw/v1.0-mini/log.json log.json
cp /ds-av/public_datasets/nuscenes/raw/v1.0-mini/map.json map.json
cp /ds-av/public_datasets/nuscenes/raw/v1.0-mini/panoptic.json panoptic.json
cp /ds-av/public_datasets/nuscenes/raw/v1.0-mini/sample.json sample.json
cp /ds-av/public_datasets/nuscenes/raw/v1.0-mini/sample_annotation.json sample_annotation.json
cp /ds-av/public_datasets/nuscenes/raw/v1.0-mini/sample_data.json sample_data.json
cp /ds-av/public_datasets/nuscenes/raw/v1.0-mini/scene.json scene.json
cp /ds-av/public_datasets/nuscenes/raw/v1.0-mini/sensor.json sensor.json
cp /ds-av/public_datasets/nuscenes/raw/v1.0-mini/visibility.json visibility.json
cd - || exit

cd $HOME/Desktop/nuScenes_tars || exit
mkdir v1.0-trainval
cd v1.0-trainval || exit
cp /ds-av/public_datasets/nuscenes/raw/v1.0-trainval/attribute.json attribute.json
cp /ds-av/public_datasets/nuscenes/raw/v1.0-trainval/calibrated_sensor.json calibrated_sensor.json
cp /ds-av/public_datasets/nuscenes/raw/v1.0-trainval/category.json category.json
cp /ds-av/public_datasets/nuscenes/raw/v1.0-trainval/ego_pose.json ego_pose.json
cp /ds-av/public_datasets/nuscenes/raw/v1.0-trainval/instance.json instance.json
cp /ds-av/public_datasets/nuscenes/raw/v1.0-trainval/lidarseg.json lidarseg.json
cp /ds-av/public_datasets/nuscenes/raw/v1.0-trainval/log.json log.json
cp /ds-av/public_datasets/nuscenes/raw/v1.0-trainval/map.json map.json
cp /ds-av/public_datasets/nuscenes/raw/v1.0-trainval/panoptic.json panoptic.json
cp /ds-av/public_datasets/nuscenes/raw/v1.0-trainval/sample.json sample.json
cp /ds-av/public_datasets/nuscenes/raw/v1.0-trainval/sample_annotation.json sample_annotation.json
cp /ds-av/public_datasets/nuscenes/raw/v1.0-trainval/sample_data.json sample_data.json
cp /ds-av/public_datasets/nuscenes/raw/v1.0-trainval/scene.json scene.json
cp /ds-av/public_datasets/nuscenes/raw/v1.0-trainval/sensor.json sensor.json
cp /ds-av/public_datasets/nuscenes/raw/v1.0-trainval/visibility.json visibility.json
cd - || exit

mkdir v1.0-test
cd v1.0-test || exit
cp /ds-av/public_datasets/nuscenes/raw/v1.0-test/attribute.json attribute.json
cp /ds-av/public_datasets/nuscenes/raw/v1.0-test/calibrated_sensor.json calibrated_sensor.json
cp /ds-av/public_datasets/nuscenes/raw/v1.0-test/category.json category.json
cp /ds-av/public_datasets/nuscenes/raw/v1.0-test/ego_pose.json ego_pose.json
cp /ds-av/public_datasets/nuscenes/raw/v1.0-test/instance.json instance.json
cp /ds-av/public_datasets/nuscenes/raw/v1.0-test/lidarseg.json lidarseg.json
cp /ds-av/public_datasets/nuscenes/raw/v1.0-test/log.json log.json
cp /ds-av/public_datasets/nuscenes/raw/v1.0-test/map.json map.json
cp /ds-av/public_datasets/nuscenes/raw/v1.0-test/panoptic.json panoptic.json
cp /ds-av/public_datasets/nuscenes/raw/v1.0-test/sample.json sample.json
cp /ds-av/public_datasets/nuscenes/raw/v1.0-test/sample_annotation.json sample_annotation.json
cp /ds-av/public_datasets/nuscenes/raw/v1.0-test/sample_data.json sample_data.json
cp /ds-av/public_datasets/nuscenes/raw/v1.0-test/scene.json scene.json
cp /ds-av/public_datasets/nuscenes/raw/v1.0-test/sensor.json sensor.json
cp /ds-av/public_datasets/nuscenes/raw/v1.0-test/visibility.json visibility.json
cd - || exit
