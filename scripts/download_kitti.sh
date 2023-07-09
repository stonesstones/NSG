#!/bin/bash
# LEFT="https://s3.eu-central-1.amazonaws.com/avg-kitti/data_tracking_image_2.zip"
RIGHT="https://s3.eu-central-1.amazonaws.com/avg-kitti/data_tracking_image_3.zip"
# IMU="https://s3.eu-central-1.amazonaws.com/avg-kitti/data_tracking_oxts.zip"
# CALIB="https://s3.eu-central-1.amazonaws.com/avg-kitti/data_tracking_calib.zip"
# LABEL="https://s3.eu-central-1.amazonaws.com/avg-kitti/data_tracking_label_2.zip"
DATADIR="/groups/gcd50654/tier4/dataset/kitti"
TEMPFILE="/groups/gcd50654/tier4/dataset/kitti/tmp.zip"

# wget $LEFT -O $TEMPFILE && unzip $TEMPFILE -d $DATADIR && rm -f $TEMPFILE
wget $RIGHT -O $TEMPFILE && unzip $TEMPFILE -d $DATADIR && rm -f $TEMPFILE
# wget $IMU -O $TEMPFILE && unzip $TEMPFILE -d $DATADIR && rm -f $TEMPFILE
# wget $CALIB -O $TEMPFILE && unzip $TEMPFILE -d $DATADIR && rm -f $TEMPFILE
# wget $LABEL -O $TEMPFILE && unzip $TEMPFILE -d $DATADIR && rm -f $TEMPFILE