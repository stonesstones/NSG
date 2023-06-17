#!/bin/bash

DATADIR="/groups/gcd50654/tier4/dataset/vkitti2/"
mkdir $DATADIR
wget -c http://download.europe.naverlabs.com//virtual_kitti_2.0.3/vkitti_2.0.3_rgb.tar -O - | tar xf - -C $DATADIR
wget -c http://download.europe.naverlabs.com//virtual_kitti_2.0.3/vkitti_2.0.3_textgt.tar.gz -O - | tar xf - -C $DATADIR
