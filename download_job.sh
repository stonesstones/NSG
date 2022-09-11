#!/bin/bash

#$-l rt_C.small=1
#$-l h_rt=10:00:00
#$-j y
#$ -o ./log4txt
#$-cwd

source /etc/profile.d/modules.sh
bash download_kitti.sh