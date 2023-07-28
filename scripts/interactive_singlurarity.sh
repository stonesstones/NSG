#!/bin/bash

#$-l rt_G.small=1
#$-l h_rt=1:00:00
#$-j y
#$ -o ./log4txt
#$-cwd

source /etc/profile.d/modules.sh
module load singularitypro 
# bash download_weights_kitti.sh
SINGULARITY_TMPDIR=$SGE_LOCALDIR singularity shell --nv \
--bind /groups/gcd50654/tier4/dataset/kitti:/groups/gcd50654/tier4/dataset/kitti \
--bind /groups/gcd50654/tier4/neural-scene-graphs:/groups/gcd50654/tier4/neural-scene-graphs \
--bind /home/acf15379bv/NSG:/home/acf15379bv/NSG \
/home/acf15379bv/NSG/docker/nerfstudio2.sif \