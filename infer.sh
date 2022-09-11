#!/bin/bash

#$-l rt_G.small=1
#$-l h_rt=1:00:00
#$-j y
#$ -o ./log4txt
#$-cwd

source /etc/profile.d/modules.sh
module load singularitypro 
# bash download_weights_kitti.sh
SINGULARITY_TMPDIR=$SGE_LOCALDIR singularity exec --nv \
--bind /groups/gcd50654/tier4/dataset/kitti:/groups/gcd50654/tier4/dataset/kitti \
--bind /groups/gcd50654/tier4/neural-scene-graphs:/groups/gcd50654/tier4/neural-scene-graphs \
/groups/gcd50654/tier4/neural-scene-graphs/nerf.sif \
bash render_nerf.sh