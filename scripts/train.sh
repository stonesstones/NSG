#!/bin/bash

#$-l rt_G.small=1
#$-l h_rt=8:00:00
#$-j y
#$ -o ./log4txt
#$-cwd

source /etc/profile.d/modules.sh
module load singularitypro 
SINGULARITY_TMPDIR=$SGE_LOCALDIR singularity exec --nv \
--bind /groups/gcd50654/tier4/dataset/kitti:/groups/gcd50654/tier4/dataset/kitti \
--bind /groups/gcd50654/tier4/neural-scene-graphs:/groups/gcd50654/tier4/neural-scene-graphs \
--bind /home/acf15379bv/NSG:/home/acf15379bv/NSG \
/home/acf15379bv/NSG/docker/nerfstudio2.sif \
bash scripts/train_nerf.sh