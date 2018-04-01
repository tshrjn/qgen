#!/bin/sh
#SBATCH --verbose
#SBATCH --job-name=QgenBase
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=40GB
#SBATCH --time=10:0:00
##SBATCH --exclusive
#SBATCH --output=slurm_qgen_%j.out
##SBATCH --partition=k80_4
#SBATCH --gres=gpu:1
#Uncomment to execute python code
export LC_ALL='en_US.utf8'
module purge
module load anaconda3/4.3.1
source ~/.conda/envs/qgen/bin/activate
python /home/ra2630/QGen/qgen.py | tee log.log
