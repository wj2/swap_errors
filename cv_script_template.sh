#!/bin/bash
#SBATCH --job-name=mediumjob
#SBATCH --account=theory
#SBATCH -c 8
#SBATCH --time=0-11:59:00
#SBATCH --mem-per-cpu=6gb
#SBATCH --array=0-{n_tot}%50
module load anaconda/3-2020.11
source activate pete
python {file_dir}/run_comparison.py $SLURM_ARRAY_TASK_ID {dset_idx}
# python -c "print('success')"