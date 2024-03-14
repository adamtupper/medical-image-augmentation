#!/bin/bash
#SBATCH --array=1-5
#SBATCH --mem=32G
#SBATCH --nodes=1
#SBATCH --cpus-per-task=10
#SBATCH --gres=gpu:1
#SBATCH --time=06:00:00
#SBATCH --mail-type=ALL

export TORCH_HOME=$project

# Check for dataset
if [ -z "1" ]; then
    echo "No dataset supplied"
    exit 1
fi

# Check for augmentation set
if [ -z "2" ]; then
    echo "No augmentation set supplied"
    exit 1
fi

SAVE_DIR=$scratch/miccai2024/trivial_augment

# Print Job info
echo "Current working directory: `pwd`"
echo "Starting run at: `date`"
echo ""
echo "Job ID: $SLURM_JOB_ID"
echo ""

module purge

# Copy data to compute node
mkdir -p $SLURM_TMPDIR/data/
tar -xf $project/data/miccai24/$1.tar.gz -C $SLURM_TMPDIR/data

# Copy code to compute node
cp -r $project/BUDA $SLURM_TMPDIR

# Change to code directory
cd $SLURM_TMPDIR/BUDA

# Create virtual environment
module load python/3.11 cuda cudnn rust
virtualenv --no-download $SLURM_TMPDIR/env
source $SLURM_TMPDIR/env/bin/activate
pip install --no-index --upgrade pip

# Install package and dependencies
pip install --no-index -r cc_requirements.txt

# Run a single trial
python trivial_augment.py \
    --data_dir $SLURM_TMPDIR/data \
    --config_dir config \
    --log_dir $SAVE_DIR \
    --workers 9 \
    --seed 0 \
    --dataset $1 \
    --subset $2 \
    