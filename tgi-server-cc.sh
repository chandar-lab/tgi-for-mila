#!/bin/bash
#SBATCH --account=rrg-bengioy-ad
#SBATCH -J tgi-server
#SBATCH --output=%x.%j.out
#SBATCH --cpus-per-task=6
#SBATCH --gpus-per-task=1
#SBATCH --ntasks=1
#SBATCH --mem=24G
#SBATCH --time=2:59:00
set -e

# Load modules
module load python/3.11 gcc/11.3.0 git-lfs/3.3.0 rust/1.65.0 protobuf/3.21.3 cuda/11.8.0 cudnn/8.6.0.163

# create env
virtualenv --app-data $SCRATCH/virtualenv --no-download $SLURM_TMPDIR/tgl-env
source $SLURM_TMPDIR/tgl-env/bin/activate
python -m pip install --no-index -U pip setuptools wheel build

# install
pip install --no-index --find-links $RELEASE_DIR/python_deps \
  'accelerate<0.20.0,>=0.19.0' \
  'einops<0.7.0,>=0.6.1' \
  $RELEASE_DIR/python_ins/*.whl
export PATH="$(realpath $RELEASE_DIR/bin/)":$PATH

# configure
export HUGGINGFACE_OFFLINE=1
export HF_HUB_DISABLE_TELEMETRY=1
export HF_HUB_ENABLE_HF_TRANSFER=1
export HUGGINGFACE_HUB_CACHE=$TGI_DIR/tgi-data

export NUM_GPUS=$(python -c 'import torch; print(torch.cuda.device_count())')
export TGI_PORT=$(expr 10000 + $(echo -n $SLURM_JOBID | tail -c 4))
export TGI_ADDR="0.0.0.0"
export MASTER_PORT=$(expr 20000 + $(echo -n $SLURM_JOBID | tail -c 4))
export MASTER_ADDR="127.0.0.1"

# start
text-generation-launcher --model-id $TGI_DIR/tgi-repos/$MODEL_ID --num-shard $NUM_GPUS \
  --port $TGI_PORT --hostname $TGI_ADDR \
  --master-port $MASTER_PORT --master-addr $MASTER_ADDR \
  --shard-uds-path $SLURM_TMPDIR/tgl-server-socket
  # --max-best-of $MAX_BEST_OF --max-total-tokens $MAX_STOP_SEQUENCES
  # --max-input-length $MAX_INPUT_LENGTH --max-stop-sequences $MAX_TOTAL_TOKENS --quantize $QUANTIZE
