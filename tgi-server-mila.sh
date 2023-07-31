#!/bin/bash
#SBATCH -J tgi-server
#SBATCH --output=%x.%j.out
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-task=1
#SBATCH --ntasks=1
#SBATCH --constraint=ampere
#SBATCH --mem=24G
#SBATCH --time=2:59:00
set -e

# Create enviorment
eval "$(~/bin/micromamba shell hook -s posix)"
micromamba create -y -p $SLURM_TMPDIR/tgl-env -c pytorch -c nvidia -c conda-forge 'python=3.11' 'git-lfs=3.3' 'pytorch==2.0.1' 'pytorch-cuda=11.7' 'cudnn=8.8' 'openssl=3'
micromamba activate $SLURM_TMPDIR/tgl-env

# install
pip install --no-index --find-links $RELEASE_DIR/python_deps \
  'accelerate<0.20.0,>=0.19.0' \
  'einops<0.7.0,>=0.6.1' \
  $RELEASE_DIR/python_ins/*.whl
export PATH="$(realpath $RELEASE_DIR/bin/)":$PATH
export LD_LIBRARY_PATH=$SLURM_TMPDIR/tgl-env/lib:$LD_LIBRARY_PATH

# configure
# These extentions require CUDA 11.8 and were not included in the build for Mila
export DISABLE_EXLLAMA='true'
export DISABLE_CUSTOM_KERNELS='true'

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
