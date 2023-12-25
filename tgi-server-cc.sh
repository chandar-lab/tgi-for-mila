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

TGI_VERSION='1.3.4'
FLASH_ATTN_VERSION='2.4.1'

# Default config
if [ -z "${RELEASE_DIR}" ]; then
    RELEASE_DIR=$HOME/tgi-release
fi
if [ -z "${TGI_DIR}" ]; then
    TGI_DIR=$SCRATCH/tgi
fi
if [ -z "${TGI_TMP}" ]; then
    TGI_TMP=$SLURM_TMPDIR/tgi
fi

# Load modules
module load StdEnv/2023
module load python/3.11 git-lfs/3.4.0 protobuf/24.4 arrow/14.0.1 cudacore/.12.2.2 cudnn/8.9.5.29

# create env
virtualenv --app-data $SCRATCH/virtualenv --no-download $TGI_TMP/pyenv
source $TGI_TMP/pyenv/bin/activate
python -m pip install --no-index -U pip setuptools wheel build

# install
pip install --no-index --find-links $RELEASE_DIR/python_deps \
  $RELEASE_DIR/python_ins/flash_attn-*.whl $RELEASE_DIR/python_ins/vllm-*.whl \
  $RELEASE_DIR/python_ins/rotary_emb-*.whl $RELEASE_DIR/python_ins/dropout_layer_norm-*.whl \
  $RELEASE_DIR/python_ins/awq_inference_engine-*.whl $RELEASE_DIR/python_ins/EETQ-*.whl \
  $RELEASE_DIR/python_ins/exllama_kernels-*.whl $RELEASE_DIR/python_ins/exllamav2_kernels-*.whl \
  $RELEASE_DIR/python_ins/custom_kernels-*.whl $RELEASE_DIR/python_ins/megablocks-*.whl \
  "$RELEASE_DIR/python_ins/text_generation_server-$TGI_VERSION-py3-none-any.whl[bnb, accelerate, quantize]"
export PATH="$(realpath $RELEASE_DIR/bin/)":$PATH

# configure
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HUB_DISABLE_TELEMETRY=1
export HF_HUB_ENABLE_HF_TRANSFER=1
export HUGGINGFACE_HUB_CACHE=$TGI_DIR/tgi-data

default_num_shard=$(python -c 'import torch; print(torch.cuda.device_count())')
default_port=$(expr 10000 + $(echo -n $SLURM_JOBID | tail -c 4))
default_master_port=$(expr 20000 + $(echo -n $SLURM_JOBID | tail -c 4))
default_shard_usd_path=$TGI_TMP/socket
default_model_path=$TGI_DIR/tgi-repos/$MODEL_ID

# Copy model to tempdir. This will make restarts faster.
rsync --archive --exclude='.git/' --update --delete --verbose --human-readable --whole-file --inplace --no-compress --progress ${MODEL_PATH:-$default_model_path}/ $TGI_TMP/model

# start
text-generation-launcher \
  --model-id $TGI_TMP/model --num-shard "${NUM_SHARD:-$default_num_shard}" \
  --port "${PORT:-$default_port}" \
  --master-port "${MASTER_PORT:-$default_master_port}" \
  --shard-uds-path "${SHARD_UDS_PATH:-$default_shard_usd_path}"
  # --max-best-of $MAX_BEST_OF --max-total-tokens $MAX_STOP_SEQUENCES
  # --max-input-length $MAX_INPUT_LENGTH --max-stop-sequences $MAX_TOTAL_TOKENS --quantize $QUANTIZE
