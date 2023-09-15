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

TGI_VERSION='1.0.2'
FLASH_ATTN_VERSION='2.0.8'

# Default config
if [ -z "${RELEASE_DIR}" ]; then
    RELEASE_DIR=$HOME/tgi-release
fi
if [ -z "${TGI_DIR}" ]; then
    TGI_DIR=$SCRATCH/tgi
fi
if [ -z "${TMP_PYENV}" ]; then
    TMP_PYENV=$SLURM_TMPDIR/tgl-env
fi

# Load modules
module load gcc/9.3.0

# Create enviorment
eval "$(~/bin/micromamba shell hook -s posix)"
micromamba create -y -p $TMP_PYENV -c pytorch -c nvidia -c conda-forge 'python=3.11' 'git-lfs=3.3' 'pyarrow=12.0.1' 'pytorch==2.0.1' 'pytorch-cuda=11.8' 'cudnn=8.8' 'openssl=3'
micromamba activate $TMP_PYENV

# install
pip install --no-index --find-links $RELEASE_DIR/python_deps \
  $RELEASE_DIR/python_ins/flash_attn-*whl $RELEASE_DIR/python_ins/vllm-*.whl \
  $RELEASE_DIR/python_ins/rotary_emb-*.whl $RELEASE_DIR/python_ins/dropout_layer_norm-*.whl \
  $RELEASE_DIR/python_ins/exllama_kernels-*.whl $RELEASE_DIR/python_ins/custom_kernels-*.whl \
  "$RELEASE_DIR/python_ins/text_generation_server-1.0.1-py3-none-any.whl[bnb, accelerate, quantize]"
export PATH="$(realpath $RELEASE_DIR/bin/)":$PATH
export LD_LIBRARY_PATH=$TMP_PYENV/lib:$LD_LIBRARY_PATH

# configure
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HUB_DISABLE_TELEMETRY=1
export HF_HUB_ENABLE_HF_TRANSFER=1
export HUGGINGFACE_HUB_CACHE=$TGI_DIR/tgi-data

export default_num_shard=$(python -c 'import torch; print(torch.cuda.device_count())')
export default_port=$(expr 10000 + $(echo -n $SLURM_JOBID | tail -c 4))
export default_master_port=$(expr 20000 + $(echo -n $SLURM_JOBID | tail -c 4))
export default_shard_usd_path=$SLURM_TMPDIR/tgl-server-socket
export default_model_path=$TGI_DIR/tgi-repos/$MODEL_ID

# start
text-generation-launcher --model-id "${MODEL_PATH:-$default_model_path}" --num-shard "${NUM_SHARD:-$default_num_shard}" \
  --port "${PORT:-$default_port}" \
  --master-port "${MASTER_PORT:-$default_master_port}" \
  --shard-uds-path "${SHARD_UDS_PATH:-$default_shard_usd_path}"
  # --max-best-of $MAX_BEST_OF --max-total-tokens $MAX_STOP_SEQUENCES
  # --max-input-length $MAX_INPUT_LENGTH --max-stop-sequences $MAX_TOTAL_TOKENS --quantize $QUANTIZE
