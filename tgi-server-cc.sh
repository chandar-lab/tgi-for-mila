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
module load python/3.11 gcc/11.3.0 git-lfs/3.3.0 rust/1.65.0 protobuf/3.21.3 cuda/11.8.0 cudnn/8.6.0.163

# create env
virtualenv --app-data $SCRATCH/virtualenv --no-download $TMP_PYENV
source $TMP_PYENV/bin/activate
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
