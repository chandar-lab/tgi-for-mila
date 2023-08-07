#!/bin/bash
#SBATCH -J tgi-download
#SBATCH --output=%x.%j.out
#SBATCH --cpus-per-task=2
#SBATCH --mem=24G
#SBATCH --time=1:00:00
set -e

echo "Downloading ${MODEL_ID}"

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

# Create env
eval "$(~/bin/micromamba shell hook -s posix)"
micromamba create -y -p $TMP_PYENV -c pytorch -c nvidia -c conda-forge 'python=3.11' 'git-lfs=3.3' 'pytorch==2.0.1' 'pytorch-cuda=11.7' 'cudnn=8.8' 'openssl=3'
micromamba activate $TMP_PYENV

# install
pip install --no-index --find-links $RELEASE_DIR/python_deps \
  'accelerate<0.20.0,>=0.19.0' \
  'einops<0.7.0,>=0.6.1' \
  $RELEASE_DIR/python_ins/*.whl
export PATH="$(realpath $RELEASE_DIR/bin/)":$PATH
export LD_LIBRARY_PATH=$TMP_PYENV/lib:$LD_LIBRARY_PATH

# prepear directories
mkdir -p $TGI_DIR/tgi-data
mkdir -p $TGI_DIR/tgi-repos

# configure
export HF_HUB_DISABLE_TELEMETRY=1
export HF_HUB_ENABLE_HF_TRANSFER=1
export HUGGINGFACE_HUB_CACHE=$TGI_DIR/tgi-data

# download files
# huggingface_hub.download_snapshot is not used because the ignore_pattern does
# not support directories. So it downloads a ton of unused files.
if [[ -z "${HF_TOKEN}" ]]; then
  hf_url=https://huggingface.co
else
  hf_url=https://hf_user:${HF_TOKEN}@huggingface.co
fi

set +e  # ensure we reach `git remote rm origin`
if [ ! -d "${TGI_DIR}/tgi-repos/${MODEL_ID}" ] ; then
    GIT_LFS_SKIP_SMUDGE=1 git clone "${hf_url}/${MODEL_ID}" "${TGI_DIR}/tgi-repos/${MODEL_ID}"
    cd "${TGI_DIR}/tgi-repos/${MODEL_ID}"
    git remote rm origin
    git lfs install
fi

cd "${TGI_DIR}/tgi-repos/${MODEL_ID}"
git remote add origin "${hf_url}/${MODEL_ID}"

# do not pull .bin files if .safetensors exists
if ls *.safetensors 1> /dev/null 2>&1; then
  git lfs pull --exclude "*.bin,*.h5,*.msgpack,events.*,/logs,/coreml"
else
  git lfs pull --exclude "*.h5,*.msgpack,events.*,/logs,/coreml"
fi
git remote rm origin  # remove token reference
set -e

# convert .bin to .safetensors if needed
text-generation-server download-weights "${TGI_DIR}/tgi-repos/${MODEL_ID}"
