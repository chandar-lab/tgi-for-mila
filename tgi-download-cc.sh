#!/bin/bash
#SBATCH --account=rrg-bengioy-ad
#SBATCH -J tgi-download
#SBATCH --output=%x.%j.out
#SBATCH --cpus-per-task=2
#SBATCH --mem=24G
#SBATCH --time=1:00:00
set -e

echo "Downloading ${MODEL_ID}"

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

# do not pull .bin files if .safetensors exists
git remote add origin "${hf_url}/${MODEL_ID}"
if ls *.safetensors 1> /dev/null 2>&1; then
  git lfs pull --exclude "*.bin,.h5,*.msgpack,events.*,/logs,/coreml"
else
  git lfs pull --exclude "**.h5,*.msgpack,events.*,/logs,/coreml"
fi
git remote rm origin  # remove token reference

set -e

# convert .bin to .safetensors if needed
text-generation-server download-weights "${TGI_DIR}/tgi-repos/${MODEL_ID}"
