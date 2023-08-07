#!/bin/bash
#SBATCH --account=rrg-bengioy-ad
#SBATCH -J compile-native
#SBATCH --output=%x.%j.out
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=6:00:00
set -e
set -v

export MAX_JOBS=10
TGI_VERSION='1.0.0'
WORK_DIR=$SLURM_TMPDIR/workspace

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
if [ -z "${WORK_DIR}" ]; then
    WORK_DIR=$SLURM_TMPDIR/workspace
fi

# debug info
echo "Storing files in $(realpath $RELEASE_DIR)"
mkdir -p $WORK_DIR

# Load modules
module load python/3.11 gcc/11.3.0 git-lfs/3.3.0 rust/1.65.0 protobuf/3.21.3 cuda/11.8.0 cudnn/8.6.0.163
export CC=$(which gcc)
export CXX=$(which g++)

# Create environment
virtualenv --app-data $SCRATCH/virtualenv --no-download $TMP_PYENV
set +v
source $TMP_PYENV/bin/activate
set -v
python -m pip install --no-index -U pip setuptools wheel build

# Create dirs
rm -rf $RELEASE_DIR
mkdir -p $RELEASE_DIR/python_deps
mkdir -p $RELEASE_DIR/python_ins
mkdir -p $RELEASE_DIR/bin

# download source
git clone https://github.com/huggingface/text-generation-inference.git $WORK_DIR/text-generation-inference
cd $WORK_DIR/text-generation-inference
git checkout tags/v${TGI_VERSION} -b v${TGI_VERSION}-branch

####
# download and compile python dependencies
####
# download
cd $RELEASE_DIR/python_deps
pip download --no-deps 'grpcio-tools==1.51.1' 'mypy-protobuf==3.4.0' 'types-protobuf>=3.20.4'
pip download --no-deps 'poetry-core>=1.6.1' 'accelerate>=0.19.0,<0.20.0'
pip download --no-deps --index-url https://download.pytorch.org/whl/cu118 'torch==2.0.1'
grep -ivE "pyarrow" $WORK_DIR/text-generation-inference/server/requirements.txt > $WORK_DIR/requirements.txt
pip download --no-deps -r $WORK_DIR/requirements.txt

# cleanup dependencies where another acceptable version is provided by compute canada
rm *+computecanada*.whl
rm -f numpy-1.25.0.tar.gz sentencepiece-0.1.99.tar.gz regex-2023.6.3.tar.gz \
    pandas-2.0.3.tar.gz yarl-1.9.2.tar.gz frozenlist-1.4.0.tar.gz \
    MarkupSafe-2.1.3.tar.gz aiohttp-3.8.5.tar.gz psutil-5.9.5.tar.gz

# remove bitsandbytes, this require custom compiling
rm bitsandbytes-0.38.1-py3-none-any.whl

# build dependencies that are not pre-compiled
pip wheel --no-index --no-deps --find-links $RELEASE_DIR/python_deps 'hf_transfer-0.1.3.tar.gz'
rm hf_transfer-0.1.3.tar.gz

tar xvf grpcio-tools-1.51.1.tar.gz
cd grpcio-tools-1.51.1
python -m build
cp dist/grpcio_tools-1.51.1-cp311-cp311-linux_x86_64.whl $RELEASE_DIR/python_deps/
cd $RELEASE_DIR/python_deps
rm -rf grpcio-tools-1.51.1 grpcio-tools-1.51.1.tar.gz

tar xvf grpcio-1.56.0.tar.gz
cd grpcio-1.56.0
python -m build
cp dist/grpcio-1.56.0-cp311-cp311-linux_x86_64.whl $RELEASE_DIR/python_deps/
cd $RELEASE_DIR/python_deps
rm -rf grpcio-1.56.0 grpcio-1.56.0.tar.gz

#
# Build bitsandbytes
#
git clone https://github.com/TimDettmers/bitsandbytes.git $WORK_DIR/bitsandbytes
cd $WORK_DIR/bitsandbytes
git checkout tags/0.40.0 -b 0.40.0-branch
CUDA_VERSION=118 make GPP=$(which g++) cuda11x
python setup.py bdist_egg
wheel convert dist/bitsandbytes-0.40.0.post4-py3.11.egg
cp bitsandbytes-0.40.0.post4-py311-none-any.whl $RELEASE_DIR/python_ins/

####
# BUILD tgi
####

#
# build server
#
cd $WORK_DIR/text-generation-inference/server
# Compile protos
pip install --no-index --find-links $RELEASE_DIR/python_deps grpcio-tools==1.51.1 mypy-protobuf==3.4.0 'types-protobuf>=3.20.4'
mkdir -p text_generation_server/pb
python -m grpc_tools.protoc -I../proto --python_out=text_generation_server/pb \
    --grpc_python_out=text_generation_server/pb --mypy_out=text_generation_server/pb ../proto/generate.proto
find text_generation_server/pb/ -type f -name "*.py" -print0 -exec sed -i -e 's/^\(import.*pb2\)/from . \1/g' {} \;
touch text_generation_server/pb/__init__.py
rm text_generation_server/pb/.gitignore
# Install specific version of torch
pip install --no-index --find-links $RELEASE_DIR/python_deps "torch==2.0.1"
# build package
pip install --no-index --find-links $RELEASE_DIR/python_deps -U 'poetry-core>=1.6.1'
pip wheel --no-deps --no-index --find-links $RELEASE_DIR/python_deps ".[bnb, accelerate]"
cp text_generation_server-${TGI_VERSION}-py3-none-any.whl $RELEASE_DIR/python_ins/
pip install --no-index --find-links $RELEASE_DIR/python_deps $RELEASE_DIR/python_ins/text_generation_server-${TGI_VERSION}-py3-none-any.whl

#
# build cli
#
# build router
cd $WORK_DIR/text-generation-inference/router
cargo build -j $MAX_JOBS --release
cp $WORK_DIR/text-generation-inference/target/release/text-generation-router $RELEASE_DIR/bin/

# build launcher
cd $WORK_DIR/text-generation-inference/launcher
cargo build -j $MAX_JOBS --release
cp $WORK_DIR/text-generation-inference/target/release/text-generation-launcher $RELEASE_DIR/bin/

# build benchmark
cd $WORK_DIR/text-generation-inference/benchmark
cargo build -j $MAX_JOBS --release
cp $WORK_DIR/text-generation-inference/target/release/text-generation-benchmark $RELEASE_DIR/bin/

#
# build kernels
#
NV_CC="8.0;8.6" # flash-attention-v2 and exllama_kernels are anyway limited to CC of 8.0+
pip install --no-index --find-links $RELEASE_DIR/python_deps packaging

# flash-attention v2
# For some reason the Dockerfile in TGI, compiles flash_attention, rotary, and layer_norm
# from flash-attention v1. And then compiles flash_attention from flash-attention v2.
# However, this is redudant as rotary and layer_norm are nearly the same and
# flash-attention v1 is only used by TGI when flash-attention v2 does not exists.
# Therefore do only compile flash-attention v2.
cd $WORK_DIR/text-generation-inference/server
git clone https://github.com/HazyResearch/flash-attention.git flash-attention-v2
cd $WORK_DIR/text-generation-inference/server/flash-attention-v2
git checkout tags/v2.0.1

# For some reason, this build step fails unless MAX_JOBS is lowered.
# It may be an hidden OOM issue. Apparently, it takes a lot to compile these files.
# Some anendotal measure, MAX_JOBS=4 runs 4 nvcc programs, each may run two threads
# CC=8.0 and CC=9.0 (for some reason), each thread can use up to 5.2 GB and ninja use 0.5GB
# total = 42.1GB ram.
cd $WORK_DIR/text-generation-inference/server/flash-attention-v2
TORCH_CUDA_ARCH_LIST=$NV_CC MAX_JOBS=4 python setup.py build
TORCH_CUDA_ARCH_LIST=$NV_CC python setup.py bdist_egg
wheel convert dist/flash_attn-2.0.1-py3.11-linux-x86_64.egg
wheel tags --python-tag=cp311 flash_attn-2.0.1-py311-cp311-linux_x86_64.whl
cp flash_attn-2.0.1-cp311-cp311-linux_x86_64.whl $RELEASE_DIR/python_ins/

cd $WORK_DIR/text-generation-inference/server/flash-attention-v2/csrc/rotary
TORCH_CUDA_ARCH_LIST=$NV_CC python setup.py build
TORCH_CUDA_ARCH_LIST=$NV_CC python setup.py bdist_egg
wheel convert dist/rotary_emb-0.1-py3.11-linux-x86_64.egg
wheel tags --python-tag=cp311 rotary_emb-0.1-py311-cp311-linux_x86_64.whl
cp rotary_emb-0.1-cp311-cp311-linux_x86_64.whl $RELEASE_DIR/python_ins/

cd $WORK_DIR/text-generation-inference/server/flash-attention-v2/csrc/layer_norm
TORCH_CUDA_ARCH_LIST=$NV_CC python setup.py build
TORCH_CUDA_ARCH_LIST=$NV_CC python setup.py bdist_egg
wheel convert dist/dropout_layer_norm-0.1-py3.11-linux-x86_64.egg
wheel tags --python-tag=cp311 dropout_layer_norm-0.1-py311-cp311-linux_x86_64.whl
cp dropout_layer_norm-0.1-cp311-cp311-linux_x86_64.whl $RELEASE_DIR/python_ins/

# vllm
cd $WORK_DIR/text-generation-inference/server
make build-vllm
cd $WORK_DIR/text-generation-inference/server/vllm
python setup.py bdist_egg
wheel convert dist/vllm-0.0.0-py3.11-linux-x86_64.egg
wheel tags --python-tag=cp311 vllm-0.0.0-py311-cp311-linux_x86_64.whl
cp vllm-0.0.0-cp311-cp311-linux_x86_64.whl $RELEASE_DIR/python_ins/

# exllama_kernels
cd $WORK_DIR/text-generation-inference/server/exllama_kernels
TORCH_CUDA_ARCH_LIST=$NV_CC+PTX python setup.py build
TORCH_CUDA_ARCH_LIST=$NV_CC+PTX python setup.py bdist_egg
wheel convert dist/exllama_kernels-0.0.0-py3.11-linux-x86_64.egg
wheel tags --python-tag=cp311 exllama_kernels-0.0.0-py311-cp311-linux_x86_64.whl
cp exllama_kernels-0.0.0-cp311-cp311-linux_x86_64.whl $RELEASE_DIR/python_ins/

# custom_kernels
cd $WORK_DIR/text-generation-inference/server/custom_kernels
TORCH_CUDA_ARCH_LIST=$NV_CC python setup.py build
TORCH_CUDA_ARCH_LIST=$NV_CC python setup.py bdist_egg
wheel convert dist/custom_kernels-0.0.0-py3.11-linux-x86_64.egg
wheel tags --python-tag=cp311 custom_kernels-0.0.0-py311-cp311-linux_x86_64.whl
cp custom_kernels-0.0.0-cp311-cp311-linux_x86_64.whl $RELEASE_DIR/python_ins/
