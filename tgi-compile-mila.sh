#!/bin/bash
#SBATCH --partition=unkillable-cpu
#SBATCH -J tgi-compile
#SBATCH --output=%x.%j.out
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=6:00:00
set -e
set -v

TGI_VERSION='1.3.4'
FLASH_ATTN_VERSION='2.4.1'
export MAX_JOBS=4

# Default config
if [ -z "${RELEASE_DIR}" ]; then
    RELEASE_DIR=$HOME/tgi-next
fi
if [ -z "${TGI_DIR}" ]; then
    TGI_DIR=$SCRATCH/tgi
fi
if [ -z "${TGI_TMP}" ]; then
    TGI_TMP=$SLURM_TMPDIR/tgi
fi
if [ -z "${WORK_DIR}" ]; then
    WORK_DIR=$SLURM_TMPDIR/workspace
fi

# debug info
echo "Storing files in $(realpath $RELEASE_DIR)"
mkdir -p $WORK_DIR

# Load modules
module load gcc/9.3.0

# Create environment
eval "$(~/bin/micromamba shell hook -s posix)"
micromamba create -y -p $TGI_TMP/pyenv -c pytorch -c "nvidia/label/cuda-12.1.0" -c conda-forge 'python=3.11' 'git-lfs=3.3' 'pyarrow=14.0.2' 'pytorch==2.1.1' 'pytorch-cuda=12.1' cuda-nvcc cuda-toolkit cuda-libraries-dev 'cudnn=8.8' 'openssl=3' 'ninja=1'
micromamba activate $TGI_TMP/pyenv
export LD_LIBRARY_PATH=$TGI_TMP/pyenv/lib:$LD_LIBRARY_PATH
export CPATH=$TGI_TMP/pyenv/include:$CPATH
export LIBRARY_PATH=$TGI_TMP/pyenv/lib:$LIBRARY_PATH
export CC=$(which gcc)
export CXX=$(which g++)

# install rust
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs --output $WORK_DIR/rustup.sh
RUSTUP_HOME=$WORK_DIR/.rustup CARGO_HOME=$WORK_DIR/.cargo sh $WORK_DIR/rustup.sh --no-modify-path -y
export PATH=$WORK_DIR/.cargo/bin:$PATH

# install protoc
curl -L https://github.com/protocolbuffers/protobuf/releases/download/v23.4/protoc-23.4-linux-x86_64.zip --output $WORK_DIR/protoc-23.4-linux-x86_64.zip
unzip $WORK_DIR/protoc-23.4-linux-x86_64.zip -d $WORK_DIR/.protoc
export PATH="$WORK_DIR/.protoc/bin:$PATH"

# Create dirs
rm -rf $RELEASE_DIR
mkdir -p $RELEASE_DIR/python_deps
mkdir -p $RELEASE_DIR/python_ins
mkdir -p $RELEASE_DIR/bin

# download source
git clone https://github.com/AndreasMadsen/text-generation-inference.git $WORK_DIR/text-generation-inference
cd $WORK_DIR/text-generation-inference
git checkout origin/v134-new-max -b v${TGI_VERSION}-branch

####
# download and compile python dependencies
####
# download dependencies
cd $RELEASE_DIR/python_deps
pip download 'grpcio-tools==1.51.1' 'mypy-protobuf==3.4.0' 'types-protobuf>=3.20.4'
pip download -r $WORK_DIR/text-generation-inference/server/requirements_cuda.txt
pip download 'bitsandbytes<0.42.0,>=0.41.1' # bnb
pip download 'datasets<3.0.0,>=2.14.0' 'texttable<2.0.0,>=1.6.7' # quantize
pip download 'accelerate<0.26.0,>=0.25.0' # accelerate
pip download 'peft<0.5.0,>=0.4.0' # peft
pip download --no-deps 'poetry-core>=1.6.1' 'ninja' 'cmake' 'lit' 'packaging' # build dependencies
pip download 'wheel' # required for `pip wheel --no-index --find-links $RELEASE_DIR/python_deps`

####
# BUILD tgi
####
pip install --no-index --find-links $RELEASE_DIR/python_deps packaging

#
# build server
#
cd $WORK_DIR/text-generation-inference/server
# Compile protos
pip install --no-index --find-links $RELEASE_DIR/python_deps 'grpcio-tools==1.51.1' 'mypy-protobuf==3.4.0' 'types-protobuf>=3.20.4'
mkdir -p text_generation_server/pb
python -m grpc_tools.protoc -I../proto --python_out=text_generation_server/pb \
    --grpc_python_out=text_generation_server/pb --mypy_out=text_generation_server/pb ../proto/generate.proto
find text_generation_server/pb/ -type f -name "*.py" -print0 -exec sed -i -e 's/^\(import.*pb2\)/from . \1/g' {} \;
touch text_generation_server/pb/__init__.py
rm text_generation_server/pb/.gitignore
# build package
pip install --no-index --find-links $RELEASE_DIR/python_deps 'poetry-core>=1.6.1'
pip wheel --no-deps --no-index --find-links $RELEASE_DIR/python_deps ".[bnb, accelerate, quantize, peft]"
cp "text_generation_server-${TGI_VERSION}-py3-none-any.whl" $RELEASE_DIR/python_ins/

#
# build cli
#
# build router
cd $WORK_DIR/text-generation-inference/router
OPENSSL_DIR=$TGI_TMP/pyenv cargo build -j $MAX_JOBS --release
cp $WORK_DIR/text-generation-inference/target/release/text-generation-router $RELEASE_DIR/bin/

# build launcher
cd $WORK_DIR/text-generation-inference/launcher
cargo build -j $MAX_JOBS --release
cp $WORK_DIR/text-generation-inference/target/release/text-generation-launcher $RELEASE_DIR/bin/

# build benchmark
cd $WORK_DIR/text-generation-inference/benchmark
OPENSSL_DIR=$TGI_TMP/pyenv cargo build -j $MAX_JOBS --release
cp $WORK_DIR/text-generation-inference/target/release/text-generation-benchmark $RELEASE_DIR/bin/

#
# build kernels
#
NV_CC="8.0;8.6" # flash-attention-v2 and exllama_kernels are anyway limited to CC of 8.0+

# flash-attention v2
cd $RELEASE_DIR/python_ins

# For some reason the Dockerfile in TGI, compiles flash_attention, rotary, and layer_norm
# from flash-attention v1. And then compiles flash_attention from flash-attention v2.
# However, this is redudant as rotary and layer_norm are nearly the same and
# flash-attention v1 is only used by TGI when flash-attention v2 does not exists.
# Therefore do only compile flash-attention v2.

cd $WORK_DIR/text-generation-inference/server
git clone https://github.com/Dao-AILab/flash-attention flash-attention-v2
cd $WORK_DIR/text-generation-inference/server/flash-attention-v2
git checkout tags/v${FLASH_ATTN_VERSION}
git submodule update --init --recursive

# With 16GB of memory, MAX_JOBS=1 is as high as it goes.
cd $WORK_DIR/text-generation-inference/server/flash-attention-v2
TORCH_CUDA_ARCH_LIST=$NV_CC MAX_JOBS=1 python setup.py build
TORCH_CUDA_ARCH_LIST=$NV_CC python setup.py bdist_egg
wheel convert dist/flash_attn-${FLASH_ATTN_VERSION}-py3.11-linux-x86_64.egg
wheel tags --python-tag=cp311 flash_attn-${FLASH_ATTN_VERSION}-py311-cp311-linux_x86_64.whl
cp flash_attn-${FLASH_ATTN_VERSION}-cp311-cp311-linux_x86_64.whl $RELEASE_DIR/python_ins/

cd $WORK_DIR/text-generation-inference/server/flash-attention-v2/csrc/rotary
TORCH_CUDA_ARCH_LIST=$NV_CC FLASH_ATTENTION_FORCE_BUILD=True python setup.py build
TORCH_CUDA_ARCH_LIST=$NV_CC python setup.py bdist_egg
wheel convert dist/rotary_emb-0.1-py3.11-linux-x86_64.egg
wheel tags --python-tag=cp311 rotary_emb-0.1-py311-cp311-linux_x86_64.whl
cp rotary_emb-0.1-cp311-cp311-linux_x86_64.whl $RELEASE_DIR/python_ins/

cd $WORK_DIR/text-generation-inference/server/flash-attention-v2/csrc/layer_norm
TORCH_CUDA_ARCH_LIST=$NV_CC FLASH_ATTENTION_FORCE_BUILD=True python setup.py build
TORCH_CUDA_ARCH_LIST=$NV_CC python setup.py bdist_egg
wheel convert dist/dropout_layer_norm-0.1-py3.11-linux-x86_64.egg
wheel tags --python-tag=cp311 dropout_layer_norm-0.1-py311-cp311-linux_x86_64.whl
cp dropout_layer_norm-0.1-cp311-cp311-linux_x86_64.whl $RELEASE_DIR/python_ins/

# vllm
cd $WORK_DIR/text-generation-inference/server
TORCH_CUDA_ARCH_LIST=$NV_CC make build-vllm-cuda
cd $WORK_DIR/text-generation-inference/server/vllm
python setup.py bdist_egg
wheel convert dist/vllm-0.0.0-py3.11-linux-x86_64.egg
wheel tags --python-tag=cp311 vllm-0.0.0-py311-cp311-linux_x86_64.whl
cp vllm-0.0.0-cp311-cp311-linux_x86_64.whl $RELEASE_DIR/python_ins/

# awq
cd $WORK_DIR/text-generation-inference/server
TORCH_CUDA_ARCH_LIST=$NV_CC+PTX make build-awq
cd $WORK_DIR/text-generation-inference/server/llm-awq/awq/kernels
TORCH_CUDA_ARCH_LIST=$NV_CC+PTX python setup.py bdist_egg
wheel convert dist/awq_inference_engine-0.0.0-py3.11-linux-x86_64.egg
wheel tags --python-tag=cp311 awq_inference_engine-0.0.0-py311-cp311-linux_x86_64.whl
cp awq_inference_engine-0.0.0-cp311-cp311-linux_x86_64.whl $RELEASE_DIR/python_ins/

# eetq
cd $WORK_DIR/text-generation-inference/server
TORCH_CUDA_ARCH_LIST=$NV_CC+PTX make build-eetq
cd $WORK_DIR/text-generation-inference/server/eetq
TORCH_CUDA_ARCH_LIST=$NV_CC+PTX python setup.py bdist_egg
wheel convert dist/EETQ-1.0.0b0-py3.11-linux-x86_64.egg
wheel tags --python-tag=cp311 EETQ-1.0.0b0-py311-cp311-linux_x86_64.whl
cp EETQ-1.0.0b0-cp311-cp311-linux_x86_64.whl $RELEASE_DIR/python_ins/

# exllama_kernels
cd $WORK_DIR/text-generation-inference/server/exllama_kernels
TORCH_CUDA_ARCH_LIST=$NV_CC+PTX python setup.py build
TORCH_CUDA_ARCH_LIST=$NV_CC+PTX python setup.py bdist_egg
wheel convert dist/exllama_kernels-0.0.0-py3.11-linux-x86_64.egg
wheel tags --python-tag=cp311 exllama_kernels-0.0.0-py311-cp311-linux_x86_64.whl
cp exllama_kernels-0.0.0-cp311-cp311-linux_x86_64.whl $RELEASE_DIR/python_ins/

# exllamav2_kernels
cd $WORK_DIR/text-generation-inference/server/exllamav2_kernels
TORCH_CUDA_ARCH_LIST=$NV_CC+PTX python setup.py build
TORCH_CUDA_ARCH_LIST=$NV_CC+PTX python setup.py bdist_egg
wheel convert dist/exllamav2_kernels-0.0.0-py3.11-linux-x86_64.egg
wheel tags --python-tag=cp311 exllamav2_kernels-0.0.0-py311-cp311-linux_x86_64.whl
cp exllamav2_kernels-0.0.0-cp311-cp311-linux_x86_64.whl $RELEASE_DIR/python_ins/

# custom_kernels
cd $WORK_DIR/text-generation-inference/server/custom_kernels
TORCH_CUDA_ARCH_LIST=$NV_CC python setup.py build
TORCH_CUDA_ARCH_LIST=$NV_CC python setup.py bdist_egg
wheel convert dist/custom_kernels-0.0.0-py3.11-linux-x86_64.egg
wheel tags --python-tag=cp311 custom_kernels-0.0.0-py311-cp311-linux_x86_64.whl
cp custom_kernels-0.0.0-cp311-cp311-linux_x86_64.whl $RELEASE_DIR/python_ins/

# megablocks
cd $RELEASE_DIR/python_ins
pip wheel --no-index --no-deps --find-links $RELEASE_DIR/python_deps git+https://github.com/OlivierDehaene/megablocks@181709df192de9a941fdf3a641cdc65a0462996e
#TODO: Maybe download dependencies

echo "***************************"
echo "* COMPILE JOB SUCCESSFULL *"
echo "***************************"
