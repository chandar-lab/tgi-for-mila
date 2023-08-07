#!/bin/bash
#SBATCH --partition=unkillable-cpu
#SBATCH -J tgi-compile
#SBATCH --output=%x.%j.out
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=6:00:00
set -e
set -v

export MAX_JOBS=4
TGI_VERSION='1.0.0'

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

# Create environment
eval "$(~/bin/micromamba shell hook -s posix)"
micromamba create -y -p $TMP_PYENV -c conda-forge python=3.11
micromamba activate $TMP_PYENV
micromamba config append channels conda-forge
micromamba config append channels nodefaults
micromamba install -y 'ninja=1' 'git-lfs=3.3' 'pytorch==2.0.1' 'pytorch-cuda=11.7' 'cuda-nvcc=11.7' 'cudatoolkit=11.7' 'cuda-libraries=11.7' 'cuda-libraries-dev=11.7' 'cudnn=8.8' 'openssl=3' 'gcc=11' 'gxx=11' -c pytorch -c nvidia
export LD_LIBRARY_PATH=$TMP_PYENV/lib:$LD_LIBRARY_PATH
export CPATH=$TMP_PYENV/include:$CPATH
export LIBRARY_PATH=$TMP_PYENV/lib:$LIBRARY_PATH
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
git clone https://github.com/huggingface/text-generation-inference.git $WORK_DIR/text-generation-inference
cd $WORK_DIR/text-generation-inference
git checkout tags/v${TGI_VERSION} -b v${TGI_VERSION}-branch

####
# download and compile python dependencies
####
# download
cd $RELEASE_DIR/python_deps
pip download 'grpcio-tools==1.51.1' 'mypy-protobuf==3.4.0' 'types-protobuf>=3.20.4'
pip download -r $WORK_DIR/text-generation-inference/server/requirements.txt
pip download --no-deps 'accelerate<0.20.0,>=0.19.0' 'bitsandbytes==0.39.1' 'poetry-core>=1.6.1'
pip download --no-deps 'ninja' 'cmake' 'lit' 'packaging'

# build dependencies that are not pre-compiled
pip wheel --no-index --no-deps --find-links $RELEASE_DIR/python_deps 'lit-16.0.6.tar.gz'
rm lit-16.0.6.tar.gz

####
# BUILD tgi
####
cd $WORK_DIR/text-generation-inference

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
pip install --no-index --find-links $RELEASE_DIR/python_deps 'accelerate<0.20.0,>=0.19.0' 'bitsandbytes==0.39.1'
pip wheel --no-deps --no-index --find-links $RELEASE_DIR/python_deps ".[bnb, accelerate]"
cp text_generation_server-${TGI_VERSION}-py3-none-any.whl $RELEASE_DIR/python_ins/
pip install --no-index --find-links $RELEASE_DIR/python_deps $RELEASE_DIR/python_ins/text_generation_server-${TGI_VERSION}-py3-none-any.whl

#
# build cli
#
# build router
cd $WORK_DIR/text-generation-inference/router
OPENSSL_DIR=$TMP_PYENV cargo build -j $MAX_JOBS --release
cp $WORK_DIR/text-generation-inference/target/release/text-generation-router $RELEASE_DIR/bin/

# build launcher
cd $WORK_DIR/text-generation-inference/launcher
cargo build -j $MAX_JOBS --release
cp $WORK_DIR/text-generation-inference/target/release/text-generation-launcher $RELEASE_DIR/bin/

# build benchmark
cd $WORK_DIR/text-generation-inference/benchmark
OPENSSL_DIR=$TMP_PYENV cargo build -j $MAX_JOBS --release
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

# With 16GB of memory, MAX_JOBS=1 is as high as it goes.
cd $WORK_DIR/text-generation-inference/server/flash-attention-v2
TORCH_CUDA_ARCH_LIST=$NV_CC MAX_JOBS=1 python setup.py build
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
git clone https://github.com/OlivierDehaene/vllm.git $WORK_DIR/text-generation-inference/server/vllm
cd $WORK_DIR/text-generation-inference/server/vllm
git fetch && git checkout d284b831c17f42a8ea63369a06138325f73c4cf9
# This patch is because CC 90 require CUDA 11.8+. However, CC 90 is for H100 only, which we don't have/need.
git apply <<EOF
diff --git a/setup.py b/setup.py
index c53005a..8919cf3 100644
--- a/setup.py
+++ b/setup.py
@@ -49,7 +49,7 @@ for i in range(device_count):
     compute_capabilities.add(major * 10 + minor)
 # If no GPU is available, add all supported compute capabilities.
 if not compute_capabilities:
-    compute_capabilities = {70, 75, 80, 86, 90}
+    compute_capabilities = {70, 75, 80, 86}
 # Add target compute capabilities to NVCC flags.
 for capability in compute_capabilities:
     NVCC_FLAGS += ["-gencode", f"arch=compute_{capability},code=sm_{capability}"]
EOF
python setup.py build
python setup.py bdist_egg
wheel convert dist/vllm-0.0.0-py3.11-linux-x86_64.egg
wheel tags --python-tag=cp311 vllm-0.0.0-py311-cp311-linux_x86_64.whl
cp vllm-0.0.0-cp311-cp311-linux_x86_64.whl $RELEASE_DIR/python_ins/

# exllama_kernels
# SKIP: because Mila does not support CUDA 11.8+ and it can be disabled

# custom_kernels
# SKIP: because Mila does not support CUDA 11.8+ and it can be disabled
