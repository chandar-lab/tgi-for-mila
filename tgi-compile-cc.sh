#!/bin/bash
#SBATCH --account=rrg-bengioy-ad
#SBATCH -J tgi-compile
#SBATCH --output=%x.%j.out
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=10:00:00
set -e
set -v

TGI_VERSION='1.4.4'
FLASH_ATTN_VERSION='2.3.2'
export MAX_JOBS=10

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
if [ -z "${WORK_DIR}" ]; then
    WORK_DIR=$SLURM_TMPDIR/workspace
fi

# debug info
echo "Storing files in $(realpath $RELEASE_DIR)"
mkdir -p $WORK_DIR

# Load modules
module load StdEnv/2023
module load python/3.11 git-lfs/3.4.0 rust/1.76.0 cmake/3.27.7 protobuf/24.4 arrow/14.0.1 cuda/12.2 cudnn/8.9.5.29
export CC=$(which gcc)
export CXX=$(which g++)

# Create environment
virtualenv --app-data $SCRATCH/virtualenv --no-download $TGI_TMP/pyenv
set +v
source $TGI_TMP/pyenv/bin/activate
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
pip download --no-deps 'mypy-protobuf==3.4.0' 'types-protobuf>=3.20.4'
grep -ivE "pyarrow|scipy|numpy" $WORK_DIR/text-generation-inference/server/requirements_cuda.txt > $WORK_DIR/requirements_cuda.txt
pip download --no-deps -r $WORK_DIR/requirements_cuda.txt
pip download 'bitsandbytes<0.44.0,>=0.43.0' # bnb
pip download 'datasets<3.0.0,>=2.14.0' 'texttable<2.0.0,>=1.6.7' # quantize
pip download 'accelerate<0.29.0,>=0.28.0' # accelerate
pip download 'peft<0.10.0,>=0.9.0' # peft
pip download --no-deps 'outlines<0.0.37,>=0.0.36' 'interegular==0.3.3' # outlines
pip download --no-deps 'poetry-core>=1.6.1' # build dependencies

# cleanup dependencies where another acceptable version is provided by compute canada
rm -f *+computecanada*.whl
rm -f grpcio-* tokenizers-* MarkupSafe-* safetensors-* aiohttp-* frozenlist-* numpy-* scipy-* pandas-* pyarrow-* xxhash-* yarl-* Pillow-* regex-* psutil-* hf_transfer-*

####
# BUILD tgi
####
pip install --no-index --find-links $RELEASE_DIR/python_deps "torch==2.1.2"
pip install --no-index --find-links $RELEASE_DIR/python_deps packaging

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
# build package
pip install --no-index --find-links $RELEASE_DIR/python_deps -U 'poetry-core>=1.6.1'
pip wheel --no-deps --no-index --find-links $RELEASE_DIR/python_deps ".[bnb, accelerate, quantize, peft, outlines]"
cp "text_generation_server-${TGI_VERSION}-py3-none-any.whl" $RELEASE_DIR/python_ins/

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

# flash-attention v2
# NOTE: compute canada does include flash_attn in the wheelhouse. However, this is not compatiable
# with the downloaded torch+cu118 version. Therefore it needs to be compiled.

# For some reason the Dockerfile in TGI, compiles flash_attention, rotary, and layer_norm
# from flash-attention v1. And then compiles flash_attention from flash-attention v2.
# However, this is redudant as rotary and layer_norm are nearly the same and
# flash-attention v1 is only used by TGI when flash-attention v2 does not exists.
# Therefore do only compile flash-attention v2.
cd $WORK_DIR/text-generation-inference/server
git clone https://github.com/Dao-AILab/flash-attention flash-attention-v2
cd $WORK_DIR/text-generation-inference/server/flash-attention-v2
git checkout '02ac572f3ffc4f402e4183aaa6824b45859d3ed3'
git submodule update --init --recursive

# With 48GB of memory, MAX_JOBS=4 is as high as it goes.
cd $WORK_DIR/text-generation-inference/server/flash-attention-v2
TORCH_CUDA_ARCH_LIST=$NV_CC MAX_JOBS=4 python setup.py build
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
sed -i 's/torch == 2.0.1/torch/g' pyproject.toml
sed -i 's/torch == 2.0.1/torch/g' requirements.txt
sed -i 's/pydantic < 2/pydantic/g' requirements.txt
sed -i 's/xformers == 0.0.22/xformers/g' requirements.txt
TORCH_CUDA_ARCH_LIST=$NV_CC python setup.py bdist_egg
wheel convert dist/vllm-0.2.1-py3.11-linux-x86_64.egg
wheel tags --python-tag=cp311 vllm-0.2.1-py311-cp311-linux_x86_64.whl
cp vllm-0.2.1-cp311-cp311-linux_x86_64.whl $RELEASE_DIR/python_ins/

# awq
cd $WORK_DIR/text-generation-inference/server
export TORCH_CUDA_ARCH_LIST=$NV_CC+PTX
make build-awq || make build-awq || make build-awq
export -n TORCH_CUDA_ARCH_LIST
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

# Build mamba kernels
cd $WORK_DIR/text-generation-inference/server
TORCH_CUDA_ARCH_LIST=$NV_CC make build-all
cd $WORK_DIR/text-generation-inference/server/causal-conv1d
TORCH_CUDA_ARCH_LIST=$NV_CC CAUSAL_CONV1D_FORCE_BUILD=TRUE python setup.py bdist_egg
wheel convert dist/causal_conv1d-1.1.1-py3.11-linux-x86_64.egg
wheel tags --python-tag=cp311 causal_conv1d-1.1.1-py311-cp311-linux_x86_64.whl
cp causal_conv1d-1.1.1-cp311-cp311-linux_x86_64.whl $RELEASE_DIR/python_ins/
cd $WORK_DIR/text-generation-inference/server/mamba
TORCH_CUDA_ARCH_LIST=$NV_CC python setup.py bdist_egg
wheel convert dist/mamba_ssm-1.1.2-py3.11-linux-x86_64.egg
wheel tags --python-tag=cp311 mamba_ssm-1.1.2-py311-cp311-linux_x86_64.whl
cp mamba_ssm-1.1.2-cp311-cp311-linux_x86_64.whl $RELEASE_DIR/python_ins/

# megablocks
cd $RELEASE_DIR/python_deps
pip wheel --no-deps --no-index --find-links $RELEASE_DIR/python_deps "git+https://github.com/OlivierDehaene/stk@0536f762df7a406de508d17a0ee4c124e8ad7f06"

cd $WORK_DIR/
git clone https://github.com/OlivierDehaene/megablocks.git
cd megablocks
git checkout "181709df192de9a941fdf3a641cdc65a0462996e"
sed -i 's#stanford-stk @ git+https://github.com/OlivierDehaene/stk.git@0536f762df7a406de508d17a0ee4c124e8ad7f06#stanford-stk#g' setup.py
sed -i 's#stanford-stk @ git+https://github.com/OlivierDehaene/stk.git@0536f762df7a406de508d17a0ee4c124e8ad7f06#stanford-stk#g' requirements.txt
TORCH_CUDA_ARCH_LIST=$NV_CC python setup.py bdist_egg
wheel convert dist/megablocks-0.5.0-py3.11-linux-x86_64.egg
wheel tags --python-tag=cp311 megablocks-0.5.0-py311-cp311-linux_x86_64.whl
cp megablocks-0.5.0-cp311-cp311-linux_x86_64.whl $RELEASE_DIR/python_ins/

echo "***************************"
echo "* COMPILE JOB SUCCESSFULL *"
echo "***************************"
