# Text Generation Inference for Compute Canada and Mila

Setups a runtime for https://github.com/huggingface/text-generation-inference, which can run nativly on
Compute Canada and Mila clusters.

## Compile release

### Mila

From a login node run:

```bash
cd ~/ && curl -Ls https://micro.mamba.pm/api/micromamba/linux-64/latest | tar -xvj bin/micromamba
```

Then schedule the compile script. Note that this script does not support preemption, so it uses the
`unkillable-cpu` partition. The release will then be saved in `$HOME/tgi-release`, can be changed with
the `RELEASE_DIR`.

```bash
sbatch tgi-compile-mila.sh
```

Note, because Mila does not support CUDA 11.8+, this will not compile the ["custom-kernels"](https://github.com/huggingface/text-generation-inference/tree/main/server/custom_kernels). But you will still get the other optimized kernels
flash-attention, etc.

### Compute Canada

The compile script needs an internet access and will therefore only work on Cedar. However, once compiled
you can copy the release to any other compute canada cluster (tested with Narval).

```bash
sbatch tgi-compile-cc.sh
```

Note that Cedar does not permit the log files to be stored on $HOME. So start the script from either
~/scratch or ~/projects.

## Download model

This script must be either run on Cedar or Mila, as those are the only clusters with internet access. If downloaded on Cedar,
the downloaded files in `$SCRATCH/tgi` can be transfered easily to other clusters with https://globus.alliancecan.ca. The directory
can also be changed with `TGI_DIR`.

For some models, like Llama 2, you will need to generate a read access token. You can do so here: https://huggingface.co/settings/tokens and provide it via the optional `HF_TOKEN` variable.

```bash
sbatch --export=ALL,HF_TOKEN=hf_abcdef,MODEL_ID=tiiuae/falcon-7b-instruct tgi-download-{mila,cc}.sh
```

## Start server

It is recommended to only run the server on a A100 GPU, or perhaps never, this is due to the ``Nvidia compute capability''. To start the server use:

```bash
sbatch --export=ALL,MODEL_ID=tiiuae/falcon-7b-instruct tgi-server-{mila,cc}.sh
```

Or use the bash script directly:

```bash
MODEL_ID=tiiuae/falcon-7b-instruct bash tgi-server-{mila,cc}.sh
```

Remember to chose either mila or cc (compute canada) accordingly.

This script creates a job called tgi, you may follow it with:

```bash
squeue -i 2 -n tgi -u $USER --format="%.8i %.8T %9N"

Wed Jul 26 08:52:33 2023
   JOBID    STATE NODELIST
 3430148  PENDING

Wed Jul 26 08:52:35 2023
   JOBID    STATE NODELIST
 3430148  RUNNING cn-g017
```

Once running, the server will eventually listen to:
* PORT: `$(expr 10000 + $(echo -n $SLURM_JOBID | tail -c 4))`
* ADDR: `"0.0.0.0"`

A simple example to check that the server is running and working.
Remember to change the JobID and PORT.

```bash
curl http://cn-g017:10148/generate \
  -X POST \
  -d '{"inputs":"What is Deep Learning?","parameters":{"max_new_tokens":20}}' \
  -H 'Content-Type: application/json'
```

You can port-forward via ssh, such that you have access to the server locally. For example:

```bash
ssh -N -f -L localhost:20001:cn-g017:10148 mila
```

### Required resources

This is a list of configurations that are confirmed to work on the Mila cluster. Less resources may
be possible. If you are using the server interactively, you may benifit from using
`--partition=short-unkillable` which allow a 3 hour, 4 GPU job. The specific memory and compute
utilization, will depend on the inputs. Those numbers are just for one large input. TGI will batch
multiple inputs of similar lengths.

| `MODEL_ID`                     | GPUs                    | Mila slurm flags                                                             | Comp. Util | Mem Util. |
| ------------------------------ | ----------------------- | ---------------------------------------------------------------------------- | ---------- | --------- |
| meta-llama/Llama-2-70b-chat-hf | 2x A100 (80GB)          | `--cpus-per-task=24 --gpus-per-task=a100l:2 --mem=128G --constraint=ampere&nvlink` | 97%        | 82%       |
| meta-llama/Llama-2-13b-chat-hf | 1x A100 (80GB)          | `--cpus-per-task=24 --gpus-per-task=a100l:1 --mem=128G --constraint=ampere`        | 96%        | 73%       |
| meta-llama/Llama-2-7b-chat-hf  | 3/7 A100 (80GB) (=40GB) | `--cpus-per-task=4 --gpus-per-task=a100l.3:1 --mem=24G --constraint=ampere`  |            |           |
| tiiuae/falcon-40b-instruct     | 2x A100 (80GB)          | `--cpus-per-task=24 --gpus-per-task=a100l:2 --mem=128G --constraint=ampere&nvlink` | 96 %       | 76 %      |
| tiiuae/falcon-7b-instruct      | 3/7 A100 (80GB) (=40GB) | `--cpus-per-task=4 --gpus-per-task=a100l.3:1 --mem=24G --constraint=ampere`  |            |           |
| google/flan-t5-xxl             | 1x A100 (80GB)          | `--cpus-per-task=24 --gpus-per-task=a100l:1 --mem=128G --constraint=ampere`        |            |           |
| bigscience/bloomz              |                         |                                                                              |            |           |

### Multiple instances on the same job

Because configuration is based on the `$SLURM_JOBID` and `$SLURM_TMPDIR`, you will need to modify some paramaters to run multiple instances in the same job. In particular, `TMP_PYENV`, `SHARD_UDS_PATH`, `PORT`, `MASTER_PORT`, `CUDA_VISIBLE_DEVICES`, and `NUM_SHARD`.

Example script to start both `meta-llama/Llama-2-70b-chat-hf` and `tiiuae/falcon-40b-instruct` on Mila:

```bash
#!/bin/bash
#SBATCH -J tgi-server
#SBATCH --output=%x.%j.out
#SBATCH --cpus-per-task=24
#SBATCH --gpus-per-task=a100l:4
#SBATCH --ntasks=1
#SBATCH --constraint=ampere&nvlink
#SBATCH --mem=128G
#SBATCH --time=2:59:00
#SBATCH --partition=short-unkillable

echo "MODEL: tiiuae/falcon-40b-instruct"
echo "PROXY: ssh -N -f -L localhost:20001:$SLURMD_NODENAME:$(expr 10000 + $(echo -n $SLURM_JOBID | tail -c 4)) mila"
echo ""
echo "MODEL: meta-llama/Llama-2-70b-chat-hf"
echo "PROXY: ssh -N -f -L localhost:20002:$SLURMD_NODENAME:$(expr 20000 + $(echo -n $SLURM_JOBID | tail -c 4)) mila"
echo ""

MODEL_ID=tiiuae/falcon-40b-instruct TMP_PYENV=$SLURM_TMPDIR/tgl-env-01 SHARD_UDS_PATH=$SLURM_TMPDIR/tgl-server-socket-01 PORT=$(expr 10000 + $(echo -n $SLURM_JOBID | tail -c 4)) MASTER_PORT=$(expr 20000 + $(echo -n $SLURM_JOBID | tail -c 4)) CUDA_VISIBLE_DEVICES=0,1 NUM_SHARD=2 bash tgi-server-mila.sh &

MODEL_ID=meta-llama/Llama-2-70b-chat-hf TMP_PYENV=$SLURM_TMPDIR/tgl-env-23 SHARD_UDS_PATH=$SLURM_TMPDIR/tgl-server-socket-23 PORT=$(expr 30000 + $(echo -n $SLURM_JOBID | tail -c 4)) MASTER_PORT=$(expr 40000 + $(echo -n $SLURM_JOBID | tail -c 4)) CUDA_VISIBLE_DEVICES=2,3 NUM_SHARD=2 bash tgi-server-mila.sh &
```

## Arguments:

### Required

Only specify one of these parameters:

**`MODEL_ID`** (download and server script only)
The name of the model to load or download. When downloading, the model will be downloaded from `hf.co/$MODEL_ID`.

**`MODEL_PATH`** (download and server script only)

If the model is not a `hf.co/` repository, you can use `MODEL_PATH` to point to a local repository (directory)
instead. This repository needs to have the same Huggingface format as a `hf.co/` repository. This can typically
be done with `model.save_pretrained(repo_dir)` and `tokenizer.save_pretrained(repo_dir)`.

### Optional script parameters

These should be set for all scripts if modified. For example:

```bash
sbatch --export=ALL,RELEASE_DIR=$HOME/tgi-custom tgi-compile-mila.sh # custom
```

**`RELEASE_DIR`**
Points to the directory where the compiled files exists. Needs to be the same for every script.

Default: `$HOME/tgi-release`.

**`TGI_DIR`**  (download and server script only)
Points to the directory where model weights and configurations are saved. Needs to be the same for the
  `tgi-download` and `tgi-server` scripts.

Default: `$SCRATCH/tgi`.

**`TMP_PYENV`**
The directory where the temporary python enviorment will be created.

Default: `$SLURM_TMPDIR/tgl-env`.

**`WORK_DIR`** (compile script only)
The directory where the temporary source files will be stored.

Default: `$SLURM_TMPDIR/workspace`.

### Optional TGI Parameters

These parameters only apply to the `tgi-server-{mila,cc}.sh` scripts.

**`NUM_SHARD`**
The number of shards to use if you don't want to use all GPUs on a given machine. You can use `CUDA_VISIBLE_DEVICES=0,1 NUM_SHARD=2` and `CUDA_VISIBLE_DEVICES=2,3 NUM_SHARD=2` to launch 2 copies with 2 shard each on a given machine with 4 GPUs for instance.

**`QUANTIZE`**
Whether you want the model to be quantized. This will use `bitsandbytes` for quantization on the fly, or `gptq`

[possible values: bitsandbytes, gptq]

**`DTYPE`**
The dtype to be forced upon the model. This option cannot be used with `--quantize`

[possible values: float16, bfloat16]

**`MAX_BEST_OF`**
This is the maximum allowed value for clients to set `best_of`. Best of makes `n` generations at the same time,
  and return the best in terms of overall log probability over the entire generated sequence

**`MAX_STOP_SEQUENCES`**
This is the maximum allowed value for clients to set `stop_sequences`. Stop sequences are used to allow the
  model to stop on more than just the EOS token, and enable more complex "prompting" where users can preprompt
  the model in a specific way and define their "own" stop token aligned with their prompt

**`MAX_INPUT_LENGTH`**
This is the maximum allowed input length (expressed in number of tokens) for users. The larger this value,
  the longer prompt users can send which can impact the overall memory required to handle the load. Please note
  that some models have a finite range of sequence they can handle

**`MAX_TOTAL_TOKENS`**
This is the most important value to set as it defines the "memory budget" of running clients requests.
  Clients will send input sequences and ask to generate `max_new_tokens` on top. with a value of `1512` users
  can send either a prompt of `1000` and ask for `512` new tokens, or send a prompt of `1` and ask for `1511`
  max_new_tokens. The larger this value, the larger amount each request will be in your RAM and the less
  effective batching can be

**`PORT`**
The port to listen on

[default: `$(expr 10000 + $(echo -n $SLURM_JOBID | tail -c 4))`]

**`SHARD_UDS_PATH`**
The name of the socket for gRPC communication between the webserver and the shards

[default: `$SLURM_TMPDIR/tgl-server-socket`]

**`MASTER_PORT`**
The address the master port will listen on. (setting used by torch distributed)

[default: `$(expr 20000 + $(echo -n $SLURM_JOBID | tail -c 4))`]
