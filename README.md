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
`unkillable-cpu` partition. You may set the `RELEASE_DIR` to any directory, the release will then be saved
in `$RELEASE_DIR`.

```bash
sbatch --export=ALL,RELEASE_DIR=$HOME/tgi-release tgi-compile-mila.sh
```

Note, because Mila does not support CUDA 11.8+, this will not compile the ["custom-kernels"](https://github.com/huggingface/text-generation-inference/tree/main/server/custom_kernels). But you will still get the other optimized kernels
flash-attention, etc.

### Compute Canada

The compile script needs an internet access and will therefore only work on Cedar. However, once compiled
you can copy the release to any other compute canada cluster (tested with Narval).

```bash
sbatch --export=ALL,RELEASE_DIR=$HOME/tgi-release tgi-compile-cc.sh
```

Note that Cedar does not permit the log files to be stored on $HOME. So start the script from either
~/scratch or ~/projects.

## Download model

This script must be either run on Cedar or Mila, as those are the only clusters with internet access. If downloaded on Cedar,
the downloaded files in $TGI_DIR can be transfered easily to other clusters with https://globus.alliancecan.ca.

For some models, like Llama 2, you will need to generate a read access token. You can do so here: https://huggingface.co/settings/tokens and provide it via the optional `HF_TOKEN` variable.

```bash
sbatch --export=ALL,HF_TOKEN=hf_abcdef,RELEASE_DIR=$HOME/tgi-release,TGI_DIR=$SCRATCH/tgi,MODEL_ID=tiiuae/falcon-7b-instruct tgi-download-{mila,cc}.sh
```

## Start server

It is recommended to only run the server on a A100 GPU, or perhaps never, this is due to the ``Nvidia compute capability''. To start the server use:

```bash
sbatch --export=ALL,RELEASE_DIR=$HOME/tgi-release,TGI_DIR=$SCRATCH/tgi,MODEL_ID=tiiuae/falcon-7b-instruct tgi-server-{mila,cc}.sh
```

Or use the bash script directly:

```bash
RELEASE_DIR=$HOME/tgi-release,TGI_DIR=$SCRATCH/tgi,MODEL_ID=tiiuae/falcon-7b-instruct bash start-native-{mila,cc}.sh
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
| meta-llama/Llama-2-70b-chat-hf | 2x A100 (80GB)          | `--cpus-per-task=24 --gpus-per-task=2 --mem=128G --constraint=ampere&nvlink` | 97%        | 82%       |
| meta-llama/Llama-2-13b-chat-hf | 1x A100 (80GB)          | `--cpus-per-task=24 --gpus-per-task=1 --mem=128G --constraint=ampere`        | 96%        | 73%       |
| meta-llama/Llama-2-7b-chat-hf  | 3/7 A100 (80GB) (=40GB) | `--cpus-per-task=4 --gpus-per-task=1=a100l.3 --mem=24G --constraint=ampere`  |            |           |
| tiiuae/falcon-40b-instruct     | 2x A100 (80GB)          | `--cpus-per-task=24 --gpus-per-task=2 --mem=128G --constraint=ampere&nvlink` | 96 %       | 76 %      |
| tiiuae/falcon-7b-instruct      | 3/7 A100 (80GB) (=40GB) | `--cpus-per-task=4 --gpus-per-task=1=a100l.3 --mem=24G --constraint=ampere`  |            |           |
| google/flan-t5-xxl             | 1x A100 (80GB)          | `--cpus-per-task=24 --gpus-per-task=1 --mem=128G --constraint=ampere`        |            |           |
| bigscience/bloomz              |                         |                                                                              |            |           |

### Extra arguments:

Besides `RELEASE_DIR` and `TGI_DIR`, the start scripts takes the following arguments:

**`MODEL_ID`**
The name of the model to load.

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

**`QUANTIZE`**
Whether you want the model to be quantized. This will use `bitsandbytes` for quantization on the fly, or `gptq`
 |