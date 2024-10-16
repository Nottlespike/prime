# prime - decentralized training at scale
prime (previously called ZeroBand) is a framework for efficient, globally distributed training of AI models over the internet.

https://github.com/user-attachments/assets/c034d2a2-400c-4bf8-acd0-c84b6c897d69

## Key Features

- **`ElasticDeviceMesh` for Fault Tolerant Training:**
    - In Prime, we’ve added a new distributed abstraction called `ElasticDeviceMesh` which encapsulates dynamic global process groups for fault-tolerant communication across the internet and local process groups for communication within a node or datacenter.
    - The `ElasticDeviceMesh` manages the resizing of the global process groups when nodes join or leave, unlike the standard `DeviceMesh` in torch distributed, which will crash and require a cold restart to resize the process group.
    - In order to know when to resize the process groups, we use a heartbeat mechanism to discover dead nodes and remove them from the process group. Crashing nodes will attempt a best effort deathrattle to fail their own heartbeat quickly, saving its comrades the timeout.
- **Asynchronous distributed checkpointing**
    - Due to the size of the model, checkpointing can be an expensive operation, taking up to 20 minutes on the nodes we tested. This would reduce our compute utilisation if it blocked the main training process.
    - In order to minimize the blocking time, we first checkpoint into `/dev/shm` which is a RAM backed filesystem. This operation is much faster and we can unblock the main training process once the checkpoint has been created in `/dev/shm`.
    - We then use two subprocesses to asynchronously copy the checkpoint out of `/dev/shm` into the checkpoint directory on disk as well as upload it to the remote.
- **Live checkpoint recovery**
    - Nodes that wish to join the run mid-training need to be able to get the most recent state of the model and optimiser before being able to contribute to the training. They must complete this operation in the time window between two outer steps, otherwise, the checkpoint they receive would be stale.
    - In order to do this quickly, we have the joining nodes request the checkpoints from its peers which all host a sidecar HTTP server serving the latest checkpoint out of `/dev/shm`.
    - Once the joining node has downloaded and initialized the model, it skips the inner steps and joins the outer step with zero pseudo-gradients. This is to prevent the joining node from stalling the existing nodes. If the joining node also performed the inner steps, it would be late to the outer step by the time it took to download and load the checkpoint, reducing the clusters compute utilisation.
- **Custom Int8 All-Reduce Kernel**
    - In our experiments, we found that we are able to perform int8 quantization on the pseudo gradients without any impact on the loss curves. This means that we can reduce the payload size of each outer step all-reduce by 4x if we communicate the pseudo-gradients in int8 instead of fp32.
    - However, we need to accumulate the reduce in fp32, dequantizing and re-quantizing intermediate results during the all-reduce. This is not supported by any collective communication libraries.
    - We thus implemented our own fully pipelined ring-reduce kernel in C++ which is JIT compiled as a custom operator using the torch library.
    - However, with the amount of quantization work we needed to perform, using the torch ops (`quantize_per_tensor`, `scatter_add`, `index`, etc) was too slow, resulting in underutilisation of our target network bandwidth of 4 Gbps.
    - We thus implemented our own multithreaded uint8 ops in C++  to perform the quantization and dequantization operations, improving the quantization speed by more than 60x.
- **Maximising bandwidth utilization:**
    - By sharding our DiLoCo pseudo-gradients in a node, we can maximise network bandwidth utilization by opening multiple connections at the same time when performing the all-reduce. This yielded a transfer speed improvement of 8x on some nodes.
    - Relying on the public IP forward resulted in poor or unstable p2p bandwidth on some compute providers. To mitigate this, we employ VPN technology to optimize peer-to-peer connections between nodes, allowing us to better utilize the available internet bandwidth between nodes by modifying the routing of packets through the internet.
    - We’ve improved bandwidth utilization between nodes in similar data center settings by up to 40x compared to our OpenDiLoCo release, achieving up to 4Gb/s connections between data centers across the whole United States.
- **PyTorch FSDP2 / DTensor ZeRO-3 implementation**
    - In order to fit the 10B model training within our given memory resources, we had to do shard the model weights, gradients and optimizer states between intra-node GPUs.
    - We achieved this using the `fully_shard` API from PyTorch FSDP2 which wraps the model parameters as `DTensor`s and registers hooks to schedule all-gather and reduce-scatter on the tensors when they are used. FSDP2 also optimizes the collectives by bucketing the parameters into `FSDPParamGroup`s. This allows us to execute the collectives on larger tensors, improving protocol-to-payload ratio and improving the overlap from pipelining. We employ the same trick for our pseudo-gradients, bucketing them by layer.
- **CPU Off-Loading**
    - Our Diloco optimizer does not add any GPU overhead. All the tensors required by the Diloco optimizer are offloaded to CPU memory.
    - Since we only perform a global sync every hundreds of steps, the reduced speed of copying and calculating the pseudo-gradient on cpu is negligible relative to the time to execute the inner steps and all-reduce.

A research paper about the framework and our INTELLECT-1 10B experiment is coming soon.

## Getting Started

1. Install `uv`:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.cargo/env
```

2. Set up the environment:
```bash
uv venv
source .venv/bin/activate
uv sync --extra all
uv pip install flash-attn --no-build-isolation
git submodule update --init --recursive
```

3. Log into Hugging Face:
prime uses gated models tokenizers [mistralai/Mistral-7B-v0.1](https://huggingface.co/mistralai/Mistral-7B-v0.1), [meta-llama/Llama-2-7b-hf](https://huggingface.co/meta-llama/Llama-2-7b-hf) and pulls the [C4:en](https://huggingface.co/datasets/allenai/c4) dataset subset. It is required to request access to the models then log into Hugging Face with a read token to begin training.
```bash
huggingface-cli login
```


### Quick Check

Verify your setup:

```bash
ZERO_BAND_LOG_LEVEL=DEBUG torchrun --nproc_per_node=2 src/zeroband/train.py @configs/debug/normal.toml
```

## Usage

### Running DiLoCo

To test DiLoCo locally you can use the helper script `scripts/simulatsimulate_multi_nodee_mutl.sh` 

```bash
# Using 4 GPUs
ZERO_BAND_LOG_LEVEL=DEBUG ./scripts/simulate_multi_node_diloco.sh 2 2 src/zeroband/train.py @configs/debug/diloco.toml

# Using 2 GPUs
ZERO_BAND_LOG_LEVEL=DEBUG ./scripts/simulate_multi_node_diloco.sh 2 1 src/zeroband/train.py @configs/debug/diloco.toml
```

> **Note:** Single GPU setups are currently not supported due to an FSDP implementation bug.

### Running Tests

Ensure you have at least two GPU to run the full test suite:
```bash
uv run pytest
```
## Environment variables
### Global Store Initialization
| Environment Variable  | Description                                      | Default Value |
|-----------------------|--------------------------------------------------|---------------|
| `GLOBAL_UNIQUE_ID`    | Unique identifier worker in global store.        | `None`  |
| `GLOBAL_ADDR`         | IP Address of the global store                   | `None`  |
| `GLOBAL_PORT`         | Port number of the global store.                 | `None` |
| `GLOBAL_WORLD_SIZE`   | The size of the global process group.            | `1` |
| `GLOBAL_RANK`         | Rank of the process in the global process group. | `0` |

### Elastic Device Mesh Configuration
| Environment Variable  | Description                                      | Default Value |
|-----------------------|--------------------------------------------------|---------------|
| `ZERO_BAND_LOG_LEVEL` | Enable debug mode for loge | `False` |
| `ZERO_BAND_GLOBAL_STORE_TIMEOUT_SECONDS` | Number of seconds before the global store operations timeout | `300` |
| `ZERO_BAND_GLOBAL_PG_TIMEOUT_SECONDS` | Number of seconds before the global process group operations timeout | `600` |
| `ZERO_BAND_GLOBAL_STORE_POLLING_INTERVAL_SECONDS` | Number of seconds between polls to the store when waiting for values | `0.1` |
| `ZERO_BAND_EDM_HEARTBEAT_INTERVAL_SECONDS` | Interval in seconds between heartbeats | `2` |
| `ZERO_BAND_EDM_HEARTBEAT_TIMEOUT_SECONDS` | Time in seconds after which a node is considered dead if no heartbeat is received | `10` |
| `ZERO_BAND_LIVE_RECO_PORT` | Port number for the live recovery server | random |  
| `ZERO_BAND_LIVE_RECO_ADDR` | IP Address for the live recovery server | `localhost` |  

## Troubleshooting

If you encounter any dataset loading errors at the beginning of training, try setting:

```bash
export HF_HUB_ETAG_TIMEOUT=500
```

## Pre-downloading datasets

To avoid potential HTTP 443 errors that can occur when streaming datasets from the Hugging Face hub during training crashing your run, you can pre-download the dataset using our custom script. This script efficiently downloads the required files for your specific data rank and world size configuration, utilizing multithreading for faster downloads.

### Prerequisites

Requirements:
- `aria2c` (recommended) or `wget` (fallback) installed on your system
  Note: `wget` is included by default in Ubuntu 22.04 LTS desktop installations

- To install aria2c on Ubuntu 22.04 LTS, run:
```bash
sudo apt install aria2
```

### Usage

Run the `subset_data.py` script with the appropriate parameters:

```bash
python scripts/subset_data.py --dataset_name <dataset_names> --data_world_size <world_size> --data_rank <rank> [--threads <num_threads>] [--dataset_ratio <ratios>]
```

Example for downloading files for data rank 5 in a training setup with data world size of 12, using the configuration from 10B/H100.toml:

```bash
python scripts/subset_data.py \
  --dataset_name PrimeIntellect/fineweb-edu,PrimeIntellect/fineweb,PrimeIntellect/StackV1-popular,PrimeIntellect/dclm-baseline-1.0-parquet,PrimeIntellect/open-web-math \
  --data_world_size 12 \
  --data_rank 5 \
  --threads 8 \
  --dataset_ratio 55:10:20:10:5
```

### Parameters

- `--dataset_name`: Comma-separated list of dataset names on Hugging Face
- `--data_world_size`: Total number of data ranks in your training setup
- `--data_rank`: The specific rank for which to download data (0 to data_world_size - 1)
- `--threads`: (Optional) Number of concurrent download threads (default is 4)
- `--filter`: (Optional) Comma-separated list of strings to filter file names
- `--dry_run`: (Optional) If set, the script will list files without downloading them
- `--max_shards`: (Optional) Maximum number of shards to download (default is 1000)
- `--dataset_ratio`: (Optional) Colon-separated list of ratios for each dataset

### Notes

- The script automatically creates a `datasets` directory in your project root and organizes downloads within it.
- It uses multithreading to download files concurrently, significantly speeding up the process.
- The script automatically selects files based on your data rank and world size, ensuring each rank downloads only its required subset of data.
- It uses aria2c for efficient downloads when available, falling back to wget if aria2c is not installed.
- Large files are handled automatically without any additional setup.
- The script supports multiple datasets and maintains the ratios specified in your configuration.

### Output

The downloaded datasets will be organized in the following structure:

```
prime/
├── datasets/
│   ├── fineweb-edu/
│   ├── fineweb/
│   ├── StackV1-popular/
│   ├── dclm-baseline-1.0-parquet/
│   └── open-web-math/
```
By using this script, you can ensure that all necessary data is downloaded before starting your training process, reducing the likelihood of interruptions due to streaming issues during training. The multithreaded approach allows for faster downloads, making the data preparation process more efficient.