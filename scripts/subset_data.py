import argparse
import logging
import os
import subprocess
from pathlib import Path
import requests
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
ch.setFormatter(formatter)
logger.addHandler(ch)

def get_hf_token():
    token_path = os.path.expanduser("~/.cache/huggingface/token")
    if os.path.exists(token_path):
        with open(token_path, "r") as f:
            return f.read().strip()
    return None

def get_dataset_files(repo_id: str, hf_token: str = None) -> list:
    api_url = f"https://huggingface.co/api/datasets/{repo_id}/tree/main"
    headers = {"Authorization": f"Bearer {hf_token}"} if hf_token else {}
    
    all_files = []
    
    def traverse_files(path=""):
        nonlocal api_url, headers, all_files
        url = f"{api_url}/{path}"
        response = requests.get(url, headers=headers)
        if response.status_code != 200:
            logger.error(f"Failed to fetch file list: {response.status_code}")
            return
        
        files = response.json()
        for file in files:
            if file['type'] == 'file':
                all_files.append(file['path'])
            elif file['type'] == 'directory':
                traverse_files(file['path'])
    
    traverse_files()
    return [f for f in all_files if f.endswith('.parquet')]

def download_file(repo_id: str, file_path: str, output_dir: Path, hf_token: str = None, threads: int = 4):
    url = f"https://huggingface.co/datasets/{repo_id}/resolve/main/{file_path}"
    local_path = output_dir / file_path
    local_path.parent.mkdir(parents=True, exist_ok=True)

    if shutil.which("aria2c"):
        cmd = [
            "aria2c", "--console-log-level=error", "--file-allocation=none",
            f"-x{threads}", f"-s{threads}", "-k1M", "-c",
            url, "-d", str(local_path.parent), "-o", local_path.name
        ]
        if hf_token:
            cmd.insert(1, f"--header=Authorization: Bearer {hf_token}")
    elif shutil.which("wget"):
        cmd = ["wget", "-c", url, "-O", str(local_path)]
        if hf_token:
            cmd.insert(1, f"--header=Authorization: Bearer {hf_token}")
    else:
        raise RuntimeError("Neither aria2c nor wget is available. Please install one of them.")

    subprocess.run(cmd, check=True)
    return file_path

def download_files(repo_id: str, files: list, output_dir: Path, hf_token: str = None, threads: int = 4):
    output_dir.mkdir(parents=True, exist_ok=True)

    with ThreadPoolExecutor(max_workers=threads) as executor:
        future_to_file = {executor.submit(download_file, repo_id, file, output_dir, hf_token, 1): file for file in files}
        for future in as_completed(future_to_file):
            file = future_to_file[future]
            try:
                future.result()
                logger.debug(f"Downloaded {file}")
            except Exception as e:
                logger.error(f"Failed to download {file}: {e}")

def main(args):
    hf_token = get_hf_token()
    if not hf_token:
        logger.warning("No Hugging Face token found. You may need to log in using 'huggingface-cli login'")
    
    dataset_names = args.dataset_name.split(',')
    
    # Create a 'datasets' directory in the project root
    project_root = Path(__file__).resolve().parent.parent
    datasets_dir = project_root / "datasets"
    datasets_dir.mkdir(exist_ok=True)
    
    # Create output directories based on dataset names
    output_dirs = [datasets_dir / name.split('/')[-1] for name in dataset_names]
    
    logger.info(f"Downloads will be saved in the project's 'datasets' directory: {datasets_dir}")
    
    ratios = [1] * len(dataset_names)
    if args.dataset_ratio:
        ratios = [float(r) for r in args.dataset_ratio.split(':')]
        if len(ratios) != len(dataset_names):
            raise ValueError("Number of ratios must match number of datasets")
    
    total_ratio = sum(ratios)
    ratios = [r / total_ratio for r in ratios]
    
    for dataset_name, output_dir, ratio in zip(dataset_names, output_dirs, ratios):
        logger.info(f"Processing dataset: {dataset_name}")
        all_files = get_dataset_files(dataset_name, hf_token)
        logger.debug(f"Total number of files for {dataset_name}: {len(all_files)}")
        
        if args.filter:
            filters = args.filter.split(",")
            data_files = [f for f in all_files if any(filter_str in f for filter_str in filters)]
        else:
            data_files = all_files

        num_files = int(args.max_shards * ratio)
        data_files = data_files[args.data_rank :: args.data_world_size][:num_files]
        logger.debug(f"Files to download for {dataset_name}: {data_files}")
        logger.debug(f"Number of files to download for {dataset_name}: {len(data_files)}")

        if not args.dry_run:
            download_files(dataset_name, data_files, output_dir, hf_token, args.threads)

        logger.info(f"Dataset {dataset_name} downloaded to {output_dir}")
    
    logger.info("Download complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download and process data from the Hugging Face dataset")
    parser.add_argument("--dataset_name", type=str, required=True, help="comma-separated list of dataset names")
    parser.add_argument("--dry_run", action="store_true", help="do not download data")
    parser.add_argument("--filter", type=str, default="", help="search shards by the filter")
    parser.add_argument("--data_rank", type=int, default=0, help="start index")
    parser.add_argument("--data_world_size", type=int, default=4, help="world size")
    parser.add_argument("--max_shards", type=int, default=1000)
    parser.add_argument("--threads", type=int, default=4, help="number of download threads")
    parser.add_argument("--dataset_ratio", type=str, help="colon-separated list of ratios for each dataset")
    args = parser.parse_args()
    main(args)