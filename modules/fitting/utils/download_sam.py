import os
import sys
import argparse
import urllib.request
import shutil

# Correct SAM 2.1 Base URLs
# Checkpoints are in the '092824' release folder
CHECKPOINT_BASE_URL = "https://dl.fbaipublicfiles.com/segment_anything_2/092824/"
# Configs are in the main facebookresearch/sam2 repo under configs/sam2.1
CONFIG_BASE_URL = "https://raw.githubusercontent.com/facebookresearch/sam2/main/sam2/configs/sam2.1/"

MODELS = {
    "tiny": {"ckpt": "sam2.1_hiera_tiny.pt", "cfg": "sam2.1_hiera_t.yaml"},
    "small": {"ckpt": "sam2.1_hiera_small.pt", "cfg": "sam2.1_hiera_s.yaml"},
    "base": {"ckpt": "sam2.1_hiera_base_plus.pt", "cfg": "sam2.1_hiera_b+.yaml"},
    "large": {"ckpt": "sam2.1_hiera_large.pt", "cfg": "sam2.1_hiera_l.yaml"},
}


def download_file(url, output_path):
    print(f"Downloading {url}...")
    try:
        # User-Agent header is sometimes required for raw.githubusercontent.com
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req) as response, open(output_path, 'wb') as out_file:
            shutil.copyfileobj(response, out_file)
        print(f"Saved to {output_path}")
    except Exception as e:
        print(f"Error downloading {url}: {e}")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description="Download SAM 2.1 weights and configs.")
    parser.add_argument("--model", type=str, default="large", choices=MODELS.keys(),
                        help="Model size to download.")
    parser.add_argument("--out_dir", type=str, default="checkpoints", help="Directory to save weights.")
    parser.add_argument("--config_dir", type=str, default="configs/sam2.1", help="Directory to save configs.")

    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(args.config_dir, exist_ok=True)

    model_info = MODELS[args.model]

    # 1. Download Checkpoint
    ckpt_name = model_info["ckpt"]
    ckpt_url = CHECKPOINT_BASE_URL + ckpt_name
    out_ckpt = os.path.join(args.out_dir, ckpt_name)

    if not os.path.exists(out_ckpt):
        download_file(ckpt_url, out_ckpt)
    else:
        print(f"Checkpoint {out_ckpt} already exists.")

    # 2. Download Config
    cfg_name = model_info["cfg"]
    cfg_url = CONFIG_BASE_URL + cfg_name
    out_cfg = os.path.join(args.config_dir, cfg_name)

    if not os.path.exists(out_cfg):
        download_file(cfg_url, out_cfg)
    else:
        print(f"Config {out_cfg} already exists.")

    print("\n[Success] SAM 2.1 Assets Ready.")
    print("Run command:")
    print(f'--sam_checkpoint "{out_ckpt}" --sam_config "{out_cfg}"')


if __name__ == "__main__":
    main()