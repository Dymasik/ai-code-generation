import argparse
from datasets import load_dataset

if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog='Upload DS to Hugging Face')
    parser.add_argument("--local_name", type=str)
    parser.add_argument("--remote_name", type=str)
    args = parser.parse_args()
    dataset = load_dataset(f"src/datasets/target/{args.local_name}")
    dataset.push_to_hub(args.remote_name, private=True)