import random
import os 
from datasets import load_dataset
from tqdm import tqdm

def yield_approximate_portion_file(input_file: str, p: float, output_file: str):
    assert (p > 0 and p <= 1)

    with open(input_file, "r") as f_in, open(output_file, "w") as f_out:
        for line in f_in:
            if random.random() < p:
                f_out.write(line)

    return output_file

def download_data():
    COORD_FILE = "./experiment/data/animalface_coordinates.txt"
    MAPPING_FILE = "./experiment/data/ILSVRC2012_mapping.txt"
    OUT_ROOT = "./experiment/data/animals"

    # ------------------------------------------------------------
    # 1. Extract required synsets from your coordinates file
    # ------------------------------------------------------------
    required_synsets = set()

    with open(COORD_FILE, "r") as f:
        for line in f:
            path = line.strip().split()[0]   # n02089078/n02089078_3613.JPEG
            synset = path.split("/")[0]      # n02089078
            required_synsets.add(synset)

    print(f"Found {len(required_synsets)} required synsets")

    # ------------------------------------------------------------
    # 2. Load label -> synset mapping
    # ------------------------------------------------------------
    id2synset = {}

    with open(MAPPING_FILE, "r") as f:
        for line in f:
            parts = line.strip().split()
            idx = int(parts[0]) - 1   # 🔴 subtract 1 here
            wnid = parts[1]
            id2synset[idx] = wnid

    # ------------------------------------------------------------
    # 3. Stream ImageNet from Hugging Face
    # ------------------------------------------------------------
    dataset = load_dataset(
        "ILSVRC/imagenet-1k",
        split="train",
        streaming=True,
        trust_remote_code=True,
    )

    # ------------------------------------------------------------
    # 4. Extract ONLY required synsets
    # ------------------------------------------------------------
    os.makedirs(OUT_ROOT, exist_ok=True)
    saved_count = 0

    for i, sample in enumerate(tqdm(dataset)):
        label_idx = sample["label"]
        synset = id2synset[label_idx]

        if synset in required_synsets:
            out_dir = os.path.join(OUT_ROOT, synset)
            os.makedirs(out_dir, exist_ok=True)

            out_path = os.path.join(out_dir, f"{synset}_{i:08d}.jpg")
            sample["image"].save(out_path)
            saved_count += 1

    print(f"\nExtracted {saved_count} images into {OUT_ROOT}")