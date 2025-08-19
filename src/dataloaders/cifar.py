import os, json
import numpy as np
import torch
from torchvision import datasets, transforms
from sklearn.utils import shuffle

def get(seed=0, pc_valid=0.10, path="../dat/", regen=False, num_workers=4, prep_batch_size=512):
    """
    50-task Split-CIFAR variant:
      - Task 0: CIFAR-10 (10-way)
      - Tasks 1..49: CIFAR-100 as 49 binary tasks (2 classes per task), deterministic from `seed`.
    """
    rng = np.random.RandomState(seed)
    size = [3, 32, 32]
    download_path = os.path.join(path, "binary_cifar")
    os.makedirs(download_path, exist_ok=True)

    # Files we expect for 50 tasks
    task_ids = list(range(50))
    expected_files = [
        os.path.join(download_path, f"data{t}{s}{xy}.bin")
        for t in task_ids for s in ["train", "test"] for xy in ["x", "y"]
    ]
    map_path = os.path.join(download_path, "cifar100_binary_map.json")

    mean = [125.3/255, 123.0/255, 113.9/255]
    std  = [63.0/255,  62.1/255,  66.7/255]
    tfm  = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)])

    need_build = regen or any(not os.path.isfile(f) for f in expected_files) or (not os.path.isfile(map_path))

    if need_build:
        # ----- Build all 50 tasks -----
        data = {}

        # Task 0: CIFAR-10 (10-way)
        d10 = {
            "train": datasets.CIFAR10(path, train=True,  download=True, transform=tfm),
            "test":  datasets.CIFAR10(path, train=False, download=True, transform=tfm),
        }
        data[0] = {"name": "cifar10-0", "ncla": 10, "train": {"x": [], "y": []}, "test": {"x": [], "y": []}}
        for s in ["train", "test"]:
            loader = torch.utils.data.DataLoader(d10[s], batch_size=prep_batch_size, shuffle=False, num_workers=num_workers)
            for images, targets in loader:
                data[0][s]["x"].append(images)
                data[0][s]["y"].append(targets)

        # Tasks 1..49: CIFAR-100 as 49 binary tasks
        # Deterministic pairing of 100 classes -> 49 pairs + merge last two into the last pair's positive class
        # Scheme: pair first 98 permuted classes into 49 pairs; classes 98,99 both map to label 1 of task 49
        classes = np.arange(100)
        perm = rng.permutation(classes)

        pairs = perm[:98].reshape(49, 2)          # 49 pairs -> tasks 1..49
        leftovers = perm[98:]                      # 2 classes
        binary_map = {}                            # class_id -> {"task": t, "label": 0/1}

        for t, (a, b) in enumerate(pairs, start=1):
            binary_map[int(a)] = {"task": t, "label": 0}
            binary_map[int(b)] = {"task": t, "label": 1}

        # Merge leftovers into task 49 as label 1 (keeps class balance roughly fine)
        for c in leftovers:
            binary_map[int(c)] = {"task": 49, "label": 1}

        # Initialize task buffers 1..49
        for t in range(1, 50):
            data[t] = {"name": f"cifar100-bin-{t}", "ncla": 2, "train": {"x": [], "y": []}, "test": {"x": [], "y": []}}

        d100 = {
            "train": datasets.CIFAR100(path, train=True,  download=True, transform=tfm),
            "test":  datasets.CIFAR100(path, train=False, download=True, transform=tfm),
        }

        def push_c100(split, images, targets):
            n = targets.numpy()
            keep_mask = np.array([int(c) in binary_map for c in n])
            if not keep_mask.any():
                return
            imgs = images[keep_mask]
            cls  = n[keep_mask]
            tids = np.array([binary_map[int(c)]["task"]  for c in cls])
            labs = np.array([binary_map[int(c)]["label"] for c in cls], dtype=np.int64)
            # scatter to per-task buffers
            for t in range(1, 50):
                mask = (tids == t)
                if mask.any():
                    data[t][split]["x"].append(imgs[mask])
                    data[t][split]["y"].append(torch.from_numpy(labs[mask]))

        for s in ["train", "test"]:
            loader = torch.utils.data.DataLoader(d100[s], batch_size=prep_batch_size, shuffle=False, num_workers=num_workers)
            for images, targets in loader:
                push_c100(s, images, targets)

        # Unify & save tensors
        for t in task_ids:
            for s in ["train", "test"]:
                X = torch.cat(data[t][s]["x"], dim=0)
                y = torch.cat(data[t][s]["y"], dim=0).long()
                data[t][s]["x"], data[t][s]["y"] = X, y
                torch.save(X, os.path.join(download_path, f"data{t}{s}x.bin"))
                torch.save(y, os.path.join(download_path, f"data{t}{s}y.bin"))

        # Persist the mapping for reproducibility
        with open(map_path, "w") as f:
            json.dump({"seed": int(seed), "pairs": pairs.tolist(), "leftovers": leftovers.tolist(), "map": binary_map}, f, indent=2)

    # ----- Load binaries -----
    data = {}
    print("Task order =", task_ids)
    for i, tid in enumerate(task_ids):
        data[i] = {"train": {}, "valid": {}, "test": {}}
        for s in ["train", "test"]:
            data[i][s] = {
                "x": torch.load(os.path.join(download_path, f"data{tid}{s}x.bin")),
                "y": torch.load(os.path.join(download_path, f"data{tid}{s}y.bin")),
            }
        # Infer ncla (10 for task 0, 2 for others)
        ncla = int(len(np.unique(data[i]["train"]["y"].numpy())))
        data[i]["ncla"] = ncla
        data[i]["name"] = "cifar10-0" if i == 0 else f"cifar100-bin-{i}"

    # ----- Validation split -----
    for t in task_ids:
        r = np.arange(data[t]["train"]["x"].size(0))
        r = np.array(shuffle(r, random_state=seed), dtype=int)
        nvalid = int(pc_valid * len(r))
        ivalid = torch.LongTensor(r[:nvalid])
        itrain = torch.LongTensor(r[nvalid:])
        data[t]["valid"] = {
            "x": data[t]["train"]["x"][ivalid].clone(),
            "y": data[t]["train"]["y"][ivalid].clone(),
        }
        data[t]["train"]["x"] = data[t]["train"]["x"][itrain].clone()
        data[t]["train"]["y"] = data[t]["train"]["y"][itrain].clone()

    # ----- Task list & total classes (sum over per-task heads) -----
    taskcla, total = [], 0
    for t in task_ids:
        taskcla.append((t, data[t]["ncla"]))
        total += data[t]["ncla"]
    data["ncla"] = total  # for multi-head setups, this is just a sum across tasks

    return data, taskcla, size
