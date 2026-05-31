from pathlib import Path
import sys

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import h5py
import numpy as np

path = "data/helmholtz/_test_o1_5_4k_mixed.h5"


def describe_dataset(name, obj):
    indent = "  " * (name.count("/") - 1)

    if isinstance(obj, h5py.Dataset):
        print(f"{indent}- DATASET: {name}")
        print(f"{indent}  shape: {obj.shape}")
        print(f"{indent}  dtype: {obj.dtype}")

        # Show attributes if present
        if obj.attrs:
            print(f"{indent}  attrs:")
            for k, v in obj.attrs.items():
                print(f"{indent}    {k}: {v}")

    elif isinstance(obj, h5py.Group):
        print(f"{indent}- GROUP: {name}")
        if obj.attrs:
            print(f"{indent}  attrs:")
            for k, v in obj.attrs.items():
                print(f"{indent}    {k}: {v}")


with h5py.File(path, "r") as f:
    print("=== FILE STRUCTURE ===")
    f.visititems(describe_dataset)

    print("\n=== TOP-LEVEL KEYS ===")
    for key in f.keys():
        print(f"- {key}")

    print("\n=== SAMPLE CONTENT (idx=0) ===")
    idx = 0

    if "fields" in f:
        src = f["fields"][idx, 0]
        sol = f["fields"][idx, 1]

        print("\nfields:")
        print("  source shape:", src.shape)
        print("  solution shape:", sol.shape)
        print("  source stats:", {
            "min": float(src.min()),
            "max": float(src.max()),
            "mean": float(src.mean()),
            "std": float(src.std()),
        })

        print("  center patch (6x6):")
        print(np.array_str(src[61:67, 61:67], precision=4, suppress_small=True))

    if "labels" in f:
        print("\nlabels:")
        print("  example:", int(f["labels"][idx]))

    if "tensor" in f:
        print("\ntensor:")
        print("  example:", f["tensor"][idx])