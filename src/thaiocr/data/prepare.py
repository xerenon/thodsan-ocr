"""โหลด iapp/thai_handwriting_dataset และตัด split แบบไม่ให้กลุ่มรั่ว."""
import random
from typing import Any


def split_by_group(
    items: list[dict[str, Any]],
    val_frac: float,
    test_frac: float,
    seed: int = 42,
    group_key: str = "group",
) -> tuple[list, list, list]:
    """แบ่ง train/val/test โดยจัดทั้งกลุ่มไปอยู่ split เดียว (กัน leakage)."""
    groups: dict[str, list] = {}
    for it in items:
        groups.setdefault(it[group_key], []).append(it)

    keys = list(groups.keys())
    random.Random(seed).shuffle(keys)

    n = len(keys)
    n_test = int(n * test_frac)
    n_val = int(n * val_frac)
    test_keys = set(keys[:n_test])
    val_keys = set(keys[n_test:n_test + n_val])

    train, val, test = [], [], []
    for k in keys:
        bucket = test if k in test_keys else val if k in val_keys else train
        bucket.extend(groups[k])
    return train, val, test


def load_and_split(out_dir: str = "data/splits", seed: int = 42):
    """โหลด dataset จริงจาก HF แล้วเซฟ split เป็น parquet. คืนจำนวนต่อ split."""
    import os
    from datasets import load_dataset

    ds = load_dataset("iapp/thai_handwriting_dataset", split="train")
    # ใช้ label_file เป็น group proxy (ตัวบ่งแหล่ง/ผู้เขียน) เพื่อกัน leakage
    # column access (ds["label_file"]) เร็วมาก — ไม่ decode รูปทีละแถวแบบ ds[i]
    labels = ds["label_file"]
    items = [{"idx": i, "group": str(labels[i])} for i in range(len(labels))]
    train, val, test = split_by_group(items, val_frac=0.05, test_frac=0.10, seed=seed)

    os.makedirs(out_dir, exist_ok=True)
    for name, part in [("train", train), ("val", val), ("test", test)]:
        idxs = [x["idx"] for x in part]
        ds.select(idxs).to_parquet(f"{out_dir}/{name}.parquet")
    return {"train": len(train), "val": len(val), "test": len(test)}


if __name__ == "__main__":
    print(load_and_split())
