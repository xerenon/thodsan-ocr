from thaiocr.data.prepare import split_by_group


def test_split_no_group_overlap():
    # 10 ตัวอย่าง จาก 5 กลุ่ม (กลุ่มละ 2)
    items = [{"id": i, "group": f"g{i // 2}"} for i in range(10)]
    train, val, test = split_by_group(items, val_frac=0.2, test_frac=0.2, seed=42)
    g = lambda xs: {x["group"] for x in xs}
    # ไม่มีกลุ่มไหนโผล่ข้าม split
    assert g(train).isdisjoint(g(val))
    assert g(train).isdisjoint(g(test))
    assert g(val).isdisjoint(g(test))


def test_split_covers_all_items():
    items = [{"id": i, "group": f"g{i // 2}"} for i in range(10)]
    train, val, test = split_by_group(items, val_frac=0.2, test_frac=0.2, seed=42)
    assert len(train) + len(val) + len(test) == 10


def test_split_deterministic_with_seed():
    items = [{"id": i, "group": f"g{i % 4}"} for i in range(20)]
    a = split_by_group(items, 0.2, 0.2, seed=1)
    b = split_by_group(items, 0.2, 0.2, seed=1)
    assert [x["id"] for x in a[0]] == [x["id"] for x in b[0]]
