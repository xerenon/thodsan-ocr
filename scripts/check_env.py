"""ตรวจ environment สำหรับเทรน QLoRA บน Blackwell GPU (เฟส 0 go/no-go)."""
import sys


def main() -> int:
    ok = True

    import torch
    print(f"torch: {torch.__version__}")
    cuda = torch.cuda.is_available()
    print(f"cuda available: {cuda}")
    if not cuda:
        print("FAIL: ไม่เจอ CUDA GPU")
        return 1

    name = torch.cuda.get_device_name(0)
    cap = torch.cuda.get_device_capability(0)
    vram_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
    print(f"gpu: {name}  sm_{cap[0]}{cap[1]}  VRAM={vram_gb:.1f}GB")
    if cap[0] < 9:
        print(f"WARN: compute capability {cap} ต่ำกว่าที่คาด (Blackwell=12.0)")

    # ทดสอบ matmul บน GPU จริง
    x = torch.randn(1024, 1024, device="cuda", dtype=torch.bfloat16)
    y = (x @ x).float().sum().item()
    print(f"bf16 matmul on GPU: OK ({y:.1f})")

    # ทดสอบ bitsandbytes 4-bit (หัวใจของ QLoRA)
    try:
        import bitsandbytes as bnb
        print(f"bitsandbytes: {bnb.__version__}")
        lin = bnb.nn.Linear4bit(1024, 1024, bias=False).cuda()
        out = lin(x)
        print(f"bnb Linear4bit forward: OK (shape={tuple(out.shape)})")
    except Exception as e:
        print(f"FAIL: bitsandbytes 4-bit ใช้ไม่ได้บน GPU นี้: {e}")
        ok = False

    print("\nRESULT:", "PASS ✅ ไปต่อได้" if ok else "FAIL ❌ ต้องแก้ก่อน")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
