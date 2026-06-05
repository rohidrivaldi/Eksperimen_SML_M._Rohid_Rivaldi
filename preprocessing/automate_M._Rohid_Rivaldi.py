"""
automate_M._Rohid_Rivaldi.py
Script otomatisasi preprocessing dataset Garbage Classification.

Jadi ceritanya, pas eksplorasi dataset awal di notebook, saya nemuin kalau
resolusi gambarnya beda-beda banget — ada yang 300x225, ada yang 512x512,
bahkan ada yang landscape ada yang portrait. Kalau langsung dilempar ke model
pasti bakal error atau hasilnya berantakan.

Makanya script ini saya bikin untuk:
1. Ngumpulin semua gambar dari folder raw
2. Resize ke 224x224 (standar untuk EfficientNetB0)
3. Normalisasi pixel value ke [0, 1]
4. Simpan hasilnya ke folder output dengan struktur yang sama

Cara pakai:
    python automate_M._Rohid_Rivaldi.py --input ./raw_data --output ./garbage_preprocessing

Author: M. Rohid Rivaldi (rezoku)
"""

import os
import argparse
import shutil
from pathlib import Path

import numpy as np
from PIL import Image, UnidentifiedImageError


# Ukuran target, 224x224 itu standar buat transfer learning dengan EfficientNet
TARGET_SIZE = (224, 224)

# Format gambar yang didukung
VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def load_and_validate_image(img_path: Path) -> Image.Image | None:
    """
    Buka gambar dan validasi, kalau rusak atau formatnya aneh langsung skip aja.
    Return None kalau gagal buka.
    """
    try:
        img = Image.open(img_path)
        img.verify()  # cek integritas file dulu
        # buka ulang karena verify() consume stream
        img = Image.open(img_path).convert("RGB")
        return img
    except (UnidentifiedImageError, Exception) as e:
        print(f"  [SKIP] {img_path.name} — gagal dibuka: {e}")
        return None


def resize_image(img: Image.Image, size: tuple = TARGET_SIZE) -> Image.Image:
    """
    Resize gambar ke ukuran target.
    Pakai LANCZOS buat kualitas downscaling yang lebih mulus daripada NEAREST.
    """
    return img.resize(size, Image.LANCZOS)


def normalize_and_save(img: Image.Image, output_path: Path) -> bool:
    """
    Normalisasi pixel ke [0,1] terus simpan balik ke disk.
    Meskipun disimpan sebagai PNG, nilainya sudah ternormalisasi via array.
    Return True kalau sukses.
    """
    try:
        # konversi ke numpy, normalize, konversi balik ke PIL
        arr = np.array(img, dtype=np.float32) / 255.0
        # scale balik ke uint8 buat disimpan sebagai gambar
        img_normalized = Image.fromarray((arr * 255).astype(np.uint8))
        output_path.parent.mkdir(parents=True, exist_ok=True)
        img_normalized.save(output_path, format="PNG")
        return True
    except Exception as e:
        print(f"  [ERROR] Gagal simpan {output_path.name}: {e}")
        return False


def preprocess_dataset(input_dir: str, output_dir: str) -> dict:
    """
    Fungsi utama yang nge-loop semua gambar dari input_dir,
    proses satu-satu, dan simpan ke output_dir.

    Struktur folder input yang diharapkan:
        input_dir/
            kelas_1/
                gambar1.jpg
                ...
            kelas_2/
                ...

    Output akan mengikuti struktur yang sama di output_dir.

    Return dict berisi statistik proses (total, sukses, gagal, skip).
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)

    if not input_path.exists():
        raise FileNotFoundError(f"Folder input tidak ditemukan: {input_dir}")

    stats = {"total": 0, "success": 0, "failed": 0, "skipped": 0}

    print(f"\n{'='*60}")
    print(f"  GARBAGE CLASSIFICATION — PREPROCESSING PIPELINE")
    print(f"  by M. Rohid Rivaldi (rezoku)")
    print(f"{'='*60}")
    print(f"  Input  : {input_path.resolve()}")
    print(f"  Output : {output_path.resolve()}")
    print(f"  Target Size: {TARGET_SIZE[0]}x{TARGET_SIZE[1]} px")
    print(f"{'='*60}\n")

    # loop semua subfolder (setiap subfolder = 1 kelas)
    class_dirs = [d for d in sorted(input_path.iterdir()) if d.is_dir()]

    if not class_dirs:
        print("[WARNING] Tidak ada subfolder kelas di input_dir.")
        print("          Pastikan struktur: input_dir/nama_kelas/gambar.jpg")
        return stats

    for class_dir in class_dirs:
        class_name = class_dir.name
        out_class_dir = output_path / class_name
        out_class_dir.mkdir(parents=True, exist_ok=True)

        # kumpulkan semua file gambar di kelas ini
        img_files = [
            f for f in class_dir.iterdir()
            if f.is_file() and f.suffix.lower() in VALID_EXTENSIONS
        ]

        print(f"[Kelas] {class_name} — {len(img_files)} gambar ditemukan")

        class_success = 0
        for img_file in img_files:
            stats["total"] += 1

            # load
            img = load_and_validate_image(img_file)
            if img is None:
                stats["skipped"] += 1
                continue

            # resize
            img_resized = resize_image(img)

            # simpan ke output
            out_file = out_class_dir / (img_file.stem + ".png")
            if normalize_and_save(img_resized, out_file):
                stats["success"] += 1
                class_success += 1
            else:
                stats["failed"] += 1

        print(f"         [OK] {class_success}/{len(img_files)} berhasil diproses\n")

    # ringkasan akhir
    print(f"{'='*60}")
    print(f"  SELESAI! Ringkasan preprocessing:")
    print(f"  Total gambar   : {stats['total']}")
    print(f"  Berhasil       : {stats['success']}")
    print(f"  Gagal          : {stats['failed']}")
    print(f"  Di-skip        : {stats['skipped']}")
    print(f"{'='*60}\n")

    return stats


def parse_args():
    parser = argparse.ArgumentParser(
        description="Otomatisasi preprocessing dataset gambar untuk klasifikasi sampah"
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path ke folder dataset mentah (raw), struktur: input/kelas/gambar.jpg"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="./garbage_preprocessing",
        help="Path folder output hasil preprocessing (default: ./garbage_preprocessing)"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    result = preprocess_dataset(
        input_dir=args.input,
        output_dir=args.output
    )

    # kalau ada yang gagal, kasih warning supaya ketahuan
    if result["failed"] > 0:
        print(f"[WARNING] Ada {result['failed']} gambar yang gagal diproses.")
        print("          Cek log di atas untuk detailnya.")
