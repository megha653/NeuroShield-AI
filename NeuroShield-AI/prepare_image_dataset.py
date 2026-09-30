from pathlib import Path
from datasets import load_dataset
from PIL import Image
import shutil

BASE = Path(__file__).parent
OUTPUT = BASE / "image_data"

REAL_DIR = OUTPUT / "real"
FAKE_DIR = OUTPUT / "fake"

REAL_LIMIT = 2000
FAKE_LIMIT = 2000

# Remove incomplete dataset from previous stopped run
if OUTPUT.exists():
    print("[+] Removing incomplete previous image dataset...")
    shutil.rmtree(OUTPUT)

REAL_DIR.mkdir(parents=True, exist_ok=True)
FAKE_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 60)
print("NEUROSHIELD AI - IMAGE DATASET PREPARATION")
print("=" * 60)

print("\nTarget:")
print(f"Real images: {REAL_LIMIT}")
print(f"Fake images: {FAKE_LIMIT}")
print(f"Total:       {REAL_LIMIT + FAKE_LIMIT}")

print("\n[+] Opening dataset in streaming mode...")

dataset = load_dataset(
    "Sowaiba01/Deepfake",
    split="train",
    streaming=True
)

real_count = 0
fake_count = 0

print("\n[+] Collecting balanced image subset...")

for sample in dataset:

    if real_count >= REAL_LIMIT and fake_count >= FAKE_LIMIT:
        break

    image = sample["image"]
    label = sample["label"]

    if not isinstance(image, Image.Image):
        image = Image.open(image)

    image = image.convert("RGB")

    label_text = str(label).lower().strip()

    # Dataset mapping used by this preparation pipeline:
    # 0 = fake
    # 1 = real

    if label_text in {"0", "fake"}:

        if fake_count >= FAKE_LIMIT:
            continue

        output_path = FAKE_DIR / f"fake_{fake_count:04d}.jpg"

        image.save(
            output_path,
            "JPEG",
            quality=85
        )

        fake_count += 1

    elif label_text in {"1", "real"}:

        if real_count >= REAL_LIMIT:
            continue

        output_path = REAL_DIR / f"real_{real_count:04d}.jpg"

        image.save(
            output_path,
            "JPEG",
            quality=85
        )

        real_count += 1

    total = real_count + fake_count

    if total > 0 and total % 250 == 0:
        print(
            f"[+] Collected {total}/4000 "
            f"(Real: {real_count}, Fake: {fake_count})"
        )


print("\n" + "=" * 60)
print("IMAGE DATASET READY")
print("=" * 60)

print(f"Real images: {real_count}")
print(f"Fake images: {fake_count}")
print(f"Total:       {real_count + fake_count}")

if real_count != REAL_LIMIT or fake_count != FAKE_LIMIT:
    print("\n[!] WARNING")
    print("Requested balanced subset could not be fully collected.")
else:
    print("\n[+] Balanced 4,000-image dataset created successfully.")

print(f"\nSaved to: {OUTPUT}")