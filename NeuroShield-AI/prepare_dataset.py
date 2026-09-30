from pathlib import Path
import io
import zipfile
import urllib.request
import pandas as pd

BASE = Path(__file__).parent
DATA_DIR = BASE / "data"
OUTPUT = DATA_DIR / "messages.csv"

DATA_DIR.mkdir(exist_ok=True)

URL = "https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip"

print("=" * 60)
print("NEUROSHIELD AI - DATASET PREPARATION")
print("=" * 60)

print("\n[+] Downloading official UCI SMS Spam Collection...")

with urllib.request.urlopen(URL) as response:
    zip_bytes = response.read()

print(f"[+] Downloaded {len(zip_bytes):,} bytes")

with zipfile.ZipFile(io.BytesIO(zip_bytes)) as archive:

    print(f"[+] Archive files: {archive.namelist()}")

    with archive.open("SMSSpamCollection") as file:

        df = pd.read_csv(
            file,
            sep="\t",
            header=None,
            names=["original_label", "text"],
            encoding="utf-8",
        )

print(f"[+] Original dataset size: {len(df)}")

# Convert official UCI labels:
# ham  -> 0
# spam -> 1

df["label"] = (
    df["original_label"]
    .astype(str)
    .str.lower()
    .str.strip()
    .map({
        "ham": 0,
        "spam": 1,
    })
)

# Basic cleaning

df["text"] = (
    df["text"]
    .astype(str)
    .str.strip()
)

df = df.dropna(
    subset=["text", "label"]
)

df = df[
    df["text"] != ""
]

df = df.drop_duplicates(
    subset=["text"]
)

df["label"] = df["label"].astype(int)

# Keep only columns required by NeuroShield

df = df[
    ["text", "label"]
]

# Shuffle reproducibly

df = df.sample(
    frac=1,
    random_state=42,
).reset_index(drop=True)

# Save

df.to_csv(
    OUTPUT,
    index=False,
    encoding="utf-8",
)

print("\n" + "=" * 60)
print("DATASET READY")
print("=" * 60)

print(f"\n[+] Saved to: {OUTPUT}")
print(f"[+] Final dataset size: {len(df)}")

print("\nCLASS DISTRIBUTION")
print("-" * 40)

print(df["label"].value_counts())

print("\nLabel meaning:")
print("0 = HAM / NORMAL")
print("1 = SPAM / SUSPICIOUS")

print("\n[+] NeuroShield dataset preparation complete.")