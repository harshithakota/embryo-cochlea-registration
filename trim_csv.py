import pandas as pd
from pathlib import Path

# =========================
# CONFIG
# =========================
INPUT = "data/Cochlea_3D_TIC.csv"
OUT_DIR = Path("data/trimmed_csvs_0_100")
OUT_DIR.mkdir(parents=True, exist_ok=True)

BASE_COLS = ["x", "y", "tissue_id"]
CHUNK_SIZE = 1_000_000

START = 0
END = 100  

# =========================
# DISCOVER m/z COLUMNS
# =========================
print("Reading CSV header...")
cols = pd.read_csv(INPUT, nrows=0).columns
mz_cols = [c for c in cols if c.startswith("m.z.")]

if END is None:
    mz_cols = mz_cols[START:]
else:
    mz_cols = mz_cols[START:END]

print(f"Saving CSVs starting from index {START}")
print(f"Total m/z values to process: {len(mz_cols)}")

# =========================
# PROCESS EACH m/z
# =========================
for mz in mz_cols:
    out_csv = OUT_DIR / f"Cochlea_3D_{mz}.csv"
    usecols = BASE_COLS + [mz]

    print(f"\nProcessing {mz}")
    first = True

    reader = pd.read_csv(
        INPUT,
        usecols=usecols,
        chunksize=CHUNK_SIZE
    )

    for i, chunk in enumerate(reader):
        chunk.to_csv(
            out_csv,
            mode="w" if first else "a",
            header=first,
            index=False
        )
        first = False

        if i % 10 == 0:
            print(f"  wrote {i * CHUNK_SIZE:,} rows")

    print(f"Finished {mz}")

print("\n✅ Done saving remaining m/z CSVs")
