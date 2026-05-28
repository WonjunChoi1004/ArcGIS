import pandas as pd
from pathlib import Path
PROJECT_ROOT = next((p for p in Path(__file__).resolve().parents if (p / ".project-root").exists()), Path(__file__).resolve().parent)

# === FILES ===
landslide_file = f"{PROJECT_ROOT}/LandslideData/All_Processed_With_Soil_Code.csv"
depth_file = f"{PROJECT_ROOT}/Buncombe_Soil_Depth_Summary.csv"
output_file = f"{PROJECT_ROOT}/LandslideData/All_With_SoilDepth.csv"

# === Load data ===
df = pd.read_csv(landslide_file)
depth_df = pd.read_csv(depth_file)

# === Clean up column headers and whitespace ===
depth_df.columns = depth_df.columns.str.strip()
depth_df["Map unit symbol"] = depth_df["Map unit symbol"].astype(str).str.strip()
df["MUSYM"] = df["MUSYM"].astype(str).str.strip()

# === Merge on MUSYM ===
merged = df.merge(
    depth_df[["Map unit symbol", "Rating (centimeters)"]],
    left_on="MUSYM",
    right_on="Map unit symbol",
    how="left"
)

# === Rename and drop helper column ===
merged.rename(columns={"Rating (centimeters)": "Soil_Depth_cm"}, inplace=True)
merged.drop(columns=["Map unit symbol"], inplace=True)

# === Reorder columns: Insert Soil_Depth_cm after MUSYM ===
cols = merged.columns.tolist()
if "MUSYM" in cols and "Soil_Depth_cm" in cols:
    musym_index = cols.index("MUSYM")
    # Remove then insert soil depth right after MUSYM
    cols.remove("Soil_Depth_cm")
    cols.insert(musym_index + 1, "Soil_Depth_cm")
    merged = merged[cols]

# === Save to CSV ===
merged.to_csv(output_file, index=False)
print(f"📦 Dataset saved with Soil_Depth_cm after MUSYM → {output_file}")
