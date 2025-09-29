import pandas as pd
import numpy as np

# === File Paths ===
main_csv = "/Users/wonjunchoi/PycharmProjects/ArcGIS/LandslideData/All_With_SoilDepth.csv"
component_path = "/Users/wonjunchoi/PycharmProjects/ArcGIS/NC021/tabular/comp.txt"
chorizon_path = "/Users/wonjunchoi/PycharmProjects/ArcGIS/NC021/tabular/chorizon.txt"
output_csv = "/Users/wonjunchoi/PycharmProjects/ArcGIS/LandslideData/All_With_Soil_FrictionAngle.csv"

# === Load main landslide dataset ===
df = pd.read_csv(main_csv)

# === Load component and chorizon data ===
comp = pd.read_csv(component_path, sep="|", dtype=str, low_memory=False)
ch = pd.read_csv(chorizon_path, sep="|", dtype=str, low_memory=False)

# === Ensure consistent column names ===
comp.columns = comp.columns.str.strip()
ch.columns = ch.columns.str.strip()

# === Get MUKEY → COKEY map ===
comp_subset = comp[["mukey", "cokey"]].dropna()
ch_subset = ch[["cokey", "sandtotal_r", "claytotal_r"]].dropna()

# === Merge to get sand and clay by MUKEY ===
merged = pd.merge(comp_subset, ch_subset, on="cokey")
merged["sandtotal_r"] = pd.to_numeric(merged["sandtotal_r"], errors="coerce")
merged["claytotal_r"] = pd.to_numeric(merged["claytotal_r"], errors="coerce")

# === Average sand/clay values by MUKEY ===
avg_texture = merged.groupby("mukey")[["sandtotal_r", "claytotal_r"]].mean().reset_index()

# === USDA Texture Classification Function ===
def classify_texture(sand, clay):
    if sand > 85 and clay < 10:
        return "Sand"
    elif sand > 70 and clay < 15:
        return "Sandy Loam"
    elif 45 < sand < 80 and 7 < clay < 20:
        return "Loam"
    elif sand < 50 and 0.2 * clay + sand > 60:
        return "Silt Loam"
    elif 27 <= clay <= 40 and 20 <= sand <= 45:
        return "Clay Loam"
    elif clay > 40 and sand < 20:
        return "Silty Clay"
    elif clay > 40 and sand < 15:
        return "Clay"
    else:
        return "Unknown"

# === Assign USDA texture class ===
avg_texture["Texture"] = avg_texture.apply(
    lambda row: classify_texture(row["sandtotal_r"], row["claytotal_r"]), axis=1
)

# === Friction Angle Lookup Table ===
texture_to_phi = {
    "Sand": "30–36",
    "Sandy Loam": "28–34",
    "Loam": "26–32",
    "Silt Loam": "25–31",
    "Clay Loam": "23–29",
    "Silty Clay": "20–26",
    "Clay": "17–24"
}

avg_texture["Friction_Angle_deg"] = avg_texture["Texture"].map(texture_to_phi)

# === Merge friction angle with main dataset using MUKEY ===
df["MUKEY"] = df["MUKEY"].astype(str)
avg_texture["mukey"] = avg_texture["mukey"].astype(str)
final = pd.merge(df, avg_texture[["mukey", "Friction_Angle_deg"]], left_on="MUKEY", right_on="mukey", how="left")
final.drop(columns=["mukey"], inplace=True)

# === Reorder columns: Soil_Depth_cm and Friction_Angle_deg after MUSYM ===
cols = final.columns.tolist()
insert_index = cols.index("MUSYM") + 1
cols.remove("Soil_Depth_cm")
cols.remove("Friction_Angle_deg")
cols.insert(insert_index, "Soil_Depth_cm")
cols.insert(insert_index + 1, "Friction_Angle_deg")
final = final[cols]

# === Save the final dataset ===
final.to_csv(output_csv, index=False)
print(f"✅ Output saved: {output_csv}")