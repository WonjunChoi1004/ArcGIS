# Landslide Prediction in Buncombe County

This project uses GIS, rainfall data, and machine learning to predict landslide occurrence in Buncombe County, NC.

## 📁 Project Structure

- `LandslideProcessing.py` — Generates random points, labels landslide and non-landslide data
- `RainfallBuncombe.py` — Processes rainfall data for Buncombe County
- `PRISM_ppt_30yr_normals/` — PRISM rainfall data used in the analysis
- `cb_2022_us_county_5m/` — Shapefiles for masking by county boundary

## 📊 Data Sources

- [USGS Landslide Inventory](https://www.usgs.gov/)
- [PRISM Climate Group](https://prism.oregonstate.edu/)
- [US Census TIGER/Line Shapefiles](https://www.census.gov/geographies/mapping-files/time-series/geo/tiger-line-file.html)

## 🚀 How to Run

1. Install required Python libraries:
pip install geopandas pandas shapely matplotlib

markdown
Copy
Edit

2. Run the processing script:
<pre> ```python def run(): print("Running landslide model") ``` </pre>


shell
Copy
Edit

## 📌 Output

The final dataset contains labeled points (`IsLandslide` = 1 or 0) ready for training classification models.

## 🧠 Author

Wonjun Choi – Asheville School Research Project