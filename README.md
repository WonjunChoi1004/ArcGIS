# Landslide Prediction in Buncombe County

This project develops a **GIS- and machine learning–based framework** to predict landslide occurrence in Buncombe County, NC.  
It integrates rainfall, slope, soil, and land cover data to create a **real-time hazard prediction system** and supports interpretability through SHAP values and feature importance analysis.  

The work also forms the basis for a research paper on localized landslide risk modeling.

---

## Project Overview

- **Goal:** Identify areas at high risk of rainfall-induced landslides in Buncombe County using environmental data and machine learning models.  
- **Methods:**  
  - Static grid creation from DEM and soil shapefiles  
  - Rainfall window aggregation (short- and long-term PRISM data)  
  - Training and evaluation of ML classifiers (Logistic Regression, Random Forest, XGBoost)  
  - Explainable AI (SHAP values, feature importance) for model interpretation  
- **Output:** Hazard probability maps and performance metrics (ROC, confusion matrix, feature rankings).

---

## Project Structure

- `build_static_grid_from_existing_assets.py` — Builds base grid combining DEM, slope, soil, and depth data  
- `RainfallBuncombe.py` — Processes PRISM rainfall data for Buncombe County  
  ![Rainfall](Images/RainfallBuncombe.png)

- `BuncombeLandslideSlopeMap.py` — Combines slope maps with historic landslide points for visualization  
  ![Slope and Historic Landslides](Images/SlopeAndHistoricLandslideBuncombe.png)

- `predict_daily_triple.py` — Generates real-time rainfall–slope–soil predictions on the grid

- `main.py` — Orchestrates the full pipeline: raw inputs → processed features → ML predictions  

- **Model performance outputs:**  
  - ROC Curve  
    ![ROC Curve](Images/ROCCurveRF.png)  
  - Confusion Matrix  
    ![Confusion Matrix](Images/ConfusionMatrixRF.png)  
  - Predicted Landslide Map (Random Forest model)  
    ![Landslide Map](Images/LandslideMapRFBuncombe.png)

---

## Data Sources

- [USGS DEM / Elevation](https://www.usgs.gov/)  
- [NC OneMap Landslide Inventory](https://www.nconemap.gov/)  
- [PRISM Climate Group](https://prism.oregonstate.edu/) — rainfall normals and daily precipitation  
- [SSURGO (NRCS)](https://www.nrcs.usda.gov/) — soil depth and texture data  
- [US Census TIGER/Line Shapefiles](https://www.census.gov/geographies/mapping-files/time-series/geo/tiger-line-file.html) — county boundaries  
- [NLCD Land Cover Database](https://www.mrlc.gov/) — land cover/vegetation classes  

> **Note:** Large rasters and shapefiles (DEM, PRISM, SSURGO) are excluded from this repository.  
> They can be downloaded from the above sources and placed into a local `data/` directory.

---

## Installation

```bash
git clone git@github.com:WonjunChoi1004/ArcGIS.git
cd ArcGIS
pip install -r requirements.txt
```

## How to Run

1. **Build static grid (DEM, slope, soil, land cover):**  
   Run `python "Code/Data processing/Website/build_static_grid_from_existing_assets.py"`

2. **Process rainfall data:**  
   Run `python RainfallBuncombe.py`

3. **Run prediction pipeline:**  
   Run `python main.py`

The pipeline outputs a **GeoDataFrame** containing grid points with environmental predictors and a predicted landslide probability.  

---

## Output

- Unified GeoDataFrame with:  
  - `Slope`  
  - `Rainfall (1-day, 7-day, 30-day)`  
  - `Soil depth`  
  - `Land cover`  
  - `IsLandslide` (binary label for supervised learning)  

- Model performance metrics: ROC AUC, precision, recall, confusion matrix  
- SHAP values and feature importance plots  
- Landslide probability map for Buncombe County  

---

## Applications

- Local hazard mapping and disaster preparedness  
- Real-time risk assessment for emergency managers  
- Academic research on rainfall-triggered landslide prediction  

---

## Author

**Wonjun Choi**  
_Asheville School Research Project_  
_Focus: Geotechnical hazard prediction using GIS and ML_