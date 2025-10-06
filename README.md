# Droughts-Predictions-with-Machine-Learning

This repository contains code and documentation for a machine learning project aimed at predicting drought conditions using meteorological data. The project involves data preprocessing, exploratory data analysis, model training, and evaluation.

## Introduction to the Dataset
### Context and Introduction  
The **US Drought & Meteorological Data** dataset, available on [Kaggle](https://www.kaggle.com/datasets/cdminix/us-drought-meteorological-data/data), combines meteorological indicators with drought severity levels from the **U.S. Drought Monitor**. It was created to support data-driven drought prediction by linking weather conditions with expert-assessed drought categories (D0–D4) and no drought at all. Each observation corresponds to a specific county in the United States, identified by its FIPS code, and includes various climatic features, daily obtained, such as temperature, precipitation, humidity, wind speed, and soil moisture, which will be explained later.

The dataset integrates information from the **National Drought Mitigation Center (NDMC)**, **NOAA**, and **USDA**, which produce weekly drought classifications based on multiple climate variables such as precipitation, temperature, and soil moisture.  
Meteorological data were aggregated and aligned with these drought severity labels to enable **machine learning applications in climate and environmental monitoring**.

The dataset is already divided into three subsets:
- **Training Set:** Contains historical data from 2000 to 2009, used to train machine learning models (it represents around 47% of the all dataset).  
- **Validation Set:** Contains data from 2010 to 2011, used for hyperparameter tuning and model selection (it represents around 10% of the all dataset).  
- **Test Set:** Contains data from 2012 to 2020, used to test the model accuracy prediction (it represents around 43% of the all dataset).

All the subsets contain the same feature structure and data organization. For the context of this project, as explained later, it will be used this splitting format.
  
**Application Domain:** This dataset belongs to the domain of **climate science and environmental modeling**, with key applications in:
- Drought risk assessment and early warning systems  
- Agricultural and water resource management  
- Climate variability analysis  

### Features Overview 
All dataset contains **19 meteorological indicators**, a **date column**, a **county identifier (`fips`)**, and a **target variable (`score`)** representing drought severity.  

- **Total entries:** 23841468 - *19300680 (training dataset), 2268840 (validation dataset), 2271948 (test dataset)*
- **Total columns:** 21  
- **Temporal coverage:** Daily observations across multiple U.S. counties  
- **Target variable:** `score` — representing drought severity intensity (ordinal numeric scale)  

| Feature | Description | Type | Unit |
|----------|--------------|------|------|
| `fips` | FIPS code identifying the USA county | int64 | – |
| `date` | Observation date | object | YYYY-MM-DD |
| `PRECTOT` | Total Precipitation | float64 | mm/day |
| `PS` | Surface Pressure | float64 | kPa |
| `QV2M` | Specific Humidity at 2 Meters | float64 | g/kg |
| `T2M` | Air Temperature at 2 Meters | float64 | °C |
| `T2MDEW` | Dew/Frost Point Temperature at 2 Meters | float64 | °C |
| `T2MWET` | Wet Bulb Temperature at 2 Meters | float64 | °C |
| `T2M_MAX` | Maximum Temperature at 2 Meters | float64 | °C |
| `T2M_MIN` | Minimum Temperature at 2 Meters | float64 | °C |
| `T2M_RANGE` | Temperature Range at 2 Meters | float64 | °C |
| `TS` | Earth Skin Temperature | float64 | °C |
| `WS10M` | Wind Speed at 10 Meters | float64 | m/s |
| `WS10M_MAX` | Maximum Wind Speed at 10 Meters | float64 | m/s |
| `WS10M_MIN` | Minimum Wind Speed at 10 Meters | float64 | m/s |
| `WS10M_RANGE` | Wind Speed Range at 10 Meters | float64 | m/s |
| `WS50M` | Wind Speed at 50 Meters | float64 | m/s |
| `WS50M_MAX` | Maximum Wind Speed at 50 Meters | float64 | m/s |
| `WS50M_MIN` | Minimum Wind Speed at 50 Meters | float64 | m/s |
| `WS50M_RANGE` | Wind Speed Range at 50 Meters | float64 | m/s |
| `score` | Drought severity indicator (target variable) | float64  | – |

### Output Variable  

- **Variable:** `score`  
- **Type:** Ordinal categorical variable (`D0`–`D4`) and no drought class (`0`)
- **Meaning:** Represents drought severity, from Abnormally Dry (D0) to Exceptional Drought (D4) and no drought shown at all.

### Learning Task Definition  
The objective of this study is to **predict drought severity levels** using meteorological and climatic indicators.  
This task is formulated as an **ordinal classification problem**, since the drought classes (`D0`–`D4`) and no drought class `0` represent an ordered sequence of severity levels.

## Preprocessing 
Before conducting exploratory analysis, the dataset will be **reprocessed and preprocessed** to make it suitable for machine learning.  
The original dataset contains multiple counties, potential missing values, and temporal dependencies that require careful handling. Considering this, the following preprocessing steps will be undertaken:
- **County Selection:** For simplicity and computational efficiency, the analysis will focus on a single county (identified by its `fips` code). This allows for consistent temporal modeling without the added complexity of spatial variability across counties. The selected county should have a representative distribution of drought classes.
- **Missing Value Handling:** Any missing values in the score target will be removed and, to avoid missing information regarding these observations, new features will be created by computing **rolling averages or sums** over the **past 180 days** for some meteorological variables.
- **Date Handling:** The `date` column will be converted to a datetime format to facilitate temporal analysis and feature engineering.
- **Feature Scaling:** Meteorological features will be normalized to ensure they are on comparable scales, which is important for many machine learning algorithms.