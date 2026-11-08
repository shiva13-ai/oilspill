# oilspill (MarineGuard)

A Flask app and a set of Python scripts for spotting possible oil spills from ship (AIS) data and Sentinel-1 radar imagery. The web UI is branded "MarineGuard".

## What it does

1. You upload an AIS CSV file and enter a vessel's MMSI number in the web app.
2. `anamoly.py` filters that vessel's rows, trains an LSTM on its track (LAT, LON, Heading -> SOG, COG) and flags points where the prediction error is an outlier (IQR rule, 1.5 x IQR). The result is a list of SOG and COG outlier points with position and time.
3. `evaluate.py` takes those outlier points and, for each one, calls `process_sar_and_predict` in `predict.py`.
4. `predict.py` requests a Sentinel-1 GRD image (VV polarization) from Sentinel Hub for a box of +/- 0.1 degrees around the point and +/- 2 days around the timestamp. The image is resized to 256x256 and passed to a pre-trained U-Net (`unet_final_model.keras`). It reports "Oil Spill Detected" if the mean of the predicted mask is above 0.5, and saves `output/grd_image.png` and `output/prediction.png`.

## Tech stack

- Python, Flask
- pandas, NumPy, scikit-learn (MinMaxScaler)
- TensorFlow / Keras (LSTM, U-Net)
- sentinelhub, rasterio, matplotlib
- HTML templates with Bootstrap

## Project structure

```
map.py          Flask app (routes, CSV upload)
anamoly.py      LSTM-based anomaly detection on AIS data
predict.py      Sentinel-1 download + U-Net prediction
evaluate.py     Runs predict.py on the detected outliers
templates/      HTML pages
static/         CSS, JS, images, sample AIS CSVs in static/uploads/
```

## Getting started

There is no requirements file in the repo. Install the libraries the code imports:

```
pip install flask pandas numpy scikit-learn tensorflow matplotlib sentinelhub rasterio joblib
```

Run the web app:

```
python map.py
```

It starts in debug mode (Flask default is http://127.0.0.1:5000). Open the `/working` page, upload a CSV, then enter an MMSI.

AIS CSVs are expected to have these columns: `MMSI`, `BaseDateTime`, `LAT`, `LON`, `SOG`, `COG`, `Heading`. A few sample files are in `static/uploads/`.

## Things to know

- The trained model file `unet_final_model.keras` is not in the repo. `predict.py` looks for it in the working directory, so you need to supply your own.
- `predict.py` needs a Sentinel Hub client ID and secret. They are currently hard-coded in the file; they should be moved to environment variables or a config file and rotated.
- `map.py` sets a placeholder Flask `secret_key` in the source.
- The Flask app only runs the anomaly detection step. `evaluate.py` (the SAR step) is not called from `map.py`; it has to be run separately with the result from `process_ais_data`.
- `predict.py` expects the timestamp as a `"YYYY-MM-DD"` string.
- Several pages linked from the templates (for example `/team` and `/quote`) have no template file in the repo.
- The marketing text and numbers on the landing page are template content, not results from this code.
