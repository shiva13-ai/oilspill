import os
import numpy as np
from joblib import load
from datetime import datetime, timedelta
from sentinelhub import SHConfig, BBox, CRS, SentinelHubRequest, DataCollection, MimeType
import rasterio
import matplotlib.pyplot as plt
import tensorflow as tf
output_folder = "./output"
def process_sar_and_predict(lat, lon, timestamp):
    client_id = 'db6f06cb-d661-439b-8942-71691e39c83b'
    client_secret = 'KOZnydqNMtb6i1czu6mLiTXrV1D4u1xe'
    model_path = 'unet_final_model.keras'  # Path to your pre-trained UNet model
    data_folder = './SAR_Images'
    """
    Generate SAR image and predict using UNet model.
    
    Parameters:
        lat (float): Latitude of the location.
        lon (float): Longitude of the location.
        timestamp (str): Timestamp in "YYYY-MM-DD" format.
        client_id (str): Sentinel Hub client ID.
        client_secret (str): Sentinel Hub client secret.
        model_path (str): Path to the pre-trained UNet model.
        data_folder (str): Folder to store the generated SAR image.

    Returns:
        predicted_mask (ndarray or None): Predicted mask from the UNet model.
    """
    def generate_bbox(lat, lon, degree_offset=0.1):
        """
        Generate a bounding box around a given latitude and longitude.
        """
        return {
            "lon_min": lon - degree_offset,
            "lat_min": lat - degree_offset,
            "lon_max": lon + degree_offset,
            "lat_max": lat + degree_offset,
        }

    def calculate_date_range(timestamp, days_offset=2):
        """
        Generate start and end dates around a given timestamp.
        """
        anomaly_date = datetime.strptime(timestamp, "%Y-%m-%d")
        start_date = (anomaly_date - timedelta(days=days_offset)).strftime("%Y-%m-%d")
        end_date = (anomaly_date + timedelta(days=days_offset)).strftime("%Y-%m-%d")
        return start_date, end_date

    def generate_grd_image(lon_min, lat_min, lon_max, lat_max, start_date, end_date, client_id, client_secret, data_folder):
        config = SHConfig()
        config.sh_client_id = client_id
        config.sh_client_secret = client_secret

        bbox = BBox(bbox=[lon_min, lat_min, lon_max, lat_max], crs=CRS.WGS84)

        request = SentinelHubRequest(
            evalscript="""
                function setup() {
                    return {
                        input: ["VV"],
                        output: { bands: 1, sampleType: "UINT8" }
                    };
                }
                function evaluatePixel(sample) {
                    return [sample.VV * 255];
                }
            """,
            input_data=[
                SentinelHubRequest.input_data(
                    data_collection=DataCollection.SENTINEL1_IW,
                    time_interval=(start_date, end_date),
                    mosaicking_order='mostRecent'
                )
            ],
            responses=[SentinelHubRequest.output_response('default', MimeType.PNG)],
            bbox=bbox,
            size=(1024, 1024),
            data_folder=data_folder,
            config=config
        )

        try:
            # Fetch and save the data
            request.get_data(save_data=True)

            # Recursively find the first PNG file in the folder
            for root, dirs, files in os.walk(data_folder):
                for file in files:
                    if file.endswith(".png"):
                        image_path = os.path.join(root, file)
                        print(f"GRD Image downloaded and saved at: {image_path}")

                        # Read the image
                        with rasterio.open(image_path) as dataset:
                            grd_image = dataset.read(1)

                        plt.figure(figsize=(10, 10))
                        plt.imshow(grd_image, cmap='gray', interpolation='nearest')
                        plt.colorbar(label="Intensity")
                        plt.title("Sentinel-1 GRD (VV Polarization)")
                        plt.savefig(os.path.join(output_folder, "grd_image.png"))
                        plt.close()

                        return grd_image

        # If no TIFF file is found
            raise FileNotFoundError(f"No TIFF files found in {data_folder} or its subdirectories.")

        except Exception as e:
            print("Error occurred while fetching the data:", str(e))
            return None
    
    def unet_predict(image, model_path, threshold=0.5):
        try:
            # Load the model
            model = tf.keras.models.load_model(model_path)

        # Ensure the image is 2D
            if len(image.shape) != 2:
                raise ValueError(f"Expected a 2D array for the image, but got shape {image.shape}")

        # Resize the image to the required input size
            target_size = (256, 256)
            image_resized = tf.image.resize(image[..., np.newaxis], target_size, method='bilinear').numpy()

        # Preprocess the image
            image_resized = np.expand_dims(image_resized, axis=0)  # Add batch dimension
            image_resized = image_resized / np.max(image_resized)  # Normalize the image

        # Predict
            prediction = model.predict(image_resized)

        # Post-process the prediction
            predicted_mask = np.squeeze(prediction)  # Remove batch and channel dimensions
        
        # Apply a threshold to the predicted mask
            oil_spill_detected = np.mean(predicted_mask) > threshold

        # Visualize the prediction
            plt.figure(figsize=(10, 10))
            plt.imshow(predicted_mask, cmap='jet', interpolation='nearest')
            plt.colorbar(label="Prediction Confidence")
            plt.title("UNet Model Prediction")
            plt.savefig(os.path.join(output_folder, "prediction.png"))
            plt.close()

        # Output decision
            if oil_spill_detected:
                print("Oil Spill Detected!")
            else:
                print("No Oil Spill Detected.")

            return predicted_mask, oil_spill_detected

        except Exception as e:
            print("Error during prediction:", str(e))
            return None, None




    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Generate bounding box and date range
    bbox = generate_bbox(lat, lon)
    start_date, end_date = calculate_date_range(timestamp)

    # Generate SAR image
    grd_image = generate_grd_image(bbox["lon_min"], bbox["lat_min"], bbox["lon_max"], bbox["lat_max"], start_date, end_date, client_id, client_secret, data_folder)

    # Perform prediction if image generation was successful
    if grd_image is not None:
        return unet_predict(grd_image, model_path)
    else:
        print("SAR image generation failed. No prediction performed.")
        return None

# # # Example usage
# latitude = 12
# longitude = 118
# timestamp = "2024-07-25"
# # client_id = 'db6f06cb-d661-439b-8942-71691e39c83b'
# # client_secret = 'KOZnydqNMtb6i1czu6mLiTXrV1D4u1xe'
# # model_path = 'unet_final_model.keras'  # Path to your pre-trained UNet model
# # data_folder = './SAR_Images'

# predicted_mask = process_sar_and_predict(latitude, longitude, timestamp)