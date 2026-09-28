from predict import process_sar_and_predict
def process_outliers(result):
    for outlier_type, outliers in result.items():
        for index, outlier in outliers.iterrows():
            lat = outlier['LAT']
            lon = outlier['LON']
            timestamp = outlier['BaseDateTime']
            print(f"Processing {outlier_type} outlier at ({lat}, {lon}) at {timestamp}")
            process_sar_and_predict(lat, lon, timestamp)
# # Sample data for SOG and COG outliers
# import pandas as pd

# # Example of SOG Outliers DataFrame
# sog_outliers = pd.DataFrame({
#     'LON': [-70.5, -70.6, -70.7],
#     'LAT': [42.5, 42.6, 42.7],
#     'BaseDateTime': ['2024-12-12 14:00:00', '2024-12-12 14:10:00', '2024-12-12 14:20:00'],
#     'SOG': [12.5, 14.0, 15.2]  # Sample speed over ground data
# })

# # Example of COG Outliers DataFrame
# cog_outliers = pd.DataFrame({
#     'LON': [-70.4, -70.5, -70.6],
#     'LAT': [42.4, 42.5, 42.6],
#     'BaseDateTime': ['2024-12-12 14:05:00', '2024-12-12 14:15:00', '2024-12-12 14:25:00'],
#     'COG': [180, 185, 190]  # Sample course over ground data
# })

# # Returning the result
# result = {
#     'SOG_Outliers': sog_outliers[['LON', 'LAT', 'BaseDateTime']].reset_index(drop=True),
#     'COG_Outliers': cog_outliers[['LON', 'LAT', 'BaseDateTime']].reset_index(drop=True)
# }

# process_outliers(result)
