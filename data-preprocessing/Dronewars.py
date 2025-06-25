import pandas as pd


df = pd.read_excel(
    "C:/Users/Kushagra Sharma/Downloads/DroneWarsData.xlsx",
    sheet_name="All_WithoutUnknown",
    header=1,
)


df.columns = [
    "Strike_ID", "Country", "Date", "President", "Location", "Coordinates",
    "Latitude", "Longitude", "People_Killed", "People_Killed_Alt",
    "Min_Civilians", "Max_Civilians", "Min_Children", "Max_Children",
    "Suspected_Target", "Unnamed_15", "Unnamed_16", "Unnamed_17"
]


df['Date (MM-DD-YYYY)'] = pd.to_datetime(df['Date (MM-DD-YYYY)'], errors='coerce')


columns_to_keep = [
    "Date", "Location", "Suspected_Target", "People_Killed",
    "Min_Civilians", "Max_Civilians", "Latitude", "Longitude", "Country"
]

df = df[columns_to_keep]

df.to_csv("final_cleaned_drone_dataa.csv", index=False)
print("\nCleaned data saved  'tofinal_cleaned_drone_data.csv'")
