import pandas as pd
df=pd.read_csv('https://data.gov.in/sites/default/files/all_india_PO_list_without_APS_offices_ver2_lat_long.csv')
df.to_csv('all_india_pincodes.csv')
print("success")