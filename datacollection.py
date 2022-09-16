import pandas as pd
import os
df=pd.read_csv('https://data.gov.in/sites/default/files/all_india_PO_list_without_APS_offices_ver2_lat_long.csv')

save_path = 'files/datasets/'
df.to_csv(os.path.join(save_path,"all_india_city_pincodes.csv"))

print("success")