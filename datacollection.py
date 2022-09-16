""""
import pandas as pd
import os
df=pd.read_csv('https://data.gov.in/sites/default/files/all_india_PO_list_without_APS_offices_ver2_lat_long.csv')

save_path = 'files/datasets/'
df.to_csv(os.path.join(save_path,"all_india_city_pincodes.csv"))

print("success") """
import os
import pgeocode
import pandas as pd

df=pd.read_csv('https://raw.githubusercontent.com/PULI-GOKULA-KISHORE-REDDY/IBM-HACK-CHALLENGE/main/files/datasets/all_india_city_pincodes.csv')
nomi = pgeocode.Nominatim('in')
li=df['pincode'].tolist()
li= map(str, li)
li=list(li)
print(len(li))
x=nomi.query_postal_code(li)

save_path = 'files/datasets/'
x.to_csv(os.path.join(save_path,"allindia_cities.csv"))
