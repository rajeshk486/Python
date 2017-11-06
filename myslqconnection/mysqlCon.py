import pandas as pd
df= pd.read_csv('/home/hadoop/ML/temp.csv')
ss=df['temperature'].max
print('asdfghj')
print(df.temperature.max())
df= pd.read_csv('/home/hadoop/Downloads/Road_Weather_Information_Stations.csv')
print(df['AirTemperature'].max())