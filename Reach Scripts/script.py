import pandas as pd
import numpy as np
import os
import csv
import datetime as dt



root_path = os.path.dirname(os.path.abspath(__file__))
fn = os.path.join(root_path, 'Raw.xlsx')

df = pd.read_excel(fn)


df = df.drop(['Status', 'Posting Title', 'Level', 'University Title'], axis=1)
df['Job Opening Status Dt'] = df['Job Opening Status Dt'].apply(lambda x: x.year)
df.rename(columns={'Job Opening Status Dt':'Year'})
df = df[~df['Demographic Selected'].isin(['Disability Unknown', 'Veteran Unknown', 'Race Unknown'])]

df_head = df.head(5)

new_df = pd.DataFrame()

for row in df.itertuples(index=False):
    for i in range(row._4):
        l = [row.Department, row._1, row._2, row._3, row._4, row._5, row._6, row._7, row._8]
        r = pd.Series(l)
        print(r)
        new_df = new_df.append(r, ignore_index=True)

new_df = new_df.rename(columns={0:'Department', 1:'Job Opening ID', 2:'Applicant Disposition', 3:'Demographic Selected', 4:'Cnt of Applicant IDs', 5:'Year', 6:'Job Family', 7:'Job Code', 8:'Level Description'})
new_df = new_df.drop(['Cnt of Applicant IDs'], axis=1)

#new_df[new_df['Demographic Selected'].str.contains("Unknown")]
#new_df.drop(list(df.filter(regex = 'Unknown')), axis = 1, inplace = True)

path = os.path.join(os.path.expanduser("~"), "Desktop", "Data NO UNKNOWNS.xls")
new_df.to_excel(path, index=False) 
print("done")