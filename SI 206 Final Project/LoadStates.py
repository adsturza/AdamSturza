import requests
import os
import sqlite3
import numpy as np
import pandas as pd

import main

"""
Takes in the SQLite database cursor and connector as inputs. Establishes the Pandas connection with the website where the 
raw data is stored and creates the table ‘States’ if it does not already exist. A dictionary maps United States state abbreviations 
to full names and loads all the data into the ‘States’ SQL table. Does not return anything.
"""
def load25(cur, conn):

    url = 'https://api.covidtracking.com/v1/states/current.csv'
    states_df = pd.read_csv(url, error_bad_lines=False)

    cur.execute("CREATE TABLE IF NOT EXISTS States (state TEXT PRIMARY KEY, positive INTEGER, deaths INTEGER, hospitalized INTEGER)")

    cols_to_keep = ['state', 'positive', 'death', 'hospitalizedCumulative']
    states_df = states_df[cols_to_keep]

    us_state_abbrev = {
    'AL': 'Alabama',
    'AK': 'Alaska',
    'AS': 'American Samoa',
    'AZ': 'Arizona',
    'AR': 'Arkansas',
    'CA': 'California',
    'CO': 'Colorado',
    'CT': 'Connecticut',
    'DE': 'Delaware',
    'DC': 'District of Columbia',
    'FL': 'Florida',
    'GA': 'Georgia',
    'GU': 'Guam',
    'HI': 'Hawaii',
    'ID': 'Idaho',
    'IL': 'Illinois',
    'IN': 'Indiana',
    'IA': 'Iowa',
    'KS': 'Kansas',
    'KY': 'Kentucky',
    'LA': 'Louisiana',
    'ME': 'Maine',
    'MD': 'Maryland',
    'MA': 'Massachusetts',
    'MI': 'Michigan',
    'MN': 'Minnesota',
    'MS': 'Mississippi',
    'MO': 'Missouri',
    'MT': 'Montana',
    'NE': 'Nebraska',
    'NV': 'Nevada',
    'NH': 'New Hampshire',
    'NJ': 'New Jersey',
    'NM': 'New Mexico',
    'NY': 'New York',
    'NC': 'North Carolina',
    'ND': 'North Dakota',
    'MP':'Northern Mariana Islands',
    'OH': 'Ohio',
    'OK': 'Oklahoma',
    'OR': 'Oregon',
    'PA': 'Pennsylvania',
    'PR': 'Puerto Rico',
    'RI': 'Rhode Island',
    'SC': 'South Carolina',
    'SD': 'South Dakota',
    'TN': 'Tennessee',
    'TX': 'Texas',
    'UT': 'Utah',
    'VT': 'Vermont',
    'VI': 'Virgin Islands',
    'VA': 'Virginia',
    'WA': 'Washington',
    'WV': 'West Virginia',
    'WI': 'Wisconsin',
    'WY': 'Wyoming'
    }

    states_df['state'] = states_df['state'].map(us_state_abbrev).fillna(states_df['state'])

    print("--- Loading state data into SQL database ---")
    num_rows = len(states_df.index)
    for i in range(num_rows):
        #print(states_df.loc[i, 'state'], states_df.loc[i, 'positive'], states_df.loc[i, 'death'], states_df.loc[i, 'hospitalizedCumulative'])
        cur.execute("INSERT OR IGNORE INTO States (state, positive, deaths, hospitalized) VALUES (?,?,?,?)", \
        (states_df.loc[i, 'state'], int(states_df.loc[i, 'positive']), int(states_df.loc[i, 'death']),        \
        states_df.loc[i, 'hospitalizedCumulative']))
    print("--- Loaded ---")
    
    cur.execute("SELECT * FROM States")
    data = cur.fetchall()
    print("Total number of items in table: " + str(len(data)))
    conn.commit()
    return


if __name__ == '__main__':
    cur, conn = main.setUpDatabase()
    load25(cur, conn)