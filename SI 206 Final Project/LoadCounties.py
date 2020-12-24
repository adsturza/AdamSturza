import requests
import os
import sqlite3
import numpy as np
import pandas as pd

import main

"""
Takes in the name of the Pandas dataframe, the index of the row of data to insert, and the SQLite database cursor and connector 
as inputs. Loads that row of data from the Pandas dataframe into the ‘Counties’ SQL table. Does not return anything.
"""
def insertIntoDatabase(df, row, cur, conn):
    cur.execute("INSERT OR IGNORE INTO Counties (county_id, county_name, county_state, cases, deaths) VALUES (?,?,?,?,?)", \
    (df.loc[row, 'fips'], df.loc[row, 'county'], df.loc[row, 'state'], int(df.loc[row, 'cases']), df.loc[row, 'deaths']))
    return

"""
Takes in the SQLite database cursor and connector as inputs. Establishes the Pandas connection with the website where the raw 
data is stored, creates the table ‘Counties’ if it does not already exist, and if there are not already 100 entries in the SQL 
table, loads 25 entries. Otherwise loads the rest of the entries. Does not return anything.
"""
def load25(cur, conn):
    
    url = 'https://raw.githubusercontent.com/nytimes/covid-19-data/master/live/us-counties.csv'
    counties_df = pd.read_csv(url, error_bad_lines=False)

    cur.execute("CREATE TABLE IF NOT EXISTS Counties (county_id INTEGER PRIMARY KEY, county_name TEXT, county_state TEXT, \
    cases INTEGER, deaths INTEGER)")
    
    cols_to_keep = ['fips', 'county', 'state', 'cases', 'deaths']
    counties_df = counties_df[cols_to_keep]
    
    lower = 0
    upper = 25
    fips = 1001

    under100 = True
    cur.execute("SELECT * FROM Counties")
    if len(cur.fetchall()) >= 100:
        under100 = False

    while True:
        if under100:
            cur.execute(f"SELECT EXISTS(SELECT * FROM Counties WHERE county_id={fips})")
            tF = cur.fetchone()[0]
            if tF == 0:
                print("--- Loading 25 items ---")
                for i in range(lower, upper):
                    insertIntoDatabase(counties_df, i, cur, conn)
                break
            else:
                lower += 25
                upper += 25
                fips += 50
        else:
            print("--- At least 100 items loaded, loading the rest ---")
            lower = 100
            upper = len(counties_df.index)
            for i in range(lower, upper):
                insertIntoDatabase(counties_df, i, cur, conn)
            print("--- Loaded ---")
            break
        
    cur.execute("SELECT * FROM Counties")
    data = cur.fetchall()
    print("Total number of items in table: " + str(len(data)))

    conn.commit()
    return


if __name__ == '__main__':
    cur, conn = main.setUpDatabase()
    load25(cur, conn)