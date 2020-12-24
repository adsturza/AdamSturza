import requests
import os
import sqlite3
import numpy as np
import pandas as pd

import main

"""
Takes in the name of the Pandas dataframe, the index of the row of data to insert, and the SQLite database cursor and 
connector as inputs. Loads that row of data from the Pandas dataframe into the ‘Mask_use’ SQL table. Does not return anything.
"""
def insertIntoDatabase(df, row, cur, conn):
    cur.execute("INSERT OR IGNORE INTO Mask_use (county_id, response_never, response_always) VALUES (?,?,?)", \
    (int(df.loc[row, 'COUNTYFP']), float(round(df.loc[row, 'NEVER'], 3)), float(round(df.loc[row, 'ALWAYS'], 3))))
    return

"""
Takes in the SQLite database cursor and connector. Establishes the Pandas connection with the website where the raw 
data is stored, creates the table ‘Mask_use’ if it does not already exist, and if there are not already 100 entries in the 
SQL table, loads 25 entries. Otherwise loads the rest of the entries. Does not return anything.
"""
def load25(cur, conn):
    
    url = 'https://raw.githubusercontent.com/nytimes/covid-19-data/master/mask-use/mask-use-by-county.csv'
    mask_use_df = pd.read_csv(url, error_bad_lines=False)

    cur.execute("CREATE TABLE IF NOT EXISTS Mask_use (county_id INTEGER PRIMARY KEY, response_never FLOAT, response_always FLOAT)")

    cols_to_keep = ['COUNTYFP', 'NEVER', 'ALWAYS']
    mask_use_df = mask_use_df[cols_to_keep]

    lower = 0
    upper = 25
    id = 1001
    
    under100 = True
    cur.execute("SELECT * FROM Mask_use")
    if len(cur.fetchall()) >= 100:
        under100 = False

    while True:
        if under100:
            cur.execute(f"SELECT EXISTS(SELECT * FROM Mask_use WHERE county_id={id})")
            tF = cur.fetchone()[0]
            if tF == 0:
                print("--- Loading 25 items ---")
                for i in range(lower, upper):
                    insertIntoDatabase(mask_use_df, i, cur, conn)
                break
            else:
                lower += 25
                upper += 25
                id += 50
        else:
            print("--- At least 100 items loaded, loading the rest ---")
            lower = 100
            upper = len(mask_use_df.index)
            for i in range(lower, upper):
                insertIntoDatabase(mask_use_df, i, cur, conn)
            print("--- Loaded ---")
            break
                
    cur.execute("SELECT * FROM Mask_use")
    data = cur.fetchall()
    print("Total number of items in table: " + str(len(data)))

    conn.commit()
    return


if __name__ == '__main__':
    cur, conn = main.setUpDatabase()
    load25(cur, conn)