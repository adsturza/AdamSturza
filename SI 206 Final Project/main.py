import requests
import os
import sqlite3
import numpy as np
from numpy.polynomial.polynomial import polyfit
import pandas as pd
import matplotlib.pyplot as plt

import LoadCounties
import LoadMaskUse
import LoadStates

"""
Creates an SQL database with the name ‘Covid’, which can be located at the path main.py is located at. Returns the cursor 
and connection to the database.
"""
def setUpDatabase():
    db_name = 'Covid'
    path = os.path.dirname(os.path.abspath(__file__))
    conn = sqlite3.connect(path+'/'+db_name)
    cur = conn.cursor()
    return cur, conn

"""
Performs an SQL join on the ‘Counties’ and ‘Mask_use’ tables and computes various averages as well as finds the counties with 
the greatest statistics (cases, deaths, etc.). Additionally, computes counts for each state for cases and deaths, and average 
proportions by state of responses to ALWAYS and NEVER wearing masks. Creates ‘data.txt’ output file and writes data to it. 
Returns list of county data fetched from the SQL join and a dictionary of data by state.
"""
def calculateWriteData(cur, conn):

    ###
    # Things to calculate:
    #   * County with highest proportion of population who NEVER wear masks and # of cases and deaths
    #   * County with highest proportion of population who ALWAYS wears masks and # of cases and deaths
    #   * County with most cases and its proportion of population who NEVER/AWLAYS wears a mask
    #   * County with most deaths and its proportion of population who NEVER/ALWAYS wears a mask
    #   * By state, number of cases and deaths
    #   * By state, average proportion of people who ALWAYS/NEVER wear masks (doesn't account for populations/weight in each county)
    #   
    #   * Use third table for hospitalizations by state
    ###

    cur.execute("SELECT Counties.county_id, Counties.county_name, Counties.county_state, Counties.cases, \
    Counties.deaths, Mask_use.response_never, Mask_use.response_always FROM Counties JOIN Mask_use ON    \
    Counties.county_id = Mask_use.county_id")
    data = cur.fetchall()
    
    highestPropNEVER = data[0]
    highestPropALWAYS = data[0]
    countyMostCases = data[0]
    countyMostDeaths = data[0]

    stateD = {} # {'state': [cases, deaths, # counties, SUM proportion NEVER wears masks, SUM proportion ALWAYS wears masks]}
                # Going to need to find average proportions by doing (SUM of total proportions) / (# of counties)

    for item in data:
        ## County with highest proportion NEVER wears masks
        if item[5] > highestPropNEVER[5]:
            highestPropNEVER = item
        
        ## County with highest proportion ALWAYS wears masks
        if item[6] > highestPropALWAYS[6]:
            highestPropALWAYS = item

        ## County with most cases
        if item[3] > countyMostCases[3]:
            countyMostCases = item

        ## County with most deaths
        if item[4] > countyMostDeaths[4]:
            countyMostDeaths = item

        ## StateD
        if item[2] not in stateD:
            value_list = [item[3], item[4], 1, item[5], item[6]]
            stateD[item[2]] = value_list
        else:
            value_list = stateD[item[2]]

            value_list[0] += item[3]
            value_list[1] += item[4]
            value_list[2] += 1
            value_list[3] += item[5]
            value_list[4] += item[6]

            value_list[3] = round(value_list[3], 3)
            value_list[4] = round(value_list[4], 3)

            stateD[item[2]] = value_list

    #print(highestPropNEVER)
    #print(highestPropALWAYS)
    #print(countyMostCases)
    #print(countyMostDeaths)

    ## stateD: {'state': [cases, deaths, AVG proportion NEVER wears masks, AVG proportion ALWAYS wears masks]}
    for state in stateD.keys():
        old_list = stateD[state]

        avg_prop_NEVER = round(float(old_list[3]/old_list[2]), 3)
        avg_prop_ALWAYS = round(float(old_list[4]/old_list[2]), 3)

        new_list = [old_list[0], old_list[1], avg_prop_NEVER, avg_prop_ALWAYS]
        stateD[state] = new_list

    ## stateD: {'state': [cases, deaths, AVG proportion NEVER wears masks, AVG proportion ALWAYS wears masks, hospitalizations]}
    cur.execute("SELECT state, hospitalized FROM States")
    state_hospitalizations = cur.fetchall()
    for state in state_hospitalizations:
        if state[0] in stateD and state[1] != None:
            stateD[state[0]].append(state[1])
    
    #print(stateD)

    root_path = os.path.dirname(os.path.abspath(__file__))
    fn = os.path.join(root_path, 'data.txt')
    with open(fn, 'w') as f:
        f.write("---Calculated COVID Data Output File :: Here are some of the calculations we did---")
        f.write("\n\n\n")
        f.write(f"{highestPropNEVER[1]}, {highestPropNEVER[2]} was the county with the highest proportion of its population reporting it never wore masks at {highestPropNEVER[5]*100}%. The county has reported {highestPropNEVER[3]} cases and {highestPropNEVER[4]} deaths.")
        f.write("\n\n")
        f.write(f"{highestPropALWAYS[1]}, {highestPropALWAYS[2]} was the county with the highest proportion of its population reporting it always wore masks at {highestPropALWAYS[6]*100}%. The county has reported {highestPropALWAYS[3]} cases and {highestPropALWAYS[4]} deaths.")
        f.write("\n\n")
        f.write(f"{countyMostCases[1]}, {countyMostCases[2]} was the county with the most reported cases at {countyMostCases[3]} cases. {round(countyMostCases[6]*100, 1)}% of the population reported always wearing masks, while {countyMostCases[5]*100}% reported never wearing masks.")
        f.write("\n\n")
        f.write(f"{countyMostDeaths[1]}, {countyMostDeaths[2]} was the county with the most reported deaths at {countyMostDeaths[4]} deaths. {round(countyMostDeaths[6]*100, 1)}% of the population reported always wearing masks, while {countyMostDeaths[5]*100}% reported never wearing masks.")
        f.write("\n\n\n")
        
        f.write("---County Statistics Converted to State Data---")
        f.write("\n\n\n")
        for state in stateD.keys():
            d_values = stateD[state] ## [cases, deaths, AVG proportion NEVER wears masks, AVG proportion ALWAYS wears masks, hospitalizations]
            cases, deaths, avg_never, avg_always = d_values[0], d_values[1], round(d_values[2]*100, 1), round(d_values[3]*100, 1)
            if len(d_values) == 5:
                hospitalizations = d_values[4]
                f.write(f"{state}\'s population, on average, reports never wearing masks {avg_never}% of the time and always wearing masks {avg_always}% of the time. {state} has reported an accumulated {cases} cases, {hospitalizations} hospitalizations, and {deaths} deaths.")
            else:
                f.write(f"{state}\'s population, on average, reports never wearing masks {avg_never}% of the time and always wearing masks {avg_always}% of the time. {state} has reported an accumulated {cases} cases and {deaths} deaths (hospitalizations not reported).")
            f.write("\n\n")
        
        f.close()
        
    
    conn.commit()
    return data, stateD

"""
Takes the list of county data and dictionary of state data returned by calculateWriteData() and creates four visualizations, 
each containing a few subplots of matplotlib graphs, which can be clicked through. Does not return anything.
"""
def visualizations(county_data, stateD):

    ##
    # Michigan Data for first two visualizations
    ##
    
    MichiganData = [county for county in county_data if county[2] == 'Michigan']
    #print(MichiganData)
    countyState_MichiganData = [str(x[1] + ", " + x[2]) for x in MichiganData]
    cases_MichiganData = [x[3] for x in MichiganData]
    deaths_MichiganData = [x[4] for x in MichiganData]
    always_mask_MichiganData = [x[6]*100 for x in MichiganData]
    never_mask_MichiganData = [x[5]*100 for x in MichiganData]
    #print(len(always_mask_MichiganData))
    top10MichCountiesByCases = sorted(MichiganData, key = lambda x : x[3])[-10:]
    #print(top10MichCountiesByCases)
    top10MichCountiesNames = [str(x[1] + ", " + x[2]) for x in top10MichCountiesByCases]
    top10MichCountiesCases = [x[3] for x in top10MichCountiesByCases]
    top10MichCountiesDeaths = [x[4] for x in top10MichCountiesByCases]
    top10MichCountiesALWAYSmasks = [x[6]*100 for x in top10MichCountiesByCases]
    top10MichCountiesNEVERmasks = [x[5]*100 for x in top10MichCountiesByCases]

    ##
    # 1st Michigan visualization window, 3 graphs
    ##

    N = len(top10MichCountiesByCases)
    width = .35
    ind = np.arange(N)

    fig = plt.figure(figsize=(14,7))

    ax = fig.add_subplot(131)

    bar1 = ax.bar(ind, top10MichCountiesCases, width, color='blue')
    ax.set_xticks(ind + width / 2 - .17)
    ax.set_xticklabels(top10MichCountiesNames)
    ax.set(xlabel='County', ylabel='Cases', title='Top 10 Michigan Counties (cases) vs Cases')
    ax.grid()

    ax2 = fig.add_subplot(132)

    bar2 = ax2.bar(ind, top10MichCountiesDeaths, width, color='orange')
    ax2.set_xticks(ind + width / 2 - .17)
    ax2.set_xticklabels(top10MichCountiesNames)
    ax2.set(xlabel='County', ylabel='Deaths', title='Top 10 Michigan Counties (cases) vs Deaths')
    ax2.grid()

    ax3 = fig.add_subplot(133)

    bar3 = ax3.bar(ind, top10MichCountiesNEVERmasks, width, color='red')
    bar4 = ax3.bar(ind + width, top10MichCountiesALWAYSmasks, width, color='green')
    ax3.set_xticks(ind + width / 2)
    ax3.set_xticklabels(top10MichCountiesNames)
    ax3.set(xlabel='County', ylabel='% Reporting They Wear Masks', title='% Reporting Wearing Masks')
    ax3.legend((bar3[0], bar4[0]), ('Never', 'Always'))
    ax3.grid()

    plt.setp(ax.get_xticklabels(), rotation=30, horizontalalignment='right')
    plt.setp(ax2.get_xticklabels(), rotation=30, horizontalalignment='right')
    plt.setp(ax3.get_xticklabels(), rotation=30, horizontalalignment='right')
    
    #plt.suptitle("Top 10 Michigan Counties By Cases Various Data")
    plt.tight_layout()
    plt.show()

    ##
    # 2nd Michigan visualization window, 2 graphs
    ##

    N = len(MichiganData)
    width = .35
    ind = np.arange(N)

    fig = plt.figure(figsize=(14,7))

    ax = fig.add_subplot(121)
    #bar = ax.bar(ind, cases_MichiganData, width, color='red')
    ax.scatter(always_mask_MichiganData, cases_MichiganData, color=(0,0,0), marker='*')
    ax.set_xlabel("% ALWAYS wearing masks")
    ax.set_ylabel("Cases")
    ax.set_title("By Michigan County, % Always Wearing Masks vs # of Cases")
    ax.set_xticks(np.arange(30, 90, 10))
    #ax.set_yticks(np.arange(0, 28000, 2800))
    ax2 = fig.add_subplot(122)
    ax2.scatter(never_mask_MichiganData, cases_MichiganData, color=(0,0,0), marker='+')
    ax2.set_xlabel("% NEVER wearing masks")
    ax2.set_ylabel("Cases")
    ax2.set_title("By Michigan County, % NEVER Wearing Masks vs # of Cases")
    ax2.set_xticks(np.arange(0, 24, 4))
    #ax.set_yticks(np.arange(0, 28000, 2800))
    plt.tight_layout()
    plt.show()

    ##
    # Getting data for top 10 counties by cases and top 10 counties by deaths
    ##

    top10Deaths = sorted(county_data, key = lambda x : x[4])[-10:]
    #print(top10Deaths)
    top10Cases = sorted(county_data, key = lambda x : x[3])[-10:]
    #bottom10Cases = sorted(county_data, key = lambda x : x[3])[:10]
    #print(top10Cases)
    #print(bottom10Cases)

    countyState_top10Cases = [str(x[1] + ", " + x[2]) for x in top10Cases]
    cases_top10Cases = [x[3] for x in top10Cases]
    deaths_top10Cases = [x[4] for x in top10Cases]
    always_mask_top10Cases = [x[6]*100 for x in top10Cases]
    never_mask_top10Cases = [x[5]*100 for x in top10Cases]

    countyState_top10Deaths = []
    for county in top10Deaths:
        countyState_top10Deaths.append(county[1] + ", " + county[2])
    #cases_top10Deaths = [x[3] for x in top10Deaths]
    deaths_top10Deaths = [x[4] for x in top10Deaths]
    always_mask_top10Deaths = [x[6]*100 for x in top10Deaths]
    #never_mask_top10Deaths = [x[5]*100 for x in top10Deaths]

    ##
    # 3rd Visualization window: 4 graphs about top 10 counties with most cases/deaths/always wearing masks in these counties
    ##
    
    N = 10
    width = .35
    ind = np.arange(N)

    fig = plt.figure(figsize=(14,7))
    
    ax2 = fig.add_subplot(221)
    p2 = ax2.bar(ind, cases_top10Cases, width, color='red')
    ax2.set_xticks(ind + width / 2 - .17)
    ax2.set_xticklabels(countyState_top10Cases)
    ax2.set(xlabel='County', ylabel='Cases', title='10 Counties With The Most Cases From COVID-19')
    ax2.grid()
    


    ax = fig.add_subplot(222)
    p1 = ax.bar(ind, deaths_top10Deaths, width, color='black')
    ax.set_xticks(ind + width / 2 - .17)
    ax.set_xticklabels(countyState_top10Deaths)
    #ax.legend((p1[0]), ('Deaths'))
    #ax.autoscale_view()
    ax.set(xlabel='County', ylabel='Deaths', title='10 Counties With The Most Deaths From COVID-19')
    ax.grid()

    ax3 = fig.add_subplot(223)
    p3 = ax3.bar(ind, always_mask_top10Cases, width, color='green')
    ax3.set_xticks(ind + width / 2 - .17)
    ax3.set_yticks([10,20,30,40,50,60,70,80,90,100])
    ax3.set_xticklabels(countyState_top10Cases)
    #ax.legend((p1[0]), ('Deaths'))
    #ax.autoscale_view()
    ax3.set(xlabel='County', ylabel='% ALWAYS wears masks', title='% Always Wearing A Mask In The 10 Counties With The Most Cases')
    ax3.grid()

    ax4 = fig.add_subplot(224)
    p4 = ax4.bar(ind, always_mask_top10Deaths, width, color='green')
    ax4.set_xticks(ind + width / 2 - .17)
    ax4.set_yticks([10,20,30,40,50,60,70,80,90,100])
    ax4.set_xticklabels(countyState_top10Deaths)
    #ax.legend((p1[0]), ('Deaths'))
    #ax.autoscale_view()
    ax4.set(xlabel='County', ylabel='% ALWAYS wears masks', title='% Always Wearing A Mask In The 10 Counties With The Most Deaths')
    ax4.grid()
    
    plt.setp(ax.get_xticklabels(), rotation=30, horizontalalignment='right')
    plt.setp(ax2.get_xticklabels(), rotation=30, horizontalalignment='right')
    plt.setp(ax3.get_xticklabels(), rotation=30, horizontalalignment='right')
    plt.setp(ax4.get_xticklabels(), rotation=30, horizontalalignment='right')

    plt.tight_layout()
    plt.show()

    ##
    # Getting data about each state and inputting into numpy arrays in order to fit regression lines
    ##
   
    #print(stateD)
    names = list(stateD.keys())
    always_x = np.array([], float)
    never_x = np.array([], float)
    deaths_y = np.array([], int)
    cases_y = np.array([], int)
    hospitalizations_x = np.array([], float)
    hospitalizations = np.array([], int)
    for state in stateD:
        values = stateD[state]
        always_x = np.append(always_x, round(values[3]*100, 1)) #float
        never_x = np.append(never_x, round(values[2]*100, 1))
        deaths_y = np.append(deaths_y, values[1])
        cases_y = np.append(cases_y, values[0]) #int
        if len(values) == 5:
            hospitalizations_x = np.append(hospitalizations_x, values[0])
            hospitalizations = np.append(hospitalizations, values[4])

    ##
    # 4th visualization window -- by state, cases vs deaths, cases vs hospitalizations
    # Could definitely make more graphs using this data using state averages for mask wearing, etc
    # Would've been better with time series data
    ##
    
    fig = plt.figure(figsize=(14,7))
    
    x = cases_y
    y = deaths_y

    b, m = polyfit(x, y, 1)
    ax1 = fig.add_subplot(1,2,1)
    ax1.scatter(x, y, color=(0,0,0), marker='*')
    ax1.plot(x, b + m * x, '-', color="red")
    ax1.set_xlabel("Cases")
    ax1.set_ylabel("Deaths")
    ax1.set_title("By state, # of cases vs # of deaths")
    ax1.set_xticks(np.arange(0, 2000000, 200000))
    ax1.set_yticks(np.arange(0, 28000, 2800))
    
    x = hospitalizations_x
    y = hospitalizations

    b, m = polyfit(x, y, 1)
    ax2 = fig.add_subplot(1,2,2)
    ax2.scatter(x, y, color=(0,0,0), marker='*')
    ax2.plot(x, b + m * x, '-', color="blue")
    ax2.set_xlabel("Cases")
    ax2.set_ylabel("Hospitalizations")
    ax2.set_title("By state, # of cases vs # of hospitalizations")
    ax2.set_xticks(np.arange(0, 2000000, 200000))
    ax2.set_yticks(np.arange(0, 100000, 10000))
    """
    x = hospitalizations_x
    y = hospitalizations

    b, m = polyfit(x, y, 1)
    ax3 = fig.add_subplot(2,3,3)
    ax3.scatter(x, y, color=(0,0,0))
    ax3.plot(x, b + m * x, '-')
    ax3.set_xlabel("Average % of state population who report NEVER wearing a mask")
    ax3.set_ylabel("Hospitalizations")
    ax3.set_title("Never wearing a mask vs hospitalizations")
    """
    
    plt.tight_layout()
    plt.show()

"""
Runs setUpDatabase(), saves the variables returned from running calculateWriteData(), and runs visualizations() with the same 
variables as inputs. Finally, closes the database connection.
"""
if __name__ == '__main__':
    cur, conn = setUpDatabase()
    data, stateD = calculateWriteData(cur, conn)
    visualizations(data, stateD)
    conn.close()
    