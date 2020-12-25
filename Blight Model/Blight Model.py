import os
import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import roc_auc_score

def import_dataset(csv):

    root_path = os.path.dirname(os.path.abspath(__file__))
    full_path = os.path.join(root_path, csv)
    
    return pd.read_csv(full_path, encoding = "ISO-8859-1", low_memory=False, warn_bad_lines=True)

def clean_data():
    
    train = import_dataset('train.csv')
    test = import_dataset('test.csv')
    add = import_dataset('addresses.csv')
    latlong = import_dataset('latlons.csv')
    
    #print(train.head(5))
    #print(test.head(5))
    #print(add.head(5))
    #print(latlong.head(5))

    #print(train.describe(include='all').T)
    #pint(train.info())

    # Make sure only dealing with instances in training data that have a target label
    #print(train.compliance.value_counts())

    # Make sure all changes to train set are also applied to test set

    # remove instances where compliance == NaN
    train.dropna(subset=['compliance'], inplace=True)

    # remove instances that are not within the USA
    train = train[train.country == 'USA']
    test = test[test.country == 'USA']

    # Merge latlongs with addresses in both train & test
    train = pd.merge(train, pd.merge(add, latlong, on='address'), on='ticket_id')
    test = pd.merge(test, pd.merge(add, latlong, on='address'), on='ticket_id')

    # Drop instances with obviously wrong zip codes
    bad_zip = train.zip_code[train.zip_code.apply(lambda x: len(str(x)) != 5)].index
    #print(bad_zipcode)
    train.drop(bad_zip, axis=0, inplace=True)
    #print(train.shape)
    train = train[~(train.hearing_date.isnull())]
    #print(train.shape)

    # drop unneeded columns
    cols = ['agency_name', 'inspector_name', 'violator_name', 'non_us_str_code', 'violation_description', 
                'grafitti_status', 'state_fee', 'admin_fee', 'ticket_issued_date', 'hearing_date', 
                'payment_amount', 'balance_due', 'payment_date', 'payment_status', 
                'collection_status', 'compliance_detail', 
                'violation_zip_code', 'country', 'address', 'violation_street_number',
                'violation_street_name', 'mailing_address_str_number', 'mailing_address_str_name', 
                'city', 'state', 'zip_code', 'address']
    train.drop(cols, axis=1, inplace=True)

    # Encode labels with strings into numbers so it is better understood and categorized by machine
    # disposition // violation code
    le = preprocessing.LabelEncoder()
    le.fit(train['disposition'].append(test['disposition'], ignore_index=True))
    train['disposition'] = le.transform(train['disposition'])
    test['disposition'] = le.transform(test['disposition'])

    le = preprocessing.LabelEncoder()
    le.fit(train['violation_code'].append(test['violation_code'], ignore_index=True))
    train['violation_code'] = le.transform(train['violation_code'])
    test['violation_code'] = le.transform(test['violation_code'])

    #print(train.head(3))
    #print(test.head(3))

    #print(train['lat'].isnull().sum())
    #print(train['lon'].isnull().sum())
    #print(test['lat'].isnull().sum())
    #print(test['lon'].isnull().sum())

    # for null values in train['lat'] and train['lon] its best to .mean() the lat and lon column and replace the 
    # null values with the mean
    #for x in [train['lat'], train['lon'], test['lat'], test['lon']]:
    #    x = x.fillna(x.mean())
    train['lat'].fillna(train['lat'].mean(), inplace=True)
    train['lon'].fillna(train['lon'].mean(), inplace=True)
    test['lat'].fillna(test['lat'].mean(), inplace=True)
    test['lon'].fillna(test['lon'].mean(), inplace=True)

    # removing true labels from test data

    cols = list(train.columns.values)
    cols.remove('compliance')
    test = test[cols]

    return train, test

def model(train, test):

    without_label = train.loc[:, train.columns != 'compliance']
    #print(without_label.head())
    X_train, X_test, y_train, y_test = train_test_split(without_label, train['compliance'])
    rfr = RandomForestRegressor()
    # Found best parameters to be n_estimators=200 and max_depth>=30 
    grid_values = {'n_estimators': [200], 'max_depth': [30]}
    grid = GridSearchCV(rfr, param_grid=grid_values, scoring='roc_auc')
    grid.fit(X_train, y_train)
    df = pd.DataFrame(grid.predict(test), index = test.ticket_id)

    return df

if __name__ == '__main__':

    train, test = clean_data()
    print(model(train, test))