import pandas as pd
from sklearn.model_selection import train_test_split
import pickle

data = pd.read_csv('../data/data.csv')
data = data.drop(columns=['Unnamed: 0', 'state_fips_code', 'county_fips_code', 'case_month'])
data.pop_estimate_2019 = data.pop_estimate_2019.str.replace(',', '').astype(float)
data = data.astype(float)

Z = data.iloc[:, :].values.astype(float)
del data

X, y = Z[:, 2:], Z[:, :2]
del Z

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=20210927)

with open('../data/data_train.pt', 'wb') as file:
    pickle.dump((X_train, y_train), file)

with open('../data/data_test.pt', 'wb') as file:
    pickle.dump((X_test, y_test), file)