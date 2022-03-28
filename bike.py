# %matplotlib inline
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler
# import matplotlib.pyplot as plt
# import seaborn as sns
import xgboost as xgb
import pickle
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, explained_variance_score, mean_absolute_error
from sklearn.linear_model import LinearRegression


df = pd.read_csv('bikes.csv')
print(df.head())

## plot
# plt.figure(figsize=(11,11))
# sns.heatmap(df.corr().round(1), annot=True)

df['date'] = df['date'].apply(pd.to_datetime)

# we don't need it for the model but let's split it up for plotting
df['year'] = [i.year for i in df['date']]
df['month'] = [i.month_name()[0:3] for i in df['date']]
df['day'] = [i.day_name()[0:3] for i in df['date']]

# # plot
# figure, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(14,4), sharey=True)
# bp1 = sns.barplot(data=df, x='day', y='count', hue='year', ax=ax1)
# bp2 = sns.barplot(data=df, x='month', y='count', hue='year', ax=ax2)
# pp = sns.pairplot(data=df, y_vars=['count'], x_vars=['temperature', 'humidity', 'windspeed'], kind='reg',height=4)

#train the model

# X = df[['temperature', 'humidity', 'windspeed']]
# y = df['count']

# XGBoost
# X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.1, random_state=1)
# classifier = xgb.sklearn.XGBClassifier(nthread=-1, seed=1)
# classifier.fit(X_train, y_train)

# Linear Regression
# TODO: Apply a scaler function?
# perform a robust scaler transform of the dataset
scaler = StandardScaler()
# scaler = MinMaxScaler()
# data = scaler.fit_transform(X)
# convert the array back to a dataframe
# X = pd.DataFrame(df)
# Apply scaler() to all the columns except the 'yes-no' and 'dummy' variables
num_vars = ['temperature', 'humidity', 'windspeed']
df[num_vars] = scaler.fit_transform(df[num_vars])

X = df[num_vars]
print(X)
y = df['count']
print(y)

# for xgboost
# X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.1, random_state=1)
# for linear
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=42)
classifier = LinearRegression()
classifier.fit(X_train, y_train)
print(classifier.score(X_test, y_test))

# dump to pickle
with open('bike_model_linear.pkl', 'wb') as file:
    pickle.dump(classifier, file)
# with open('bike_model_xgboost.pkl', 'wb') as file:
#     pickle.dump(classifier, file)

predictions = classifier.predict(X_test)
print(f'R^2 score: {r2_score(y_true=y_test, y_pred=predictions):.2f}')
print(f'MAE score: {mean_absolute_error(y_true=y_test, y_pred=predictions):.2f}')
print(f'EVS score: {explained_variance_score(y_true=y_test, y_pred=predictions):.2f}')
# rp = sns.regplot(x=y_test, y=predictions)

# 