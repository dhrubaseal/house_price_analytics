# Find publicly available data for key factors that influence US home prices nationally. Then, build a data science model that explains how these factors impacted home prices over the last 20 years.
# Use the S&P Case-Schiller Home Price Index as a proxy for home prices:(fred.stlouisfed.org/series/CSUSHPISA).

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Load datasets
home_prices = pd.read_csv('https://fred.stlouisfed..vorg/data/CSUSHPISA.csv', parse_dates=['DATE'], index_col='DATE')
mortgage_rates = pd.read_csv('https://fred.stlouisfed.org/data/MORTGAGE30US.csv', parse_dates=['DATE'], index_col='DATE')
unemployment = pd.read_csv('https://fred.stlouisfed.org/data/UNRATE.csv', parse_dates=['DATE'], index_col='DATE')
income = pd.read_csv('https://fred.stlouisfed.org/data/MEHOINUSA672N.csv', parse_dates=['DATE'], index_col='DATE')
housing_starts = pd.read_csv('https://fred.stlouisfed.org/data/HOUST.csv', parse_dates=['DATE'], index_col='DATE')
cpi = pd.read_csv('https://fred.stlouisfed.org/data/CPIAUCSL.csv', parse_dates=['DATE'], index_col='DATE')

# Merge datasets on DATE
df = home_prices.join([mortgage_rates, unemployment, income, housing_starts, cpi], how='inner')
df.columns = ['Home_Price_Index', 'Mortgage_Rate', 'Unemployment_Rate', 'Median_Income', 'Housing_Starts', 'CPI']

# Fill missing values if any
df.fillna(method='ffill', inplace=True)

# Exploratory Data Analysis
plt.figure(figsize=(14, 7))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
plt.title('Correlation Matrix')
plt.show()

# Model Building
X = df[['Mortgage_Rate', 'Unemployment_Rate', 'Median_Income', 'Housing_Starts', 'CPI']]
y = df['Home_Price_Index']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Model Evaluation
mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error: {mse}')

# Coefficients
coef_df = pd.DataFrame(model.coef_, X.columns, columns=['Coefficient'])
print(coef_df)

# Plot actual vs predicted
plt.figure(figsize=(10, 5))
plt.plot(y_test.index, y_test, label='Actual')
plt.plot(y_test.index, y_pred, label='Predicted')
plt.legend()
plt.show()