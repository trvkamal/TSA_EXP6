# Ex.No: 6               HOLT WINTERS METHOD

### Name: KAMALESH V

### AIM:

### ALGORITHM:
1. You import the necessary libraries
2. You load a CSV file containing daily sales data into a DataFrame, parse the 'date' column as
datetime, and perform some initial data exploration
3. You group the data by date and resample it to a monthly frequency (beginning of the month
4. You plot the time series data
5. You import the necessary 'statsmodels' libraries for time series analysis
6. You decompose the time series data into its additive components and plot them:
7. You calculate the root mean squared error (RMSE) to evaluate the model's performance
8. You calculate the mean and standard deviation of the entire sales dataset, then fit a Holt-
Winters model to the entire dataset and make future predictions
9. You plot the original sales data and the predictions
### PROGRAM:
```
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.seasonal import seasonal_decompose


data = pd.read_csv("silver.csv", parse_dates=['Date'], index_col='Date')
print(data.head())


data_monthly = data['USD'].resample('MS').mean()   # Monthly average prices
print(data_monthly.head())

data_monthly.plot(title="Monthly Silver Price")
plt.show()

scaler = MinMaxScaler()
scaled_data = pd.Series(
    scaler.fit_transform(data_monthly.values.reshape(-1, 1)).flatten(),
    index=data_monthly.index
)

scaled_data.plot(title="Scaled Silver Data")
plt.show()


decomposition = seasonal_decompose(data_monthly, model="additive")
decomposition.plot()
plt.show()


scaled_data = scaled_data + 1   # multiplicative seasonality needs >0
train_data = scaled_data[:int(len(scaled_data) * 0.8)]
test_data = scaled_data[int(len(scaled_data) * 0.8):]

model = ExponentialSmoothing(train_data, trend='add', seasonal='mul', seasonal_periods=12).fit()
test_predictions = model.forecast(steps=len(test_data))

# Plot predictions vs test
ax = train_data.plot(label="Train Data")
test_predictions.plot(ax=ax, label="Predictions")
test_data.plot(ax=ax, label="Test Data")
ax.legend()
ax.set_title("Visual Evaluation - Silver Price")
plt.show()

rmse = np.sqrt(mean_squared_error(test_data, test_predictions))
print("Test RMSE:", rmse)
print("Variance:", np.sqrt(scaled_data.var()), "Mean:", scaled_data.mean())

final_model = ExponentialSmoothing(data_monthly, trend='add', seasonal='mul', seasonal_periods=12).fit()
future_steps = int(len(data_monthly) / 4)  # predict ~ next 1/4th of series length
final_predictions = final_model.forecast(steps=future_steps)

# Plot final predictions
ax = data_monthly.plot(label="Silver Prices")
final_predictions.plot(ax=ax, label="Forecast")
ax.legend()
ax.set_xlabel("Months")
ax.set_ylabel("Silver Price (USD)")
ax.set_title("Silver Price Forecasting (Holt-Winters)")
plt.show()
```
### OUTPUT:

<img width="664" height="546" alt="image" src="https://github.com/user-attachments/assets/08d03783-ea2a-418a-946c-68d56657713d" />
<img width="743" height="570" alt="image" src="https://github.com/user-attachments/assets/78716dcd-a018-4937-beef-f57d47a02451" />
<img width="694" height="594" alt="image" src="https://github.com/user-attachments/assets/e8727e07-c450-4d2b-a592-26eb4f6461c3" />


### RESULT:
Thus the program run successfully based on the Holt Winters Method model.
