import pandas as pd
from datetime import timedelta
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.inspection import permutation_importance
import numpy as np
from matplotlib import pyplot as plt

# Define file paths for Federal Reserve and Zillow data
fed_files = ["MORTGAGE30US (2).csv", "RRVRUSQ156N (2).csv", "CPIAUCSL (2).csv"]
zillow_files = ["Metro_median_sale_price_uc_sfrcondo_week (1).csv", "Metro_zhvi_uc_sfrcondo_tier_0.33_0.67_month (1).csv"]

# Load and process Federal Reserve data
dfs = [pd.read_csv(f, parse_dates=True, index_col=0) for f in fed_files]
fed_data = pd.concat(dfs, axis=1)
fed_data = fed_data.ffill()  # Forward fill missing values

# Load and process Zillow data for Syracuse (RegionID 395143)
dfs = [pd.read_csv(f) for f in zillow_files]
data = dfs[0]
syr_data = data[data['RegionID'] == 395143]
syr_data = pd.DataFrame(syr_data.iloc[0,5:])

data = dfs[1]
syr_data2 = data[data['RegionID'] == 395143]
syr_data2 = pd.DataFrame(syr_data2.iloc[0,5:])

dfs = [syr_data, syr_data2]

# Convert index to datetime and add month column
for df in dfs:
    df.index = pd.to_datetime(df.index)
    df["month"] = df.index.to_period("M")

# Merge price data
price_data = dfs[0].merge(dfs[1], on="month")
price_data.index = dfs[0].index
del price_data["month"]

price_data.columns = ["Price", "Value"]

# Align Federal Reserve data with price data
fed_data = fed_data.dropna()
fed_data.index = fed_data.index + timedelta(days=2)

price_data = fed_data.merge(price_data, left_index=True, right_index=True)

price_data.columns = ["interest", "vacancy", "cpi", "price", "value"]

# Plot raw price data
price_data.plot.line(y="price", use_index=True)

# Calculate and plot adjusted price
price_data["adj_price"] = price_data["price"] / price_data["cpi"] * 100
price_data.plot.line(y="adj_price", use_index=True)

# Calculate adjusted value
price_data["adj_value"] = price_data["value"] / price_data["cpi"] * 100

# Calculate next quarter's price and price change
price_data["next_quarter"] = price_data["adj_price"].shift(-13)
price_data.dropna(inplace=True)
price_data["change"] = (price_data['next_quarter'] > price_data["adj_price"]).astype(int)

# Print value counts of price changes
price_data["change"].value_counts()

# Define predictors and target variable
predictors = ["interest", "vacancy", "adj_price", "adj_value"]
target = "change"

# Set parameters for backtesting
START = 260
STEP = 52

# Define prediction function
def predict(train, test, predictors, target):
    rf = RandomForestClassifier(min_samples_split=10, random_state=1)
    rf.fit(train[predictors], train[target])
    preds = rf.predict(test[predictors])
    return preds

# Define backtesting function
def backtest(data, predictors, target):
    all_preds = []
    for i in range(START, data.shape[0], STEP):
        train = price_data.iloc[:i]
        test = price_data.iloc[i:(i+STEP)]
        all_preds.append(predict(train,test,predictors,target))

    preds = np.concatenate(all_preds)
    return preds, accuracy_score(data.iloc[START:][target], preds)

# Calculate yearly averages and ratios
yearly = price_data.rolling(52, min_periods=1).mean()
yearly_ratios = [p + "_year" for p in predictors]
price_data[yearly_ratios] = price_data[predictors] / yearly[predictors]

# Perform backtesting
preds, accuracy = backtest(price_data, predictors + yearly_ratios, target)

# Prepare data for plotting predictions
pred_match = (preds == price_data[target].iloc[START:])
pred_match[pred_match == True] = 'green'
pred_match[pred_match == False] = 'red'

plot_data = price_data.iloc[START:].copy()
plot_data.reset_index().plot.scatter(x="index", y="adj_price", color=pred_match)
plt.show()

# Train final model and calculate feature importance
rf = RandomForestClassifier(min_samples_split=10, random_state=1)
rf.fit(price_data[predictors], price_data[target])
result = permutation_importance(rf, price_data[predictors], price_data[target], n_repeats=10, random_state=1)

# Print the final dataset
print(accuracy)