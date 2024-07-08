import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_log_error, mean_absolute_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV
import pickle

# # Exploring the data--------------------------------------------------------------------------------
# df = pd.read_csv("./data/TrainAndValid.csv",
#                  low_memory=False,  # low_memory = false, means do not worry about ram, we have enough
#                  parse_dates=["saledate"])  # parse_dates tell pandas whether some column is of datatype or not
# # print(df.info())
# # print(df.isna().sum())
#
# # Visualizing the data------------------------------------------------------------------------------
# fig, ax = plt.subplots()
# ax.scatter(df["saledate"][:1000], df["SalePrice"][:1000])
# # plt.show()
# # the problem with this plot is that the x-label is overcrowded due to the issue that matplotlib does not
# # recognize it as date but rather as some random integer values to fix this we can use 'timeparse'
#
# # Sort data frame by sale date----------------------------------------------------------------------
# df.sort_values(by=["saledate"], inplace=True, ascending=True)
# # print(df["saledate"].head())
#
# # Making the copy of the original dataframe---------------------------------------------------------
# df_temp = df.copy()
# # print(df_temp["saledate"].head(20))
#
# # Adding datetime parameters for saledate column----------------------------------------------------
# df_temp["saleYear"] = df_temp["saledate"].dt.year
# df_temp["saleMonth"] = df_temp["saledate"].dt.month
# df_temp["saleDay"] = df_temp["saledate"].dt.day
# df_temp["saleDayofWeek"] = df_temp["saledate"].dt.dayofweek
# df_temp["saleDayofYear"] = df_temp["saledate"].dt.dayofyear
# df_temp.drop("saledate", axis=1, inplace=True)
#
# # print(df_temp.head().T)
#
# # Converting string into pandas categorical datatype------------------------------------------------
# for label, content in df_temp.items():
#     if pd.api.types.is_float_dtype(content):
#         pass
#     else:
#         if pd.api.types.is_string_dtype(content) or content.apply(pd.isnull).any():
#             df_temp[label] = content.astype("category").cat.as_ordered()
#
# # Saving the processed data-------------------------------------------------------------------------
# df_temp.info()
# df_temp.to_csv("./data/train_temp.csv")
# df_temp.to_pickle("./data/train_temp.pkl")
# # There is a problem in pandas.to_csv() it does not store the changed datatypes so to resolve this issue
# # we have used pickle

# Retrieving data from the saved file---------------------------------------------------------------
df_temp = pd.read_pickle("./data/train_temp.pkl")
# df_temp.info()

# Filling numerical missing values------------------------------------------------------------------
for label, content in df_temp.items():
    if pd.api.types.is_numeric_dtype(content):
        if pd.isnull(content).sum():
            df_temp[label + "_is_missing"] = pd.isnull(content)
            df_temp[label] = content.fillna(content.median())

# Turn categorical variables into numbers and fill missing-------------------------------------------
for label, content in df_temp.items():
    if not pd.api.types.is_numeric_dtype(content):
        df_temp[label + "_is_missing"] = pd.isnull(content)
        df_temp[label] = pd.Categorical(content).codes + 1  # This will extract the categorical code and fill
        # the code in the label if the data is missing then it is filled with -1 therefore we are adding +1
        # so that it becomes 0

# Splitting the data---------------------------------------------------------------------------------
df_val = df_temp[df_temp.saleYear == 2012]
df_train = df_temp[df_temp.saleYear != 2012]

X_train, y_train = df_train.drop("SalePrice", axis=1), df_train.SalePrice
X_val, y_val = df_val.drop("SalePrice", axis=1), df_val.SalePrice


# Building an evaluation function--------------------------------------------------------------------
def rmsle(y_test, y_preds):
    """
    Calculate the root mean squared log error
    :param y_test: original data with labels
    :param y_preds: predicted data with labels
    :return: root mean squared log error
    """
    return np.sqrt(mean_squared_log_error(y_test, y_preds))


def show_scores(model):
    train_preds = model.predict(X_train)
    val_preds = model.predict(X_val)
    scores = {"Training MAE": mean_absolute_error(y_train, train_preds),
              "Valid MAE": mean_absolute_error(y_val, val_preds),
              "Training RMSLE": rmsle(y_train, train_preds),
              "Valid RMSLE": rmsle(y_val, val_preds),
              "Training R^2": r2_score(y_train, train_preds),
              "Valid R^2": r2_score(y_val, val_preds)}

    return scores


# Model-----------------------------------------------------------------------------------------------
model = RandomForestRegressor(n_jobs=-1,
                              random_state=42,
                              max_samples=10000)
model.fit(X_train, y_train)
# print(show_scores(model))

# Hyper-parameter tuning-------------------------------------------------------------------------------
# rf_grid = {"n_estimators": np.arange(10, 100, 10),
#            "max_depth": [None, 3, 5, 10],
#            "min_samples_split": np.arange(2, 20, 2),
#            "min_samples_leaf": np.arange(1, 20, 2),
#            "max_features": [0.5, 1, "sqrt", "auto"],
#            "max_samples": [10000]}
#
# rs_model = RandomizedSearchCV(RandomForestRegressor(n_jobs=-1,
#                                                     random_state=42),
#                               param_distributions=rf_grid,
#                               n_iter=2,
#                               cv=5,
#                               verbose=True)
# rs_model.fit(X_train, y_train)

# Ideal model-----------------------------------------------------------------------------------------
ideal_model = RandomForestRegressor(n_estimators=40,
                                    min_samples_leaf=1,
                                    min_samples_split=14,
                                    max_features=0.5,
                                    n_jobs=-1,
                                    max_samples=None,
                                    random_state=42)
ideal_model.fit(X_train, y_train)
print(show_scores(ideal_model))

