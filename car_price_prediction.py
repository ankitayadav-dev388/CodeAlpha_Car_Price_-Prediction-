import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# 1. Load Dataset
url = "https://raw.githubusercontent.com/jey1987/DATA605/master/CarPrice_Assignment.csv"

df = pd.read_csv(url)

print("Dataset loaded successfully!")
print("Shape:", df.shape)

print("\nFirst 5 Rows:")
print(df.head())


# 2. Dataset Information
print("\nDataset Information:")
df.info()

print("\nMissing Values:")
print(df.isnull().sum())

print("\nDuplicate Rows:")
print(df.duplicated().sum())

print("\nStatistical Summary:")
print(df.describe())


# 3. Data Cleaning
df = df.drop(columns=["car_ID"])

print("\nColumns:")
print(df.columns.tolist())


# 4. Price Distribution
plt.figure(figsize=(8, 5))
sns.histplot(df["price"], kde=True)
plt.title("Car Price Distribution")
plt.xlabel("Price")
plt.ylabel("Number of Cars")
plt.show()


# 5. Horsepower vs Price
plt.figure(figsize=(8, 5))
sns.scatterplot(data=df, x="horsepower", y="price")
plt.title("Horsepower vs Car Price")
plt.xlabel("Horsepower")
plt.ylabel("Price")
plt.show()


# 6. Prepare Features and Target
X = df.drop(columns=["price"])
y = df["price"]

print("Features:", X.shape)
print("Target:", y.shape)


# 7. Identify Numerical and Categorical Features
numeric_features = X.select_dtypes(
    include=["int64", "float64"]
).columns

categorical_features = X.select_dtypes(
    include=["object"]
).columns

print("\nNumerical Features:")
print(list(numeric_features))

print("\nCategorical Features:")
print(list(categorical_features))


# 8. Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("\nTraining data:", X_train.shape)
print("Testing data:", X_test.shape)


# 9. Preprocessing
preprocessor = ColumnTransformer(
    transformers=[
        ("num", "passthrough", numeric_features),
        (
            "cat",
            OneHotEncoder(handle_unknown="ignore"),
            categorical_features
        )
    ]
)


# 10. Random Forest Regression Model
model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        (
            "regressor",
            RandomForestRegressor(
                n_estimators=200,
                random_state=42
            )
        )
    ]
)

model.fit(X_train, y_train)

print("\nModel trained successfully!")


# 11. Prediction
y_pred = model.predict(X_test)

print("\nActual Prices:")
print(y_test.head().values)

print("\nPredicted Prices:")
print(y_pred[:5])


# 12. Model Evaluation
mae = mean_absolute_error(y_test, y_pred)

rmse = np.sqrt(
    mean_squared_error(y_test, y_pred)
)

r2 = r2_score(y_test, y_pred)

print("\nModel Evaluation")
print("-------------------------")
print("MAE:", round(mae, 2))
print("RMSE:", round(rmse, 2))
print("R2 Score:", round(r2, 4))


# 13. Actual vs Predicted Price
plt.figure(figsize=(8, 5))

plt.scatter(y_test, y_pred)

plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("Actual vs Predicted Car Prices")

plt.show()


# 14. Predict Price of a New Car
new_car = pd.DataFrame({
    "symboling": [0],
    "CarName": ["toyota camry"],
    "fueltype": ["gas"],
    "aspiration": ["std"],
    "doornumber": ["four"],
    "carbody": ["sedan"],
    "drivewheel": ["fwd"],
    "enginelocation": ["front"],
    "wheelbase": [100.0],
    "carlength": [180.0],
    "carwidth": [68.0],
    "carheight": [55.0],
    "curbweight": [2800],
    "enginetype": ["ohc"],
    "cylindernumber": ["four"],
    "enginesize": [130],
    "fuelsystem": ["mpfi"],
    "boreratio": [3.2],
    "stroke": [3.2],
    "compressionratio": [9.0],
    "horsepower": [110],
    "peakrpm": [5500],
    "citympg": [25],
    "highwaympg": [30]
})

predicted_price = model.predict(new_car)

print("\nPredicted Car Price:",
      round(predicted_price[0], 2))


# 15. Final Result
print("\n===================================")
print("      CAR PRICE PREDICTION")
print("===================================")
print("Model: Random Forest Regression")
print("R2 Score:", round(r2, 4))
print("MAE:", round(mae, 2))
print("RMSE:", round(rmse, 2))
print("\nProject completed successfully!")
