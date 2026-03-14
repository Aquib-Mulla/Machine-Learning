# Step 1: Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error

# Step 2: Load Dataset
df = pd.read_csv(r"D:\Collage\6th Sem\ML\uber.csv\uber.csv")
print(df.head())

# Step 3: Data Preprocessing
df.dropna(inplace=True)

# Remove invalid fares
df = df[df['fare_amount'] > 0]

# Remove invalid passenger counts
df = df[(df['passenger_count'] > 0) & (df['passenger_count'] <= 6)]

# Step 4: Distance Calculation (Haversine Formula)
def haversine(lat1, lon1, lat2, lon2):
    R = 6371  # Earth radius in km
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    return R * c

# Create distance feature
df['distance_km'] = haversine(
    df['pickup_latitude'],
    df['pickup_longitude'],
    df['dropoff_latitude'],
    df['dropoff_longitude']
)

# Step 5: Outlier Detection
sns.boxplot(x=df['fare_amount'])
plt.title("Fare Amount Outliers")
plt.show()

# Remove extreme outliers
df = df[df['fare_amount'] < 100]

# Step 6: Correlation Analysis
sns.heatmap(
    df[['fare_amount', 'distance_km', 'passenger_count']].corr(),
    annot=True,
    cmap='coolwarm'
)
plt.title("Correlation Heatmap")
plt.show()

# Step 7: Feature Selection
X = df[['distance_km', 'passenger_count']]
y = df['fare_amount']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Step 8: Linear Regression Model
lr = LinearRegression()
lr.fit(X_train, y_train)

y_pred_lr = lr.predict(X_test)

r2_lr = r2_score(y_test, y_pred_lr)
rmse_lr = np.sqrt(mean_squared_error(y_test, y_pred_lr))

print("Linear Regression R²:", r2_lr)
print("Linear Regression RMSE:", rmse_lr)

# Step 9: Random Forest Regression
rf = RandomForestRegressor(
    n_estimators=100,
    random_state=42,
    n_jobs=-1
)

rf.fit(X_train, y_train)

y_pred_rf = rf.predict(X_test)

r2_rf = r2_score(y_test, y_pred_rf)
rmse_rf = np.sqrt(mean_squared_error(y_test, y_pred_rf))

print("Random Forest R²:", r2_rf)
print("Random Forest RMSE:", rmse_rf)

