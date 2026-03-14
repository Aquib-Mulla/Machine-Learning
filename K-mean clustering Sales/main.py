# Experiment 7
# Implement K-Means clustering and Hierarchical clustering
# Determine optimal clusters using the Elbow Method

# Step 1: Import libraries
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from scipy.cluster.hierarchy import dendrogram, linkage

# Step 2: Load dataset
df = pd.read_csv("sales_data_sample.csv", encoding='ISO-8859-1')

# Step 3: Display dataset information
print("First 5 rows of dataset:")
print(df.head())

print("\nDataset Shape:")
print(df.shape)

print("\nDataset Columns:")
print(df.columns)

# Step 4: Select numerical columns for clustering
data = df[['SALES', 'QUANTITYORDERED', 'PRICEEACH']]

# Step 5: Normalize data
scaler = StandardScaler()
scaled_data = scaler.fit_transform(data)

# Step 6: Elbow Method to find optimal number of clusters
wcss = []

for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, random_state=42)
    kmeans.fit(scaled_data)
    wcss.append(kmeans.inertia_)

# Plot Elbow Graph
plt.figure()
plt.plot(range(1, 11), wcss, marker='o')
plt.title("Elbow Method for Optimal Clusters")
plt.xlabel("Number of Clusters")
plt.ylabel("WCSS")
plt.show()

# Step 7: Apply K-Means with optimal clusters (example = 3)
kmeans = KMeans(n_clusters=3, random_state=42)
clusters = kmeans.fit_predict(scaled_data)

# Add cluster column to dataset
df['Cluster'] = clusters

print("\nClustered Data Sample:")
print(df[['SALES','QUANTITYORDERED','PRICEEACH','Cluster']].head())

# Step 8: Visualize clusters
plt.figure()
plt.scatter(df['SALES'], df['QUANTITYORDERED'], c=df['Cluster'])
plt.title("K-Means Clustering Result")
plt.xlabel("Sales")
plt.ylabel("Quantity Ordered")
plt.show()

# Step 9: Hierarchical Clustering
linked = linkage(scaled_data, method='ward')

plt.figure(figsize=(10,5))
dendrogram(linked)
plt.title("Hierarchical Clustering Dendrogram")
plt.xlabel("Data Points")
plt.ylabel("Distance")
plt.show()