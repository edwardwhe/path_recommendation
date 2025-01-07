import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns

# Sample dataset loading (replace with your actual data file)
# Assume CSV format: student_id, exercises_completed, average_time_per_exercise, average_score
data = pd.read_csv('student_exercise_log.csv')

# Select relevant features for clustering
features = data[['exercises_completed', 'average_time_per_exercise', 'average_score']]

# Data normalization
scaler = StandardScaler()
normalized_features = scaler.fit_transform(features)

# Finding the optimal number of clusters using the elbow method
wcss = []  # Within-cluster sum of squares
for k in range(1, 11):
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(normalized_features)
    wcss.append(kmeans.inertia_)

# Plotting the elbow curve
plt.figure(figsize=(8, 5))
plt.plot(range(1, 11), wcss, marker='o')
plt.title('Elbow Method for Optimal k')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('WCSS')
plt.show()

# Choose an optimal k based on the elbow method (e.g., k = 3)
optimal_k = 3
kmeans = KMeans(n_clusters=optimal_k, random_state=42)
clusters = kmeans.fit_predict(normalized_features)

# Add the cluster labels to the original data
data['Cluster'] = clusters

# Visualize the clusters (only for 2D or 3D, adjust if necessary)
sns.scatterplot(x='exercises_completed', y='average_score', hue='Cluster', data=data, palette='viridis')
plt.title('Student Clusters based on Exercise Performance')
plt.show()

# Display the cluster centers (useful for understanding each group's characteristics)
cluster_centers = scaler.inverse_transform(kmeans.cluster_centers_)
cluster_df = pd.DataFrame(cluster_centers, columns=features.columns)
print("Cluster Centers:")
print(cluster_df)