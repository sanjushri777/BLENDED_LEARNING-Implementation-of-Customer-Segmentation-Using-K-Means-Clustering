# BLENDED LEARNING
# Implementation of Customer Segmentation Using K-Means Clustering

## AIM:
To implement customer segmentation using K-Means clustering on the Mall Customers dataset to group customers based on purchasing habits.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. **Import Libraries**: Import necessary libraries (`pandas`, `KMeans`, `StandardScaler`, `matplotlib`).

2. **Load Data**: Load the dataset using `pandas.read_csv()`.

3. **Select Features**: Extract features: 'Annual Income (k$)' and 'Spending Score (1-100)'.

4. **Scale Data**: Standardize the features using `StandardScaler`.

5. **Determine Optimal K**: Use the Elbow Method (plot WCSS) to find the optimal number of clusters.

6. **K-Means Clustering**: Perform K-Means clustering with `K=5` (optimal clusters).

7. **Assign Cluster Labels**: Add the cluster labels to the dataset.

8. **Visualize Clusters**: Create a scatter plot of the clusters using `Annual Income` and `Spending Score`.


## Program:
```python
/*
Program to implement customer segmentation using K-Means clustering on the Mall Customers dataset.
Developed by: SANJUSHRI A
RegisterNumber: 212223040187
*/
```python
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import  silhouette_score


df=pd.read_csv(r"C:\Users\admin\Desktop\ML-final\EX9\CustomerData.csv")
print(df.head())

features=["Annual Income (k$)","Spending Score (1-100)"]
x=df[features]

scaler=StandardScaler()
x_scaled=scaler.fit_transform(x)
wcss=[]

for i in range(1,11):
    kmeans=KMeans(n_clusters=i,random_state=42)
    kmeans.fit(x_scaled)
    wcss.append(kmeans.inertia_)
plt.figure(figsize=(8,6))
plt.plot(range(1,11),wcss,marker='o',linestyle='-')
plt.xlabel("Number of clusters")
plt.ylabel("Wcss")
plt.title("Elbow method for optimal clusters")
plt.show()

oc=4
kmeans=KMeans(n_clusters=oc,random_state=42)
kmeans.fit(x_scaled)
df["clusters"]=kmeans.labels_

sil_score=silhouette_score(x_scaled,kmeans.labels_)
print(f'Silhouette Score: {sil_score:.2f}')

plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, x='Annual Income (k$)', y='Spending Score (1-100)', hue='clusters', palette='viridis', s=100)
plt.title('Customer Segmentation based on Annual Income and Spending Score')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.legend(title='Cluster')
plt.show()


```

## Output:

![image](https://github.com/user-attachments/assets/b0f8ad57-9056-430d-b7ac-bc78efa8bb97)

![image](https://github.com/user-attachments/assets/d0b2c2d5-272b-4ea9-b43d-96206054de18)

![image](https://github.com/user-attachments/assets/958eb024-3cf3-4a61-9bbd-1b27de21e3c1)

![image](https://github.com/user-attachments/assets/fd932f91-bd28-47fc-aea2-8b777d75eb6f)





## Result:
Thus, customer segmentation was successfully implemented using K-Means clustering, grouping customers into distinct segments based on their annual income and spending score. 
