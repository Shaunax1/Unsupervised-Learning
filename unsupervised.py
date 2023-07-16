import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

data = pd.read_csv("wine-clustering.csv")

data.head()

data.isna().sum()

sns.set(style='white',font_scale=1.3, rc={'figure.figsize':(20,20)})
ax=data.hist(bins=20,color='red')


#%% K-Means Clustering

data2 = data.iloc[:,[0,1]]

from sklearn.cluster import KMeans

wcss = []
for k in range(1,15):
    kmeans = KMeans(n_clusters = k)
    kmeans.fit(data2)
    wcss.append(kmeans.inertia_)
    
plt.plot(range(1,15),wcss,marker="+")
plt.xlabel("number of k(cluster) value")
plt.ylabel("wcss")
plt.show()


kmeans = KMeans(n_clusters = 2)
kmeans.fit(data2)
labels = kmeans.predict(data2)

plt.scatter(data2["Alcohol"],data2["Malic_Acid"], c = labels)
plt.xlabel("Alcohol")
plt.ylabel("Malic Acid")
plt.show()


kmeans2 = KMeans(n_clusters=2)
clusters = kmeans2.fit_predict(data2)
data["label"] = clusters

plt.scatter(data.Alcohol[data.label == 0],data.Malic_Acid[data.label == 0 ],color="red")
plt.scatter(data.Alcohol[data.label == 1],data.Malic_Acid[data.label == 1 ],color="green")
plt.scatter(kmeans2.cluster_centers_[:,0],kmeans2.cluster_centers_[:,1],color="orange",marker="+",s=500,linewidths=3)
plt.show()

#%% Standardization

data3 = data.drop(["label"],axis=1)
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
scalar = StandardScaler()
kmeans = KMeans(n_clusters = 2)
pipe = make_pipeline(scalar,kmeans)
pipe.fit(data3)
labels = pipe.predict(data3)
df = pd.DataFrame({'labels':labels, "label" : data['label']})
ct = pd.crosstab(df['labels'],df['label'])
print(ct)

#%% Hierarcial Clustering

from scipy.cluster.hierarchy import linkage,dendrogram

merg = linkage(data3,method="ward")
dendrogram(merg,leaf_rotation=90)
plt.xlabel("data point")
plt.ylabel("euclidean distance")
plt.show()

from sklearn.cluster import AgglomerativeClustering

hc = AgglomerativeClustering(n_clusters=2, affinity="euclidean",linkage="ward")
cluster = hc.fit_predict(data2)

data["label"] = cluster
plt.scatter(data.Alcohol[data.label == 0],data.Malic_Acid[data.label == 0 ],color="red")
plt.scatter(data.Alcohol[data.label == 1],data.Malic_Acid[data.label == 1 ],color="green")
plt.scatter(kmeans2.cluster_centers_[:,0],kmeans2.cluster_centers_[:,1],color="orange",marker="+",s=500,linewidths=3)
plt.show()

