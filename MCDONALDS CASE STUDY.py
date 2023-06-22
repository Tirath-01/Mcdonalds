#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np

# Load the data
mcdonalds = pd.read_csv("mcdonalds.csv")

# Print column names
print(mcdonalds.columns)

# Get the dimensions of the data
print(mcdonalds.shape)

# Print the first 3 rows
print(mcdonalds.head(3))


# In[2]:


import pandas as pd
import numpy as np

# Convert mcdonalds DataFrame to a matrix
MD_x = mcdonalds.iloc[:, 0:11].values

# Convert "Yes" to 1 and "No" to 0
MD_x = np.where(MD_x == "Yes", 1, 0)

# Calculate column means
column_means = np.round(np.mean(MD_x, axis=0), 2)

# Create a DataFrame to display the results
column_means_df = pd.DataFrame({"Column": mcdonalds.columns[0:11], "Mean": column_means})

# Print the column means
print(column_means_df)


# In[3]:


from sklearn.decomposition import PCA

# Perform PCA on MD_x
pca = PCA()
MD_pca = pca.fit_transform(MD_x)

# Get the summary of the PCA results
explained_variance_ratio = pca.explained_variance_ratio_
cumulative_variance_ratio = np.cumsum(explained_variance_ratio)

# Create a DataFrame to display the results
summary_df = pd.DataFrame({
    "PC": range(1, len(explained_variance_ratio) + 1),
    "Standard deviation": np.round(np.sqrt(pca.explained_variance_), 4),
    "Proportion of Variance": np.round(explained_variance_ratio, 4),
    "Cumulative Proportion": np.round(cumulative_variance_ratio, 4)
})

# Print the summary
print(summary_df)


# In[4]:


from sklearn.decomposition import PCA

# Perform PCA on MD_x
pca = PCA()
MD_pca = pca.fit_transform(MD_x)

# Print the standard deviations
print("Standard deviations (1, .., p={0}):".format(pca.n_components_))
print(pca.explained_variance_.round(1))

# Print the rotation matrix
print("Rotation (n x k) = ({0} x {1}):".format(*pca.components_.shape))
rotation_df = pd.DataFrame(pca.components_, columns=mcdonalds.columns[:11])
print(rotation_df.round(2))


# In[5]:


import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# Perform PCA on MD_x
pca = PCA()
MD_pca = pca.fit_transform(MD_x)

# Plot the PCA results
plt.scatter(MD_pca[:, 0], MD_pca[:, 1], c='grey')
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.title("PCA Plot")
plt.show()

# Plot the projected axes
plt.plot([0, pca.components_[0, 0]], [0, pca.components_[0, 1]], 'r-', label="PC1")
plt.plot([0, pca.components_[1, 0]], [0, pca.components_[1, 1]], 'b-', label="PC2")
plt.xlabel("Variable")
plt.ylabel("Projection")
plt.title("Projected Axes")
plt.legend()
plt.show()


# In[6]:


import numpy as np
from sklearn.cluster import KMeans

# Set the random seed
np.random.seed(1234)

# Perform K-means clustering
k_range = range(2, 9)  # Cluster sizes 2 to 8
best_score = None
best_labels = None

for k in k_range:
    kmeans = KMeans(n_clusters=k, n_init=10, random_state=1234)
    kmeans.fit(MD_x)
    score = kmeans.inertia_
    if best_score is None or score < best_score:
        best_score = score
        best_labels = kmeans.labels_

# Relabel the clusters
unique_labels = np.unique(best_labels)
relabel_dict = {label: i for i, label in enumerate(unique_labels)}
relabeled_labels = np.array([relabel_dict[label] for label in best_labels])


# In[7]:


import matplotlib.pyplot as plt

# Plot the clustering results
plt.plot(range(2, 9),marker='o')
plt.xlabel("Number of Segments")
plt.ylabel("Clustering Score")
plt.title("Clustering Results")
plt.show()


# In[8]:


import matplotlib.pyplot as plt

# Plot the clustering results
plt.plot(range(2, 9),marker='o')
plt.xlabel("Number of Segments")
plt.ylabel("Clustering Score")
plt.title("Clustering Results")
plt.show()


# In[9]:


df = pd.read_csv("mcdonalds.csv")
df.shape
df.head()
df.dtypes
# 11 variable(cols) has yes or no values.

# checking for null data --> No null data
df.info()
df.isnull().sum()


# In[10]:


df['Gender'].value_counts()
df['VisitFrequency'].value_counts()
df['Like'].value_counts()


# In[11]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


# In[12]:


#Customer segmentation - based on socio-demographs (Age & Gender)

#Gender
labels = ['Female', 'Male']
size = df['Gender'].value_counts()
colors = ['pink', 'cyan']
explode = [0, 0.1]
plt.rcParams['figure.figsize'] = (7, 7)
plt.pie(size, colors = colors, explode = explode, labels = labels, shadow = True, autopct = '%.2f%%')
plt.title('Gender', fontsize = 20)
plt.axis('off')
plt.legend()
plt.show()
#we infer that there are more female customers than male.

#Age
plt.rcParams['figure.figsize'] = (25, 8)
f = sns.countplot(x=df['Age'],palette = 'hsv')
f.bar_label(f.containers[0])
plt.title('Age distribution of customers')
plt.show()


# In[13]:


#Customer segmentation - based on pyschographic segmentation

#For convinence renaming the category
df['Like']= df['Like'].replace({'I hate it!-5': '-5','I love it!+5':'+5'})
#Like 
sns.catplot(x="Like", y="Age",data=df, 
            orient="v", height=5, aspect=2, palette="Set2",kind="swarm")
plt.title('Likelyness of McDonald w.r.t Age')
plt.show()


# In[14]:


from sklearn.preprocessing import LabelEncoder
def labelling(x):
    df[x] = LabelEncoder().fit_transform(df[x])
    return df

cat = ['yummy', 'convenient', 'spicy', 'fattening', 'greasy', 'fast', 'cheap',
       'tasty', 'expensive', 'healthy', 'disgusting']

for i in cat:
    labelling(i)
df


# In[15]:


#Histogram of the each attributes
plt.rcParams['figure.figsize'] = (12,14)
df.hist()
plt.show()


# In[16]:


#Considering only first 11 attributes
df_eleven = df.loc[:,cat]
df_eleven


# In[17]:


#Considering only the 11 cols and converting it into array
x = df.loc[:,cat].values
x


# In[18]:


from sklearn.decomposition import PCA
from sklearn import preprocessing

pca_data = preprocessing.scale(x)

pca = PCA(n_components=11)
pc = pca.fit_transform(x)
names = ['pc1','pc2','pc3','pc4','pc5','pc6','pc7','pc8','pc9','pc10','pc11']
pf = pd.DataFrame(data = pc, columns = names)
pf


# In[19]:


#Proportion of Variance (from PC1 to PC11)
pca.explained_variance_ratio_


# In[20]:


np.cumsum(pca.explained_variance_ratio_)


# In[21]:


loadings = pca.components_
num_pc = pca.n_features_
pc_list = ["PC"+str(i) for i in list(range(1, num_pc+1))]
loadings_df = pd.DataFrame.from_dict(dict(zip(pc_list, loadings)))
loadings_df['variable'] = df_eleven.columns.values
loadings_df = loadings_df.set_index('variable')
loadings_df


# In[22]:


#Correlation matrix plot for loadings 
plt.rcParams['figure.figsize'] = (20,15)
ax = sns.heatmap(loadings_df, annot=True, cmap='Spectral')
plt.show()


# In[23]:


#Scree plot (Elbow test)- PCA
from bioinfokit.visuz import cluster
cluster.screeplot(obj=[pc_list, pca.explained_variance_ratio_],show=True,dim=(10,5))


# In[24]:


# get PC scores
pca_scores = PCA().fit_transform(x)

# get 2D biplot
cluster.biplot(cscore=pca_scores, loadings=loadings, labels=df.columns.values, var1=round(pca.explained_variance_ratio_[0]*100, 2),
    var2=round(pca.explained_variance_ratio_[1]*100, 2),show=True,dim=(10,5))


# In[25]:


#Extracting segments

#Using k-means clustering analysis
from sklearn.cluster import KMeans
from yellowbrick.cluster import KElbowVisualizer
model = KMeans()
visualizer = KElbowVisualizer(model, k=(1,12)).fit(df_eleven)
visualizer.show()


# In[26]:


#K-means clustering 

kmeans = KMeans(n_clusters=4, init='k-means++', random_state=0).fit(df_eleven)
df['cluster_num'] = kmeans.labels_ #adding to df
print (kmeans.labels_) #Label assigned for each data point
print (kmeans.inertia_) #gives within-cluster sum of squares. 
print(kmeans.n_iter_) #number of iterations that k-means algorithm runs to get a minimum within-cluster sum of squares
print(kmeans.cluster_centers_) #Location of the centroids on each cluster. 


# In[27]:


#To see each cluster size
from collections import Counter
Counter(kmeans.labels_)


# In[28]:


#Visulazing clusters
sns.scatterplot(data=pf, x="pc1", y="pc2", hue=kmeans.labels_)
plt.scatter(kmeans.cluster_centers_[:,0], kmeans.cluster_centers_[:,1], 
            marker="X", c="r", s=80, label="centroids")
plt.legend()
plt.show()


# In[29]:


from statsmodels.graphics.mosaicplot import mosaic
from itertools import product

crosstab =pd.crosstab(df['cluster_num'],df['Like'])
#Reordering cols
crosstab = crosstab[['-5','-4','-3','-2','-1','0','+1','+2','+3','+4','+5']]
crosstab 


# In[30]:


#MOSAIC PLOT
plt.rcParams['figure.figsize'] = (7,5)
mosaic(crosstab.stack())
plt.show()


# In[31]:


#Mosaic plot gender vs segment
crosstab_gender =pd.crosstab(df['cluster_num'],df['Gender'])
crosstab_gender


# In[32]:


plt.rcParams['figure.figsize'] = (7,5)
mosaic(crosstab_gender.stack())
plt.show()


# In[33]:


#box plot for age

sns.boxplot(x="cluster_num", y="Age", data=df)


# In[34]:


#Calculating the mean
#Visit frequency
df['VisitFrequency'] = LabelEncoder().fit_transform(df['VisitFrequency'])
visit = df.groupby('cluster_num')['VisitFrequency'].mean()
visit = visit.to_frame().reset_index()
visit


# In[35]:


#Like
df['Like'] = LabelEncoder().fit_transform(df['Like'])
Like = df.groupby('cluster_num')['Like'].mean()
Like = Like.to_frame().reset_index()
Like


# In[36]:


#Gender
df['Gender'] = LabelEncoder().fit_transform(df['Gender'])
Gender = df.groupby('cluster_num')['Gender'].mean()
Gender = Gender.to_frame().reset_index()
Gender


# In[37]:


segment = Gender.merge(Like, on='cluster_num', how='left').merge(visit, on='cluster_num', how='left')
segment


# In[38]:


plt.figure(figsize = (9,4))
sns.scatterplot(x = "VisitFrequency", y = "Like",data=segment,s=400, color="r")
plt.title("Simple segment evaluation plot for the fast food data set",
          fontsize = 15) 
plt.xlabel("Visit", fontsize = 12) 
plt.ylabel("Like", fontsize = 12) 
plt.show()


# In[ ]:




