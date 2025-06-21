####------------------- IMPORTING THE REQUIRED LIBRARIIES -------------------#
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from IPython.core.pylabtools import figsize
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import KMeans

####------------------- IMPORTING THE DATASETS START -------------------------- #
df = pd.read_csv("CustomerSegmentation.csv")
print(df.head())
print(df.shape)
print(df.columns)
####------------------- IMPORTING THE DATASETS END  -------------------------- #

####------------------- DATA PRE-PROCESSING START -------------------------- #
#============================================================================#
####---------------- TO CHECK THE NULL VALUES IN EACH COLUMN ---------#
for each_column in df.columns:
    null_column = df[each_column].isnull().sum()
    if null_column > 0:
        print(f"Column {each_column} contains {null_column} null values")
####---------------- TO REMOVE THE NULL VALUES ---------#
df = df.dropna()
print("Total Value of dataset after removing the null values", len(df))

####---------------- TO FIND THE UNIQUE VALUES IN EACH COLUMNS ---------#
print("No. of Unique values each columns", df.nunique())
#### As here Column "Z_CostContact" and "Z_Revenue" has same value as 1, it can be droped. Also ID Column is Unique for each row its has no importance.
#### Date Columns also can be split into 3 columns as Date, Month and Year.

####---------------- SPLIT THE DATE COLUMNS IN TO 3COLUMNS ---------#
col_split = df["Dt_Customer"].str.split("-" or "/",n=3, expand=True)   # Date with "-" and "/" signs are split.
df["Date"] = col_split[0].astype('int')
df["Month"] = col_split[1].astype('int')
df["Year"] = col_split[2].astype('int')
# print(df.nunique())
# print(df.shape)

####---------------- REMOVE THE COLUMNS WHICH ARE NOT REQUIRED FOR THE ANALYSIS ---------#
print("Shape of dataset before removing the columns", df.shape)
df.drop(['ID','Dt_Customer','Z_CostContact','Z_Revenue'], axis=1, inplace=True)
print("Shape of dataset after removing the columns", df.shape)

####------------------- DATA PRE-PROCESSING END -------------------------- #


####------------------- DATA VISUALIZATION AND ANALYSIS START -------------------------- #
#========================================================================================#

####------------ First we will get columns based datatypes ------------------------ #

floats,objects = [],[]
for each_column in df.columns:
    if df[each_column].dtype==object:
        objects.append((each_column))
    if df[each_column].dtype==float:
        floats.append((each_column))

print("Columns with Object datatypes are ",objects)
print("Columns with floats datatypes are ",floats)

#========================================================================================#

####----------- TO GET THE COUNT PLOT FOR THE ABOVE OBJECT DATA TYPE COLUMNS ------------#
#========================================================================================#

plt.figure(figsize=(20, 15))
plt.title("COUNT PLOT FOR OBJECT DATA TYPE")
for i, each_column in enumerate(objects):
    plt.subplot(2,2, i+1)
    sns.countplot(x=df[each_column], palette='Set1', hue=df[each_column])
#plt.show()
print("Marital Status ", df['Marital_Status'].value_counts())
#========================================================================================#


####----------- LABEL ENCODING, TO CONVERT THE CATEGORICAL DATA INTO NUMERICAL DATA ------------#
#========================================================================================#
for each_column in df.columns:
    if df[each_column].dtype==object:
        label_encoder = LabelEncoder()
        df[each_column] = label_encoder.fit_transform(df[each_column])
        print("Columns After Label Encoding", df[each_column])

#========================================================================================#
####---------------------FIND THE DATA CORRELATION USING HEAT MAP GENRATION (ONLY FOR NUMERICAL VALUES) --------------------------------------------#
#=============================================================================# ===========#
# Assuming df is your DataFrame
data_correlation = df.corr()

# Convert the DataFrame to a NumPy array
data_correlation_numpy = data_correlation.to_numpy()

# Create a figure and axis for the heatmap
fig, axis = plt.subplots(figsize=(25, 25))  # Increase the figure size
HM_image = axis.imshow(data_correlation_numpy, cmap="Reds")

# Get the feature names
Features = list(data_correlation.columns)

# Set the labels for the x and y axes
axis.set_xticks(np.arange(len(Features)))
axis.set_yticks(np.arange(len(Features)))
axis.set_xticklabels(Features)
axis.set_yticklabels(Features)

# Rotate the x labels so they don't overlap
plt.setp(axis.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

# Add a colorbar
fig.colorbar(HM_image)

# Plot the correlation values in each cell
for i in range(len(Features)):
    for j in range(len(Features)):
        text = axis.text(j, i, "%.2f" % data_correlation_numpy[i, j], ha="center", va="center", color="black", rotation=45)

# Show the heatmap
plt.show()


#### --------------------------------------------------------------------

#========================================================================================#
####--------------------- STANDARDARTIZATION USING KMEAN --------------------------------------------#
#========================================================================================#
### Example - Find the Mean and Standard Deviation of Income , MntWines and MntMeatProducts Before Standardatization
# Print the mean and standard deviation before standardization
print(f" Before Standardization :\n  Income mean :{ np.mean(df['Income']) } \n  Income std : {np.std(df['Income']) }")
print("#========================================================================================#")
print(f" Before Standardization :\n  MntWines mean : { np.mean(df['MntWines']) } \n  MntWines std : {np.std(df['MntWines']) }")
print("#========================================================================================#")
print(f" Before Standardization :\n  MntMeatProducts mean : { np.mean(df['MntMeatProducts']) } \n  MntMeatProducts std : {np.std(df['MntMeatProducts']) }")
print("#========================================================================================#")

# Scatter plot of Income vs MntWines before Standaradization
sns.scatterplot(x=df.Income,y=df.MntWines, data=df, s=100)
plt.title("'Scatter plot of Income vs MntWines' Before Standardization")
plt.show()

#========================================================================================#"
### Compute the Mean and Standard Deviation and Rescale the Mean and Std Deviation data for the selected columns
std_scaler = StandardScaler() ### Instantiate the Standard Scalar
scaled_data = std_scaler.fit_transform(df[["Income","MntWines","MntMeatProducts"]])
print(scaled_data)
print("#========================================================================================#")
### Example - Find the Mean and Standard Deviation of Income , MntWines and MntMeatProducts After Standardatization
print(f" After Standardization :\n  Income mean : { scaled_data[:,0].mean()} \n Income std : {scaled_data[:,0].std() }")
print("#========================================================================================#")
print(f" After Standardization :\n  MntWines mean : { scaled_data[:,1].mean() } \n MntWines std : {scaled_data[:,1].std() }")
print("#========================================================================================#")
print(f" After Standardization :\n  MntMeatProducts mean : { scaled_data[:,2].mean() } \n MntMeatProducts std : {scaled_data[:,2].std()}")
print("#========================================================================================#")

#========================================================================================#"

### Compute the Mean and Standard Deviation and Rescale the Mean and Std Deviationdata for all the columns
# Scatter plot of Income vs MntWines before Standaradization
sns.scatterplot(x=scaled_data[:,0],y=scaled_data[:,1], s=100)
plt.title("'Scatter plot of Income vs MntWines' After Standardization")
plt.show()
#========================================================================================#"

########### SEGMENTATION USING T-Distributed Stochastic Neighbor Embedding ###################

from sklearn.manifold import TSNE
model = TSNE(n_components=2, random_state=0)
tsne_data = model.fit_transform(df)
plt.figure(figsize=(7, 7))
plt.scatter(tsne_data[:, 0], tsne_data[:, 1])
plt.show()

#========================================================================================#"
#### Find K Value Based on ELBOW Method #################################################
cluster = []
for n_clusters in range(1, 21):
    model = KMeans(init='k-means++', n_clusters=n_clusters, max_iter=500, random_state=22)
    model.fit(df)
    cluster.append(model.inertia_)  # inertia is sum of squared distances within the clusters.
    print(f"Number of clusters: {n_clusters}, Inertia: {model.inertia_}")

# Check lengths
print(f"Length of cluster list: {len(cluster)}")
print(f"Range length: {len(range(1, 21))}")

# Plotting the cluster values
plt.figure(figsize=(10, 5))
sns.lineplot(x=range(1, 21), y=cluster, marker='o')
plt.title('Elbow Method for Optimal k')
plt.xlabel('Number of clusters')
plt.ylabel('Inertia')
plt.show()
### Here by using the elbow method we can say that k = 6 is the optimal number of clusters that should be made as
### after k = 6 the value of the inertia is not decreasing drastically.
#========================================================================================#"

model = KMeans(init='k-means++',n_clusters=5,max_iter=500,random_state=22)
segments = model.fit_predict(df)
plt.figure(figsize=(7, 7))
df_tsne = pd.DataFrame({'x': tsne_data[:, 0], 'y': tsne_data[:, 1], 'segment': segments})
sns.scatterplot(x='x', y='y', hue='segment', data=df_tsne)
plt.show()

#========================================================================================#"