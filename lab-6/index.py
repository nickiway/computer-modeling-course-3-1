from tkinter import N
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from sklearn import metrics
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from tabulate import tabulate 
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

path = './Wine.csv'
dataset = pd.read_csv(path)

wine_class = dataset.iloc[:, 0].values
wine_size = dataset.iloc[:, 1:].values

# scaling
scaler = MinMaxScaler()
scaler_wine_size = scaler.fit_transform(wine_size)

clusterer = KMeans(n_clusters=3)
outputs_array = [False, False]

def task_wrapper(wine_size, label, outputed_index):
    # TASK 2
    # clustering with euclidean metric
    print(label)
    clusterer.fit(wine_size)
    
    labels = clusterer.labels_
    metrics.silhouette_score(wine_size, labels, metric='euclidean')

    predictions = clusterer.predict(wine_size)

    dataset['clusterer']=predictions
    centroids = clusterer.cluster_centers_

    # TASK 4 COUNTING CLUSTERS
    count_clusters = Counter(labels)

    # getting the number of object of class in every cluster
    cluster_content = dataset.groupby(["clusterer", "Wine"]).size().unstack(fill_value=0)
    cluster_content["Total"] = cluster_content.sum(axis=1)
    cluster_content.loc["Total"] = cluster_content.sum()
    
    # To output the task 1 once only
    if outputs_array[outputed_index] == False:
        print('results: ')
        print(dataset, "\n")
        
        print("Coordinates of centroids:")
        print(centroids, "\n")
        
        print("The amount in clusters")
        print(count_clusters, "\n")
        print(tabulate(cluster_content, headers="keys", tablefmt="psql"))
        
        outputs_array[outputed_index] = True
    
    # TASK 3
    fig, ax = plt.subplots()
    fig.suptitle(f"{label}")
    scatter1 = ax.scatter(wine_size[:, index[0]], wine_size[:, index[1]], c=predictions, s=15, cmap="brg")
    handles, labels_scatter = scatter1.legend_elements()
    legend1 = ax.legend(handles, labels_scatter, loc = "upper right")
    ax.add_artist(legend1)
    
    scatter2 = ax.scatter(centroids[:, index[0]], centroids[:, index[1]], marker="x", c="purple", s=200, linewidths=3, label="centroids")
    plt.legend(loc="lower right")
    plt.xlabel(f"{dataset.columns[0]} {dataset.columns[index[0]]}")
    plt.ylabel(f"{dataset.columns[1]} {dataset.columns[index[1]]}")
    plt.show()  
    
    
index_list =[[2,4], [4,6], [1, 3], [5, 7]]

for index in index_list:
    # task wrapper to watch the result after scaling and before scaling
    task_wrapper(wine_size=wine_size, label="Task without scaling", outputed_index=0)
    task_wrapper(wine_size=scaler_wine_size, label="Task after scaling",  outputed_index=1)    
    
def calculate_best_cluster_number(wine_size, label):
    print(label)
    df = pd.DataFrame(columns=["Number of clusters", "WCSS", "Silhouette", "DB"])
    for i in range(2, 11):
        clusterer_i = KMeans(n_clusters=i).fit(wine_size)
        predictions_i = clusterer_i.predict(wine_size)
        
        WCSS=clusterer_i.inertia_
        Silhouette=metrics.silhouette_score(wine_size,predictions_i)
        DB=metrics.davies_bouldin_score(wine_size, predictions_i)
        new_row_df = pd.DataFrame([[i, WCSS, Silhouette, DB]], columns=df.columns)
        df = pd.concat([df, new_row_df], ignore_index=True)
        
    print(tabulate(df, headers="keys", tablefmt="psql", floatfmt=".3f"))
    
    
    # метод ліктя
    plt.plot(df["Number of clusters"], df["WCSS"], marker='o', linestyle="None", label="WCSS")
    plt.xlabel("Number of clusters")
    plt.xlabel("WCSS")
    plt.title('Метод ліктя')
    plt.legend()
    plt.show()
    
    # метод силуету
    plt.plot(df["Number of clusters"], df["Silhouette"], marker='o', linestyle="None", label="Silhouette")
    plt.xlabel("Number of clusters")
    plt.xlabel("Silhouette")
    plt.title('Метод силуету')
    plt.legend()
    plt.show()
    
    # метод силуету
    plt.plot(df["Number of clusters"], df["DB"], marker='o', linestyle="None", label="DB")
    plt.xlabel("Number of clusters")
    plt.xlabel("DB")
    plt.title('Метод Девіса-Булдінга')
    plt.legend()
    plt.show()
    
calculate_best_cluster_number(wine_size=wine_size, label="Without scaling searching the optimal number of clusters")
calculate_best_cluster_number(wine_size=scaler_wine_size, label="With scaling searching the optimal number of clusters")