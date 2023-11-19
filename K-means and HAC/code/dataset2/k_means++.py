import pandas as pd
import random
import matplotlib.pyplot as plt
import numpy as np
import os

def plot_clusters(clusters, centroids, k, filename):
    save_name = f"clusters(k={k}).png"

    # Create the directory
    if not os.path.exists(filename):
        os.makedirs(filename)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Plotting the clusters
    for cluster in clusters:
        x = [point[0] for point in cluster]
        y = [point[1] for point in cluster]
        z = [point[2] for point in cluster]
        ax.scatter(x, y, z)

    # Plotting the centroids
    centroid_x = [centroid[0] for centroid in centroids]
    centroid_y = [centroid[1] for centroid in centroids]
    centroid_z = [centroid[2] for centroid in centroids]
    ax.scatter(centroid_x, centroid_y, centroid_z, marker='x', color='red')

    # Save the figure
    plt.savefig(os.path.join(filename, save_name))
    plt.close()

def plot_k_mean(mean_list, k_list, filename):
    save_name = f"line_graph.png"
    
    if not os.path.exists(filename):
        os.makedirs(filename)
    
    # Plotting the line
    plt.plot(k_list, mean_list, marker='o')  # Add a marker for clarity

    # Adding labels and title for clarity
    plt.xlabel('Number of Clusters (k)')
    plt.ylabel('Average Cost')
    plt.title('K-means Clustering: Average Cost per Number of Clusters')

    plt.savefig(os.path.join(filename, save_name))
    plt.close() 

def updateCentroids(clusters):
    centroids = []
    for cluster in clusters:
        if not cluster: 
            continue

        sum_x = sum_y = sum_z = 0
        num_points = len(cluster)

        for point in cluster:
            sum_x += point[0]
            sum_y += point[1]
            sum_z += point[2]

        mean_x = sum_x / num_points
        mean_y = sum_y / num_points
        mean_z = sum_z / num_points

        centroids.append([mean_x, mean_y, mean_z])
    
    return centroids

def euclidean_distance(point1, point2):
    return sum((p1 - p2) ** 2 for p1, p2 in zip(point1, point2)) ** 0.5

def read_dataset(file):
    dataset = file

    return pd.read_csv(dataset)

def initialize_centroids_plusplus(points, k):
    centroids = [random.choice(points)]

    for _ in range(1, k):
        distances = [min(euclidean_distance(point, centroid) ** 2 for centroid in centroids) for point in points]
        total_distance = sum(distances)
        distances = [d / total_distance for d in distances]

        # Choose a new centroid based on weighted probabilities
        new_centroid = random.choices(points, weights=distances, k=1)[0]
        centroids.append(new_centroid)

def k_means(df, k, max_iters=100):
   
    points = [tuple(x) for x in df.to_numpy()]
    centroids = random.sample(points, k)
    point_assignments = [-1] * len(points)

    for _ in range(max_iters):
        change_flag = False
        clusters = [[] for _ in range(k)]

        for i, point in enumerate(points):
            
            distances = [euclidean_distance(point, centroid) for centroid in centroids]
            closest_centroid = distances.index(min(distances))

            # Check if the point's assignment has changed
            if point_assignments[i] != closest_centroid:
                change_flag = True
                point_assignments[i] = closest_centroid

            clusters[closest_centroid].append(point)
            
        # If no points changed their cluster, convergence is reached
        if not change_flag:
            break
        
        centroids = updateCentroids(clusters)

    cluster_costs = []
    for idx, cluster in enumerate(clusters):
        cost = sum(euclidean_distance(point, centroids[idx]) ** 2 for point in cluster)
        cluster_costs.append(cost)

    return clusters, centroids, cluster_costs

def main():
    data = read_dataset("data/dataset2.csv")
    
    initial_k = 2
    final_k = 10
    k_list = [i for i in range(initial_k, final_k+1)]
    mean_list = []
    iterations = 5


    for k in range(initial_k, final_k+1):
        mean = []
        for _ in range(iterations):
            clusters, centroids, costs = k_means(data, k)      
            mean.append(np.mean(costs))
        mean_list.append(np.mean(mean))

        plot_clusters(clusters, centroids, k, f"visuals\dataset2\k_means++\cluster_graphs/")
    plot_k_mean(mean_list, k_list, "visuals\dataset2\k_means++\line_graph/")


if __name__ == '__main__':
    main()
