import json
import math
from random import randint
from sklearn.cluster import KMeans
import numpy as np
from sklearn.metrics.pairwise import euclidean_distances


class CustomKMeans(KMeans):
    def __init__(self, n_clusters):
        super().__init__(n_clusters)

    def predict(self, X, sample_weight="deprecated"):
        check_is_fitted(self)

        X = self._check_test_data(X)
        if not (isinstance(sample_weight, str) and sample_weight == "deprecated"):
            warnings.warn(
                (
                    "'sample_weight' was deprecated in version 1.3 and "
                    "will be removed in 1.5."
                ),
                FutureWarning,
            )
            sample_weight = _check_sample_weight(sample_weight, X, dtype=X.dtype)
        else:
            sample_weight = _check_sample_weight(None, X, dtype=X.dtype)

        labels = _labels_inertia_threadpool_limit(
            X,
            sample_weight,
            self.cluster_centers_,
            n_threads=self._n_threads,
            return_inertia=False,
        )

        return labels

    def fit_transform(self, X, y=None, sample_weight=None):
        return self.fit(X, sample_weight=sample_weight)._transform(X)

    def _transform(self, X):
        return euclidean_distances(X, self.cluster_centers_)


def compute_r(data):
    r = 0
    last_time = 0
    last_time_add = 0.01
    while True:
        count = 0
        for d in data:
            count += compute_theta(data, d, r)
        if count / len(data) >= 2.5:
            return r
        if count / len(data) - last_time < 1:
            last_time_add = 10 * last_time_add
        last_time = count / len(data)
        r += last_time_add


def compute_theta(data, d, r):
    count = 0
    for d1 in data:
        if math.dist(d, d1) <= r:
            count += 1
    return count - 1


def compute_p(data, d, r):
    count = 0
    for d1 in data:
        if math.dist(d, (d1[1], d1[0])) <= r or math.dist(d, d1) <= r:
            count += 1
    return count


def compute_reflectivity(l1, l2):
    data = []
    for d1 in l1:
        for d2 in l2:
            data.append((d1, d2))
    reflectivity = 0
    data = [(1, 1000000), (100, 2000000), (200, 3000000), (300, 4000000), (400, 5000000), (500, 6000000)]
    r = compute_r(data)
    print("r", r)
    for d in data:
        ref = compute_theta(data, d, r) / compute_p(data, d, r) if compute_p(data, d, r) > 0 else 0
        # if ref != 1:
        #     ref = 0
        reflectivity += ref
    print(reflectivity)
    print(len(data))
    return 1 - reflectivity / len(data)


def main():
    xx = ["1000.0", "205.0", "66.0", "244.0", "88.5", "80.0", "145.0", "85.0", "3600.0", "1123.7", "460.0", "368.0", "3900.0", "144.5", "2460.0", "9.999999999999999e-05", "2.05e-05", "6.5999999999999995e-06", "2.44e-05", "8.85e-06", "8e-06", "1.45e-05", "8.5e-06", "0.00035999999999999997", "0.00011237", "4.6e-05", "3.68e-05", "0.00039", "1.4449999999999999e-05", "0.00024599999999999996"]
    x = [float(n) for n in xx]
    x = np.array(x)
    reshaped_data = x.reshape(-1, 1)
    print(x, reshaped_data)
    clusters = k_means_clustering(reshaped_data, 3)
    for key in clusters:
        print(clusters[key])
    return
    heights_in_cm = [randint(100, 150) for _ in range(50)]
    heights_in_cm = list(dict.fromkeys(heights_in_cm))
    salam = [d * 1.3 for d in heights_in_cm]
    heights_in_cm.sort()
    salam.sort()
    print("heights_in_cm", heights_in_cm)
    print("salam", salam)
    x1 = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 10000, 20000, 30000, 40000, 50000, 60000, 1000000000, 2000000000, 3000000000, 4000000000, 5000000000, 6000000000, 7000000000, 8000000000, 9000000000, 10000000000]
    d1, d2, d3, d4 = k_means_clustering(x1)
    print(d1)
    print(d2)
    print(d3)
    print(d4)
    reflectivity = compute_reflectivity(heights_in_cm, salam)
    print(reflectivity)
    print(d1)
    print(d2)
    reflectivity = compute_reflectivity(d1, d2)
    print(reflectivity)
    return
    new_columns = {"centimeter": heights_in_cm}
    with open("conversion_rates_v2.json", "r") as f:
        conversion_rates = json.load(f)
        all_units = conversion_rates['centimeter']
        for new_unit in all_units:
            new_columns[new_unit] = [all_units[new_unit] * d for d in heights_in_cm]
    new_columns['meter'].extend(new_columns['yard'])
    new_columns['meter'].sort()
    print(new_columns['meter'])
    return
    for i in range(1, len(list(new_columns.keys()))):
        for j in range(i + 1, len(list(new_columns.keys()))):
            key1 = list(new_columns.keys())[i]
            key2 = list(new_columns.keys())[j]
            reflectivity = compute_reflectivity(new_columns[key1], new_columns[key2])
            print(key1, key2)
            if reflectivity != 0:
                print(reflectivity)


def agglomerative_clustering():
    from sklearn.cluster import AgglomerativeClustering
    import numpy as np

    # Your numerical list
    data = [1, 2, 3, 4, 1000, 2000, 3000, 4000]

    # Reshape the data to meet the input requirements of Agglomerative Clustering
    X = np.array(data).reshape(-1, 1)

    # Perform Agglomerative Clustering with 2 clusters
    n_clusters = 2
    clustering = AgglomerativeClustering(n_clusters=n_clusters).fit(X)

    # Get the cluster labels for each data point
    cluster_labels = clustering.labels_

    # Print the cluster labels
    print("Cluster Labels:", cluster_labels)


def get_tp(l, cluster_0_data):
    tp = 0
    checked_indexes = []
    for d in cluster_0_data:
        for i in range(len(l)):
            if i in checked_indexes:
                continue
            if d == l[i]:
                checked_indexes.append(i)
                tp += 1
                break
    return tp


def compute_precision(l, cluster_0_data, cluster_1_data):
    tp0 = get_tp(l, cluster_0_data)
    tp1 = get_tp(l, cluster_1_data)
    return tp0 / len(cluster_0_data) if tp0 > tp1 else tp1 / len(cluster_1_data)


def compute_recall(l, cluster_0_data, cluster_1_data):
    tp = max(get_tp(l, cluster_0_data), get_tp(l, cluster_1_data))
    return tp / len(l)


def compute_f1(l, cluster_0_data, cluster_1_data):
    precision = compute_precision(l, cluster_0_data, cluster_1_data)
    recall = compute_recall(l, cluster_0_data, cluster_1_data)
    return 2 * precision * recall / (precision + recall)



def normal_k_means_clustering(data, num_clusters=2):
    # Sample data points
    data = np.array(data)

    # Initialize K-means with custom distance function
    kmeans = KMeans(n_clusters=num_clusters)
    reshaped_data = data.reshape(-1, 1)
    kmeans.fit(reshaped_data)

    # Get cluster labels and centroids
    labels = kmeans.labels_
    centroids = kmeans.cluster_centers_

    clusters = {}
    cluster_0 = []
    cluster_1 = []
    for cluster_label, data_point in zip(labels, data.flatten()):
        if cluster_label not in clusters:
            clusters[cluster_label] = []
        clusters[cluster_label].append(data_point)
    return clusters


def k_means_clustering(data, num_clusters=2):
    # Sample data points
    data = np.array(data)

    # Initialize K-means with custom distance function
    kmeans = CustomKMeans(n_clusters=num_clusters)
    pairwise_distances = np.zeros((len(data), len(data)))
    for i in range(len(data)):
        for j in range(len(data)):
            # pairwise_distances[i, j] = abs(data[i] - data[j])
            pairwise_distances[i, j] = custom_distance(data[i], data[j])
    kmeans.fit(pairwise_distances)

    # Get cluster labels and centroids
    labels = kmeans.labels_
    centroids = kmeans.cluster_centers_

    clusters = {}
    for cluster_label, data_point in zip(labels, data.flatten()):
        if cluster_label not in clusters:
            clusters[cluster_label] = []
        clusters[cluster_label].append(data_point)
    return clusters


def average(l):
    return sum(l) / len(l)


def compute_cost_for_elbow_method(column):
    avg = average(column)
    cost = 0
    for row in column:
        cost += abs(avg - row)
    return cost / avg


def find_number_of_clusters(column):
    i = 0
    costs = []
    while True:
        i += 1
        clusters = k_means(column, k=i)
        cost = 0
        for key in clusters:
            cost += compute_cost_for_elbow_method(clusters[key])
        # print(f"cost for {i} cluster is {cost}")
        costs.append(cost)
        if i == 6:
            break
    second_derivative = [costs[i] - 2 * costs[i+1] + costs[i+2] for i in range(len(costs)-2)]
    elbow_index = second_derivative.index(max(second_derivative))
    return elbow_index + 2


def find_number_of_clusters_with_normal_k_means(column):
    i = 0
    costs = []
    while True:
        i += 1
        clusters = normal_k_means_clustering(column, num_clusters=i)
        cost = 0
        for key in clusters:
            cost += compute_cost_for_elbow_method(clusters[key])
        # print(f"cost for {i} cluster is {cost}")
        costs.append(cost)
        if i == 6:
            break
    second_derivative = [costs[i] - 2 * costs[i+1] + costs[i+2] for i in range(len(costs)-2)]
    elbow_index = second_derivative.index(max(second_derivative))
    return elbow_index + 2


def separate_columns_using_gap(l):
    l2 = []
    l.sort()
    print(l)
    last_element = None
    for d in l:
        if last_element:
            l2.append(d - last_element)
        last_element = d
    print(l2)


def custom_distance(d1, d2):
    """ Custom distance function based on relative differences. """
    return abs(d1 - d2) / max(abs(d1), abs(d2)) if d1 != 0 or d2 != 0 else 0


def initialize_centroids(data, k):
    """ Randomly select k data points as initial centroids. """
    indices = np.random.choice(len(data), k, replace=False)
    return data[indices]


def assign_clusters(data, centroids):
    """ Assign each data point to the nearest centroid using the custom distance. """
    distances = np.array([[custom_distance(x, c) for c in centroids] for x in data])
    return np.argmin(distances, axis=1)


def update_centroids(data, assignments, k):
    """ Update centroids as the mean of assigned data points. """
    new_centroids = np.array([data[assignments == i].mean() for i in range(k)])
    return new_centroids


def k_means(data, k, max_iters=2000):
    """ Perform k-means clustering with a custom distance function. """
    data = np.array(data)
    centroids = initialize_centroids(data, k)
    clusters = {}
    for _ in range(max_iters):
        assignments = assign_clusters(data, centroids)
        new_centroids = update_centroids(data, assignments, k)
        if np.all(centroids == new_centroids):
            break
        centroids = new_centroids
        clusters = {}
        for i in range(len(data)):
            if assignments[i] not in clusters:
                clusters[assignments[i]] = []
            clusters[assignments[i]].append(data[i])
    return clusters

if __name__ == '__main__':
    a = [1000, 2000, 3000]
    b = [-10000, -1000, 0]
    print(compute_reflectivity(a, b))
    # f = open("results-units/khoroji_dorost.txt", 'r')
    # f2 = open("results-units/khoroji_dorost_2.txt", 'r')
    # line = f.readline().strip()
    # line2 = f2.readline().strip()
    # while line != '' and line2 != '':
    #     table = json.loads(line)
    #     table2 = json.loads(line2)
    #     t2_values = table2['values']
    #     table["values"].extend(t2_values)
    #     print(table)
    #
    #     line = f.readline().strip()
    #     line2 = f2.readline().strip()
    # main()
