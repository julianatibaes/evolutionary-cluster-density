import random
import numpy
from scipy.spatial import distance
from operator import add

#Functions for performing the density clustering algorithm
#class density():

#Functions for performing the k-means clustering algorithm
class k_means():
        
    #Function to cluster the data using Euclidean distance from the cluster centers.
    @staticmethod
    def cluster_data(centers, data, c, feat_num):
        data_length = len(data)
        centers_length = len(centers)
        #Iterating over each feature vector
        for i in xrange(0, data_length):
            #Iterating over each cluster center
            for j in xrange(0, centers_length):
                #Calculating euclidean distance of the vector from the cluster center.
                curr_dist = distance.euclidean(data[i][0:feat_num:1], centers[j][0:feat_num:1])
                if(j == 0):
                    dist = curr_dist
                    cluster = 0
                if(curr_dist < dist):
                    dist = curr_dist
                    cluster = j
            #Appending the cluster index to the feature vector if its the first occurence of this method call.
            if(c):
                data[i].append(cluster)
            #Replacing the old cluster index with the new one if it is not the first occurence of this method call.
            else:
                data[i][-1] = cluster
        return data

    #Function to caluclate the mean centroid of the clusters.
    @staticmethod
    def calculate_centroids(data, k, label_pos, feat_num):
        data_length = len(data)
        new_centers = []
        #Iterating over the clusters.
        for i in xrange(0, k):
            count = 0
            total = []
            for j in xrange(0, feat_num):
                total.append(0)
            for j in xrange(0, data_length):
                #If the data point has the same cluster number as the currently iterating one, add it to the total.
                if(data[j][label_pos] == i):
                    count += 1
                    total = map(add, data[j][0:feat_num:1], total)
            #Calculating the mean center.
            for l in xrange(0, feat_num):
                if (count == 0):
                    return True
                total[l] /= count   
            new_centers.append(total)
        return new_centers

    #Function to calculate the nearest data point to the mean cluster center (medoid)
    @staticmethod
    def calculate_medoids(data, centers, k):
        medoids = []
        data_length = len(data)
        #Iterating over the clusters.
        for i in xrange(0, k):
            #A variable to hold the index of the closest point to the mean center.
            closest = 0
            firstFlag = True
            for j in xrange(0, data_length):
                if(firstFlag):
                    closest = j
                    #Caclulating the distance of the points to the center.
                    curr_dist = distance.euclidean(data[j][0:4:1], centers[i][0:4:1])
                    firstFlag = False
                else:
                    #Checking if the distance is smaller than the current closest.
                    if(distance.euclidean(data[j][0:4:1], centers[i][0:4:1]) < curr_dist):
                        closest = j
                        curr_dist = distance.euclidean(data[j][0:4:1], centers[i][0:4:1])
            #Returning the closest data points as the centers.
            medoids.append(data[closest][0:4:1])
        return medoids                    

    #Function to count the number of items in a cluster.
    @staticmethod
    def count_cluster_items(data, k, label_pos):
        count = numpy.zeros(k)
        data_length = len(data)
        for i in xrange(0, k):
            for j in xrange(0, data_length):
                if(data[j][label_pos] == i):
                    count[i] += 1
        return count

    #Function to check if the clusters are all non-zero.
    @staticmethod
    def check_count(count, k):
        for i in xrange(0, k):
            if (count[i] == 0):
                return False
        return True

    #Function to compare two lists.
    @staticmethod
    def compare_counts(old_count, new_count, k):
        for i in xrange(0, k):
            if(old_count[i] != new_count[i]):
                return False
            else:
                continue
        return True
    
#Functions for performing the fuzzy c-means algorithm.
class fuzzy_c_means():
    
    #Function to generate a partition matrix.
    @staticmethod
    def get_partition_matrix(data, k):
        data_len = len(data)
        partition_matrix = []
        #Iterating over the number of data-points.
        for i in xrange(0, data_len):
            temp_vect = []
            #For each cluster;
            for j in xrange(0, k):
                #Assigining a random partition value between 0 and 1.
                temp_vect.append(round(random.random(), 2))
            partition_matrix.append(temp_vect)
        return partition_matrix  

    #Function to calculate the mean centroid of the clusters.
    @staticmethod
    def calculate_centroids(data, partition_matrix, k, feat_num, m):
        data_len = len(data)
        centers = []
        #Iterating over each cluster.
        for i in xrange(0, k):
            numerator = []
            denominator = []
            temp_center = []
            #Initializing lists to hold the vectors
            for j in xrange(0, feat_num):
                numerator.append(0)
                denominator.append(0)
                temp_center.append(0)
            #Iterating over all the data-points.
            for j in xrange(0, data_len):
                #Iterating over each feature in the data-points
                for l in xrange(0, feat_num):
                    #Performing the calculations to calculate the numerator and denominator for the FCM algorithm.
                    numerator[l] += (data[j][l] * (partition_matrix[j][i] ** m))
                    denominator[l] += (partition_matrix[j][i] ** m)
            #Using numerator and denominator to calculate the center.
            for j in xrange(0, feat_num):
                temp_center[j] = numerator [j] / denominator[j]
            centers.append(temp_center)
        return centers

    #Function to update the partition matrix.
    @staticmethod
    def update_partition_matrix(data, partition_matrix, k, feat_num, centers, m):
        data_len = len(data)
        #Iterating over every data-point.
        for i in xrange(0, data_len):
            #Iterating over each cluster.
            for j in xrange(0, k):
                #Calculating the new partition and putting in the matrix.
                partition_matrix[i][j] = fuzzy_c_means.calc_uij(data[i][0:feat_num:1], j, centers, m, feat_num)
        return partition_matrix

    #Function to calculate the partition for a data-point.
    @staticmethod
    def calc_uij(data_point, curr_cluster, centers, m, feat_num):
        num_centers = len(centers)
        #Calculating euclidean distance between the data-point and the center of the current cluster.
        #print data_point, centers[curr_cluster][0:feat_num:1]
        numerator = distance.euclidean(data_point, centers[curr_cluster][0:feat_num:1])
        uij = 0
        #Iterating over all the centers;
        for i in xrange(0, num_centers):
            #Calculating the euclidean distance between the data-point and each center.
            denominator = distance.euclidean(data_point, centers[i][0:feat_num:1])
            #Preventing divide by zero
            if (denominator == 0):
                return 0
            uij += (numerator / denominator) ** (2 / (m - 1))
        uij = 1 / uij
        return uij

    #Function to count the number of items assigned to a cluster.
    @staticmethod
    def count_cluster_items(partition_matrix, k):
        count = numpy.zeros(k)
        matrix_len = len(partition_matrix)
        for i in xrange(0, matrix_len):
            index = 0
            best = 0
            for j in xrange(0, k):
                if(partition_matrix[i][j] > best):
                    index = j
                    best = partition_matrix[i][j]
            count[index] += 1
        return count

    #Function to label data for statistical analysis.
    @staticmethod
    def label_data(data, partition_matrix, k):
        labelled_data = []
        matrix_len = len(partition_matrix)
        for i in xrange(0, matrix_len):
            index = 0
            best = 0
            for j in xrange(0, k):
                if(partition_matrix[i][j] > best):
                    index = j
                    best = partition_matrix[i][j]
            data[i].append(index)
            labelled_data.append(data[i])
        return labelled_data

    #Function to check if the clusters are all non-zero.
    @staticmethod
    def check_count(count, k):
        for i in xrange(0, k):
            if (count[i] == 0):
                return True
        return False
    
#Functions for performing spectral clustering
class spectral_clustering():
    
    @staticmethod
    def gen_centers(data, k):
        data_len = len(data)
        rand_set = set()
        centers = []
        for i in xrange(0, k):
            rand = random.randint(0, (data_len - 1))
            while rand in rand_set:
                rand = random.randint(0, (data_len - 1))
            centers.append(data[rand])
        return centers
    
    #Function to measure the distance between all point and generate a distance matrix.
    @staticmethod
    def get_distance_matrix(data):
        data_len = len(data)
        dist_matrix = []
        for i in xrange(0, data_len):
            dist_matrix.append([])
        for i in xrange(0, data_len):
            for j in xrange(0, data_len):
                if (i == j):
                    dist_matrix[i].append(0)
                else:
                    dist_matrix[i].append(round(distance.euclidean(data[i], data[j]), 3))
        return dist_matrix

    #Function to calculate adjacent nodes using their Euclidean distance.
    @staticmethod
    def get_adjacency_matrix(distance_matrix, similarity_threshold):
        data_len = len(distance_matrix)
        adj_matrix = []
        for i in xrange(0, data_len):
            adj_matrix.append([])
        for i in xrange(0, data_len):
            for j in xrange(0, data_len):
                if (distance_matrix[i][j] < similarity_threshold):
                    adj_matrix[i].append(1)
                else:
                    adj_matrix[i].append(0)
        return adj_matrix

    #Function to generate a degree matrix from an adjacency matrix.
    @staticmethod
    def get_degree_matrix_simple(adjacency_matrix):
        data_len = len(adjacency_matrix)
        deg_matrix = []
        for i in xrange(0, data_len):
            deg_matrix.append([])
        for i in xrange(0, data_len):
            for j in xrange(0, data_len):
                if (i == j):
                    degree = 0
                    for k in xrange(0, data_len):
                        if (adjacency_matrix [i][k] == 1):
                            degree += 1
                    deg_matrix[i].append(degree)
                else:
                    deg_matrix[i].append(0)
        return deg_matrix    

    #Function to generate a degree matrix from an adjacency matrix.
    @staticmethod
    def get_degree_matrix_complex(similarity_matrix):
        data_len = len(similarity_matrix)
        deg_matrix = []
        for i in xrange(0, data_len):
            deg_matrix.append([])
        for i in xrange(0, data_len):
            for j in xrange(0, data_len):
                if (i == j):
                    degree = 0
                    for k in xrange(0, data_len):
                        degree += similarity_matrix[i][k]
                    deg_matrix[i].append(round(degree, 3))
                else:
                    deg_matrix[i].append(0)
        return deg_matrix

    #Function to invert a degree matrix.
    @staticmethod
    def invert_degree_matrix(degree_matrix):
        data_len = len(degree_matrix)
        for i in xrange(0, data_len):
            for j in xrange(0, data_len):
                if (i == j):
                    degree_matrix[i][j] = round(1 / degree_matrix[i][j], 4)
                else:
                    continue
        return degree_matrix   

    #Function to generate the unnormalized laplacian matrix by using the degree matrix and adjacency matrix.
    @staticmethod
    def get_unnormalized_laplacian_matrix(degree_matrix, adjacency_matrix):
        data_len = len(degree_matrix)
        lap_matrix = []
        for i in xrange(0, data_len):
            lap_matrix.append([])
        for i in xrange(0, data_len):
            for j in xrange(0, data_len):
                #Diagonal elements are set to the degrees.
                if(i == j):
                    lap_matrix[i].append(degree_matrix[i][j])
                #Non-diagonal elements that are adjacent are set to -1.
                if((i != j) & (adjacency_matrix[i][j] == 1)):
                    lap_matrix[i].append(-1)
                #Non-diagonal, non-adjacent elements are set to 0.
                if((i != j) & (adjacency_matrix[i][j] == 0)):
                    lap_matrix[i].append(0)
        return lap_matrix

    #Function to generate the normalized laplacian matrix by using the degree matrix and inverse degree matrix.
    @staticmethod
    def get_normalized_laplacian_matrix(degree_matrix, inverse_degree_matrix):
        data_len = len(degree_matrix)
        lap_matrix = []
        for i in xrange(0, data_len):
            lap_matrix.append([])
        for i in xrange(0, data_len):
            for j in xrange(0, data_len):
                #Diagonal elements are set to 1 if the corresponding degree is non-zero.
                if ((i == j) & (degree_matrix[i][j] != 0)):
                    lap_matrix[i].append(1)
                #Other elements are set to the negative of the inverse degree.
                else:
                    lap_matrix[i].append(inverse_degree_matrix[i][i] * -1)
        return lap_matrix

    #Function that uses eigenvectors to generate the matrix to be clustered.
    @staticmethod
    def get_cluster_matrix(eigenvectors):
        data_len = len(eigenvectors)
        clust_matrix = []
        for i in xrange(0, data_len):
            clust_matrix.append([])
        for i in xrange(0, data_len):
            for j in xrange(0, data_len):
                #All of the ith elements in each eigenvector are put into one clusterable vector.
                clust_matrix[i].append(eigenvectors[j][i])
        return clust_matrix


