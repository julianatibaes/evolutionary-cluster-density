from scipy.spatial import distance

################################################################################
################################## Dunn Index ##################################
################################################################################

def calc_diameter(max_k, data, lab_pos):
    data_len = len(data)
    cluster_diameters = []
    for i in xrange(0, max_k):
        diameter = 0
        for j in xrange(0, data_len):
            if (data[j][lab_pos] == i):
                for k in xrange(0, data_len):
                    if (data[k][lab_pos] == i):
                        dist = distance.euclidean(data[j], data[k])
                        if (dist > diameter):
                            diameter = dist
        cluster_diameters.append(diameter)
    return cluster_diameters
                

################################################################################
############################### Silhouette Index ###############################
################################################################################

def calc_silhouette(data, max_k, lab_pos, feat_num):
    data_len = len(data)
    overall_silhouette = 0
    for i in xrange(0, data_len):
        cluster = data[i][lab_pos]
        av_dissim_own_cluster = calc_average_dissimilarity(data_len, data[i], cluster, data, lab_pos, feat_num)
        min_av_dissim_other_clusters = 999999999
        for j in xrange(0, max_k):
            if (j == cluster):
                continue
            else:
                temp_dissim = calc_average_dissimilarity(data_len, data[i], j, data, lab_pos, feat_num)
                if (temp_dissim < min_av_dissim_other_clusters):
                    min_av_dissim_other_clusters = temp_dissim
        silhouette_denom = max(av_dissim_own_cluster, min_av_dissim_other_clusters)
        silhouette = (min_av_dissim_other_clusters - av_dissim_own_cluster) / silhouette_denom
        overall_silhouette += silhouette
    overall_silhouette /= data_len
    return overall_silhouette

def calc_average_dissimilarity(data_len, point, cluster, data, lab_pos, feat_num):
    dissim = 0
    counter = 0
    for i in xrange(0, data_len):
        if (data[i][lab_pos] == cluster):
            dissim += distance.euclidean(point[0:feat_num:1], data[i][0:feat_num:1])
            counter += 1
    dissim /= counter
    return dissim
        
################################################################################
############################# Davies-Bouldin Index #############################
################################################################################
        
def calc_within_to_between_cluster_ratio(data, centers, lab_pos, cluster_1, cluster_2, feat_num):
    data_len = len(data)
    counter = 0
    d_c_1 = 0
    d_c_2 = 0

    for i in xrange(0, data_len):
        if (data[i][lab_pos] == cluster_1):
            d_c_1 += distance.euclidean(data[i][0:feat_num:1], centers[cluster_1][0:feat_num:1])
            counter = counter + 1
    d_c_1 /= counter

    counter = 0
    for i in xrange(0, data_len):
        if (data[i][lab_pos] == cluster_2):
            d_c_2 += distance.euclidean(data[i][0:feat_num:1], centers[cluster_2][0:feat_num:1])
            counter = counter + 1
    d_c_2 /= counter


    d_numerator = d_c_1 + d_c_2
    d_denominator = distance.euclidean(centers[cluster_1][0:feat_num:1], centers[cluster_2][0:feat_num:1])
    return d_numerator/ d_denominator

def calc_dbi(data, centers, k, lab_pos, data_point_len):
    dbi = 0
    max_arr = []

    for i in xrange(0, k):
        max_arr.append(0)


    for i in xrange(0, k):
        for j in xrange(0, k):
            if (i != j):
                wbcr = calc_within_to_between_cluster_ratio(data, centers, lab_pos, i, j, data_point_len)
            else:
                continue

            if (wbcr > max_arr[i]):
                max_arr[i] = wbcr
        dbi += max_arr[i]
    dbi *= (float(1) / float(k))

    return dbi