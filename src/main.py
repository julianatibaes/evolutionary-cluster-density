#Functions to read the settings file
import config
#Functions related to importing datasets from files
import datasets
#Functions to perform clustering algorithms
import algorithms
#Functions to perform evolutionary functions.
import genetic_algorithm
import differential_evolution
import particle_swarm_optimization
#Functions for calculating fitness.
import stats
#Functions for linear algebra calculations.
import numpy
#Functions for runtime.
import timeit
import math
#Functions to allow array shuffling.
import random
#Functions to allow graph plotting.
import matplotlib.pyplot as plt

if __name__ == "__main__":
    
    #Starting a timer to calculate run-time.
    start_time = timeit.default_timer()
    
    #Parsing settings.ini
    settings = config.parse_settings_file()   
    config.check_valid_settings(settings)
    labels = config.get_print_info()
    
    #Printing settings to user.
    print "Function:", labels.get("function")
    print "Algorithm:", labels.get("algorithm")
    print "Dataset:", labels.get("dataset")
    print "Number of Data Points:", labels.get("num_data")
    print "Fitness function:", labels.get("fitness_func")
    if (settings.get("debug") == "on"):
        debug_flag = True
    else:
        debug_flag = False

    #Housekeeping variables for iris dataset.
    if (settings.get("dataset") == "iris"):
        label_pos = 4
        feat_num = 4
    #Housekeeping variables for wisconsin dataset.
    if (settings.get("dataset") == "wisconsin"):
        label_pos = 9
        feat_num = 9
    #Housekeeping variables for s1 and spiral datasets.
    if ((settings.get("dataset") == "s1") or (settings.get("dataset") == "spiral") or (settings.get("dataset") == "flame")):
        label_pos = 2
        feat_num = 2
    #Housekeeping variables for dim3 dataset.
    if (settings.get("dataset") == "dim3"):
        label_pos = 3
        feat_num = 3
    
################################################################################
############################## Genetic Algorithm ###############################
################################################################################

    if (settings.get("function") == "geneticalgorithm"):
        invalid_flag = False
        
        #Housekeeping variables for GA
        fitnesses = []
        best = []
        best_fitness = 99999
        best_part = 0
        best_chrom = 0

        if (settings.get("algorithm") == "kmeans"):
            print "\nMaximum number of clusters:", settings.get("max_k")
            print "Population size:", settings.get("pop_size")
            print "Number of generations:", settings.get("max_iter")
            print "Mutation Threshold:", settings.get("mutation_threshold")
            print "-------------------------------"
            #Iterating up to the maximum generation.
            for a in xrange(0, settings.get("max_iter")):
                #Intitializing the fitnesses.
                for i in xrange(0, settings.get("pop_size")):
                    fitnesses.append(0)      

                print "\nGeneration", a + 1

                i = 0
                #Iterating over each chromosome.
                while i < settings.get("pop_size"):
                    #Intitialize the feature vectors.
                    if (debug_flag == True):
                        print "Initializing data vectors...\n"
                    if (settings.get("dataset") == "iris"):
                        all_vects = datasets.iris.get_feat_vects()
                    if (settings.get("dataset") == "wisconsin"):
                        all_vects = datasets.wisconsin.get_feat_vects()
                    if (settings.get("dataset") == "s1"):
                        all_vects = datasets.s1.get_feat_vects()
                    if (settings.get("dataset") == "dim3"):
                        all_vects = datasets.dim3.get_feat_vects()
                    if (settings.get("dataset") == "spiral"):
                        all_vects = datasets.spiral.get_feat_vects()
                    if (settings.get("dataset") == "flame"):
                        all_vects = datasets.flame.get_feat_vects()
                    feat_vects = random.sample(all_vects, settings.get("num_data"))
                    #Initialize the chromosomes if it is the first iteration of the program.
                    if (i == 0) & (a == 0): 
                        if (debug_flag == True): 
                            print "Initializing chromosomes..."
                        chromosomes = genetic_algorithm.init_ga(settings.get("pop_size"), feat_vects, settings.get("max_k"), debug_flag)
                    #If the chromosome is invalid, reinitialize it.
                    if (invalid_flag == True):
                        chromosomes = genetic_algorithm.reinit_chrom(feat_vects, settings.get("max_k"), chromosomes, i, debug_flag)
                        invalid_flag = False
                    if (debug_flag == True):
                        print "\nAnalyzing chromosome", i + 1, "..."
                        print chromosomes[i]
                    #Decode the centers in the current chromosome.
                    centers = genetic_algorithm.get_center(chromosomes[i])
                    #Decoding the k value.
                    k = len(centers)
                    if (debug_flag == True):
                        print "Clustering data..."
                    #If there is less than one cluster, the chromosome is invalid.
                    if (k <= 1):
                        if (debug_flag == True):
                            print "Chromosome", i + 1, "is invalid."
                        invalid_flag = True
                        continue
                    #Clustering the data using the decoded centers.
                    clustered_data = algorithms.k_means.cluster_data(centers, feat_vects, True, feat_num)
                    if (debug_flag == True):
                        print "Counting cluster members..."
                    #Counting the cluster members.
                    count = algorithms.k_means.count_cluster_items(clustered_data, k, label_pos)
                    converge_flag = False
                    counter = 1
                    #This code repeatedly executes until the algorithm converges (i.e cluster member count remains stable between iterations).
                    while not converge_flag:
                        if (debug_flag == True):
                            print "\nIteration", counter, "( Chromosome", i + 1, ")"
                        old_count = count
                        if (debug_flag == True):
                            print "\nCalculating new cluster centers..."
                        #Calculating the new centers.
                        centers = algorithms.k_means.calculate_centroids(clustered_data, k, label_pos, feat_num)
                        #Checking chromosome validity.
                        if (centers == True or k <= 1):
                            if (debug_flag == True):
                                print "Chromosome", i + 1, "is invalid."
                            invalid_flag = True
                            break
                        if (debug_flag == True):    
                            print "Clustering data..."
                        #Clustering the data.
                        clustered_data = algorithms.k_means.cluster_data(centers, feat_vects, False, feat_num)
                        if (debug_flag == True):
                            print "Counting cluster members..."
                        #Counting the cluster members.
                        count = algorithms.k_means.count_cluster_items(clustered_data, k, label_pos)
                        if (debug_flag == True):
                            print count
                        #Check if the algorithm has converged. While loop terminates if it has.
                        converge_flag = algorithms.k_means.compare_counts(old_count, count, k)
                        counter += 1    

                    #If the chromosome is valid and the algorithm has converged, this executes.
                    if not invalid_flag:
                        if (debug_flag == True):
                            print "\nCount has stabilized, clusters have converged to an optima."
                            print  "Evaluating clusters..."
                        #The cluster fitness is calculated and saved.
                        if (settings.get("fitness_func") == "dbi"):
                            dbi = stats.calc_dbi(feat_vects, centers, k, label_pos, label_pos)                            
                            fitnesses[i] = dbi
                            if (debug_flag == True):
                                print "Chromosome's fitness (lower is better):", dbi
                                print "Generation's fitnesses:", fitnesses
                            #If it is the first iteration for the algorithm, set the chromosome as the best.
                            if (a == 0):
                                best = chromosomes[i]
                                best_fitness = dbi
                                best_it = a + 1
                                best_chrom = i + 1
                            #If the DBI is better than the saved best fitness, set the chromosome as the new best.
                            if (dbi < best_fitness):
                                best = chromosomes[i]
                                best_fitness = dbi
                                best_it = a + 1
                                best_chrom = i + 1
                        if (settings.get("fitness_func") == "silhouette"):
                            silhouette = stats.calc_silhouette(clustered_data, k, label_pos, feat_num)
                            fitnesses[i] = silhouette
                            if (debug_flag == True):
                                print "Chromosome's fitness (closer to one is better):", silhouette
                                print "Generation's fitnesses:", fitnesses
                            #If it is the first iteration for the algorithm, set the chromosome as the best.
                            if (a == 0):
                                best = chromosomes[i]
                                best_fitness = silhouette
                                best_it = a + 1
                                best_chrom = i + 1
                            #If the silhouette index is better than the saved best fitness, set the chromosome as the new best.
                            if (silhouette > best_fitness):
                                best = chromosomes[i]
                                best_fitness = silhouette
                                best_it = a + 1
                                best_chrom = i + 1
                        #A successful chromosomal iteration allows the counter to be incremented.
                        i = i + 1
                        if (debug_flag == True):
                            #Printing a line for spacing issues.
                            print
                #If it is not the final generation, this executes. Final generation chromosomes do not need to be crossed or mutated.
                if (a != settings.get("max_iter") - 1):
                    #The chromosomes are ordered by fitness.
                    if (settings.get("fitness_func") == "dbi"):
                        ordered_chromosomes = genetic_algorithm.order_chroms(chromosomes, fitnesses, "minimise", debug_flag)
                    if (settings.get("fitness_func") == "silhouette"):
                        ordered_chromosomes = genetic_algorithm.order_chroms(chromosomes, fitnesses, "maximise", debug_flag)
                    #The chromosomes are crossed and mutated to produce children.
                    children = genetic_algorithm.crossover(ordered_chromosomes, 1, debug_flag)
                    children = genetic_algorithm.mutation(settings.get("mutation_threshold"), children, feat_vects, label_pos, settings.get("max_k"), debug_flag)
                    #Children are saved into the chromosome array to be used as the next generation.
                    chromosomes = children
                    if (debug_flag == True):
                        print "\nAfter Generation", a + 1, ", the best chromosome is:"
                        print best, "\nwith a fitness of " + str(best_fitness) + ". (Chromosome " + str(best_chrom) + ", Generation " + str(best_it) + ".)" 

            #After the algorithm finishes, final results are printed.
            if (debug_flag == False):
                print
            print "With a fitness of " + str(best_fitness) + ", the best chromosome found is:"
            print best
            best_centers = genetic_algorithm.get_center(best)
            best_k = len(best_centers)
            print "\nThis translates to a K-value of: ", best_k
            print "Using the centroids: "
            for i in xrange(0, best_k):
                print best_centers[i]
            
            #The timer is stopped and runtime is calculated.
            stop_time = timeit.default_timer()
            runtime_mins = int(math.trunc(((stop_time - start_time) / 60)))
            runtime_secs = int(round(((stop_time - start_time) % 60), 0))
            print "\nRuntime:", runtime_mins, "minute(s)", runtime_secs, "second(s)"
            
            #The best chromosome is then run through the algorithm and results are plotted.
            if (settings.get("plot") == "on"):
                print "\nPlotting data."
                #Clustering the data
                clustered_data = algorithms.k_means.cluster_data(best_centers, feat_vects, False, feat_num)
                #Counting the cluster members.
                count = algorithms.k_means.count_cluster_items(clustered_data, best_k, label_pos)
                converge_flag = False
                counter = 1
                #This code repeatedly executes until the algorithm converges (i.e cluster member count remains stable between iterations).
                while not converge_flag:
                    old_count = count
                    #Calculating the new centers.
                    centers = algorithms.k_means.calculate_centroids(clustered_data, best_k, label_pos, feat_num)
                    #Clustering the data.
                    clustered_data = algorithms.k_means.cluster_data(centers, feat_vects, False, feat_num)
                    #Counting the cluster members.
                    count = algorithms.k_means.count_cluster_items(clustered_data, best_k, label_pos)
                    #Check if the algorithm has converged. While loop terminates if it has.
                    converge_flag = algorithms.k_means.compare_counts(old_count, count, best_k)
                    counter += 1
                colours = ["#FF0000", "#0000FF", "#00FF00", "#00CCFF", "#FF33CC", "#FFFF00", "#000000", "#CCCCCC", "#522900", "#FFFFCC",  "#99CC00", "#FF9999", "#336699", "#006600", "#CCFFFF", "#CCCC00", "#FFCCFF", "#9900FF", "#006666", "#993366"]
                for i in xrange(0, best_k):
                    for j in xrange(0, len(clustered_data)):
                        if(clustered_data[j][label_pos] == i):
                            plt.scatter(clustered_data[j][0], clustered_data[j][1], s=10, color=colours[i])
                plt.show()
            

        if (settings.get("algorithm") == "fuzzycmeans"):
            invalid_flag = False
            fitnesses = []
            print "\nMaximum number of clusters:", settings.get("max_k")
            print "Population size:", settings.get("pop_size")
            print "Number of generations:", settings.get("max_iter")
            print "Mutation Threshold:", settings.get("mutation_threshold")
            print "Convergence Threshold:", settings.get("convergence_threshold")
            print "Fuzziness:", settings.get("m")
            print "-------------------------------"
            #Iterating up to the maximum generation.
            for a in xrange(0, settings.get("max_iter")):
                #Initializing the fitnesses.
                for i in xrange(0, settings.get("pop_size")):
                    fitnesses.append(0) 

                print "\nGeneration", a + 1
                i = 0
                #Iterating over each chromosome.
                while (i < settings.get("pop_size")):
                    #Initialize the feature vectors.
                    if (debug_flag == True):
                        print "Initializing data vectors...\n"
                    if (settings.get("dataset") == "iris"):
                        all_vects = datasets.iris.get_feat_vects()
                    if (settings.get("dataset") == "wisconsin"):
                        all_vects = datasets.wisconsin.get_feat_vects()
                    if (settings.get("dataset") == "s1"):
                        all_vects = datasets.s1.get_feat_vects()
                    if (settings.get("dataset") == "dim3"):
                        all_vects = datasets.dim3.get_feat_vects()
                    if (settings.get("dataset") == "spiral"):
                        all_vects = datasets.spiral.get_feat_vects()
                    if (settings.get("dataset") == "flame"):
                        all_vects = datasets.flame.get_feat_vects()
                    feat_vects = random.sample(all_vects, settings.get("num_data"))
                    #Initialize the chromosomes if it is the program's first iteration.
                    if (i == 0) & (a == 0):    
                        if (debug_flag == True):       
                            print "Initializing chromosomes..."                        
                        chromosomes = genetic_algorithm.init_ga(settings.get("pop_size"), feat_vects, settings.get("max_k"), debug_flag)
                    if (debug_flag == True):
                        print "\nAnalyzing chromosome", i + 1, "..."
                        print chromosomes[i]
                    #Decode the centers from the chromosome.
                    centers = genetic_algorithm.get_center(chromosomes[i])
                    #Decode the k-value from the chromosome.
                    k = len(centers)
                    #Chromosome is invalid if there are fewer than 2 clusters encoded.
                    if (k < 2):
                        if (debug_flag == True):
                            print "Invalid chromosome detected..."
                        #Invalid chromosome is reinitialized and the loop iteration is retried.
                        chromosomes = genetic_algorithm.reinit_chrom(feat_vects, settings.get("max_k"), chromosomes, i, debug_flag)
                        break
                    if (debug_flag == True):
                        print "Generating partition matrix..."
                    #Randomly generating a partition matrix and calculating its L2 norm.
                    partition_matrix_before = algorithms.fuzzy_c_means.get_partition_matrix(feat_vects, k)
                    dist_before = numpy.linalg.norm(partition_matrix_before)
                    if (debug_flag == True):
                        print "Updating partition matrix..."
                    #Updating the partition matrix using the centers from the chromosome and calculating its L2 norm.
                    partition_matrix_after = algorithms.fuzzy_c_means.update_partition_matrix(feat_vects, partition_matrix_before, k, feat_num, centers, settings.get("m"))
                    dist_after = numpy.linalg.norm(partition_matrix_after)
                    #Calculating the convergence using the L2 norms.
                    convergence = abs(dist_after - dist_before)
                    if (debug_flag == True):
                        print "Convergence (threshold of " + str(settings.get("convergence_threshold")) + "):", convergence
                    counter = 1
                    #Executes until the algorithm converges.
                    while (convergence > settings.get("convergence_threshold")):
                        if (debug_flag == True):
                            print "\nIteration", counter, "( Chromosome", i + 1, ")"
                            print "Updating partition matrix..."
                        partition_matrix_before = partition_matrix_after
                        dist_before = dist_after
                        if (debug_flag == True):
                            print "Calcuating centers..."
                        #Centers are calculated and used to update the partition matrix and L2 norm.
                        centers = algorithms.fuzzy_c_means.calculate_centroids(feat_vects, partition_matrix_before, k, feat_num, settings.get("m"))
                        partition_matrix_after = algorithms.fuzzy_c_means.update_partition_matrix(feat_vects, partition_matrix_before, k, feat_num, centers, settings.get("m"))
                        dist_after = numpy.linalg.norm(partition_matrix_after)
                        #Convergence is calculated and checked against the threshold.
                        convergence = abs(dist_after - dist_before)
                        if (debug_flag == True):
                            print "Convergence (threshold of " + str(settings.get("convergence_threshold")) + "):", convergence
                        counter += 1
                    if (debug_flag == True):    
                        print "\nAlgorithm has converged on Generation " + str(counter) + "."
                        print "Counting cluster items..."
                    #Cluster members are counted after convergence.
                    count = algorithms.fuzzy_c_means.count_cluster_items(partition_matrix_after, k)
                    if (debug_flag == True):
                        print count
                        print "Checking chromosome validity..."
                    #Count is checked for validity. Any zero count is an invalid cluster and results in a reinitialization of the chromosome.
                    invalid_flag = algorithms.fuzzy_c_means.check_count(count, k)
                    if not invalid_flag:
                        #Data is labelled with cluster information.
                        labelled_data = algorithms.fuzzy_c_means.label_data(feat_vects, partition_matrix_after, k)
                        if (debug_flag == True):
                            print "Calculating fitness..."
                        #Cluster fitness is calculated (evaluates how good the clusters are).
                        if (settings.get("fitness_func") == "dbi"):
                            dbi = stats.calc_dbi(labelled_data, centers, k, label_pos, feat_num)
                            fitnesses[i] = dbi
                            if (debug_flag == True):
                                print "Chromosome's fitness (lower is better):", dbi
                                print "Generation's fitnesses:", fitnesses
                            #If it is the first iteration for the algorithm, set the chromosome as the best.
                            if (a == 0):
                                best = chromosomes[i]
                                best_fitness = dbi
                                best_it = a + 1
                                best_chrom = i + 1
                            #If the DBI is better than the saved best fitness, set the chromosome as the new best.
                            if (dbi < best_fitness):
                                best = chromosomes[i]
                                best_fitness = dbi
                                best_it = a + 1
                                best_chrom = i + 1
                        if (settings.get("fitness_func") == "silhouette"):
                            silhouette = stats.calc_silhouette(labelled_data, k, label_pos, feat_num)
                            fitnesses[i] = silhouette
                            if (debug_flag == True):
                                print "Chromosome's fitness (closer to one is better):", silhouette
                                print "Generation's fitnesses:", fitnesses
                            #If it is the first iteration for the algorithm, set the chromosome as the best.
                            if (a == 0):
                                best = chromosomes[i]
                                best_fitness = silhouette
                                best_it = a + 1
                                best_chrom = i + 1
                            #If the silhouette index is better than the saved best fitness, set the chromosome as the new best.
                            if (silhouette > best_fitness):
                                best = chromosomes[i]
                                best_fitness = silhouette
                                best_it = a + 1
                                best_chrom = i + 1                                 
                        i = i + 1
                    else:
                        if (debug_flag == True):
                            print "Invalid chromosome detected..."
                        #Reinitialize invalid chromosome.
                        chromosomes = genetic_algorithm.reinit_chrom(feat_vects, settings.get("max_k"), chromosomes, i, debug_flag)
                        
                #If it is not the final generation, this executes. Final generation chromosomes do not need to be crossed or mutated.
                if (a != settings.get("max_iter") - 1):
                    #The chromosomes are ordered by fitness.
                    if (settings.get("fitness_func") == "dbi"):
                        ordered_chromosomes = genetic_algorithm.order_chroms(chromosomes, fitnesses, "minimise", debug_flag)
                    if (settings.get("fitness_func") == "silhouette"):
                        ordered_chromosomes = genetic_algorithm.order_chroms(chromosomes, fitnesses, "maximise", debug_flag)
                    #The chromosomes are crossed and mutated.
                    children = genetic_algorithm.crossover(ordered_chromosomes, 1, debug_flag)
                    children = genetic_algorithm.mutation(settings.get("mutation_threshold"), children, feat_vects, label_pos, settings.get("max_k"), debug_flag)
                    chromosomes = children
                    if (debug_flag == True):
                        print "\nAfter Generation", a + 1, ", the best chromosome is:"
                        print best, "with a fitness of " + str(best_fitness) + ". (Chromosome " + str(best_chrom) + ", Generation " + str(best_it) + ".)" 

            #After the algorithm has finished, final results are printed.
            print
            print "With a fitness of " + str(best_fitness) + ", the best chromosome found is:"
            print best
            best_centers = genetic_algorithm.get_center(best)
            best_k = len(best_centers)
            print "\nThis translates to a K-value of: ", best_k
            print "Using the centroids: "
            for i in xrange(0, best_k):
                print best_centers[i]
            
            #When the program has finished, stop the timer and display runtime.
            stop_time = timeit.default_timer()
            runtime_mins = int(math.trunc(((stop_time - start_time) / 60)))
            runtime_secs = int(round(((stop_time - start_time) % 60), 0))

            print "\n\nRuntime:", runtime_mins, "minute(s)", runtime_secs, "second(s)"
                
            if (settings.get("plot") == "on"):
                print "\nPlotting data."
                #Randomly generating a partition matrix and calculating its L2 norm.
                partition_matrix_before = algorithms.fuzzy_c_means.get_partition_matrix(feat_vects, best_k)
                dist_before = numpy.linalg.norm(partition_matrix_before)
                #Updating the partition matrix using the centers from the chromosome and calculating its L2 norm.
                partition_matrix_after = algorithms.fuzzy_c_means.update_partition_matrix(feat_vects, partition_matrix_before, best_k, feat_num, best_centers, settings.get("m"))
                dist_after = numpy.linalg.norm(partition_matrix_after)
                #Calculating the convergence using the L2 norms.
                convergence = abs(dist_after - dist_before)
                #Executes until the algorithm converges.
                while (convergence > settings.get("convergence_threshold")):
                    partition_matrix_before = partition_matrix_after
                    dist_before = dist_after
                    #Centers are calculated and used to update the partition matrix and L2 norm.
                    centers = algorithms.fuzzy_c_means.calculate_centroids(feat_vects, partition_matrix_before, best_k, feat_num, settings.get("m"))
                    partition_matrix_after = algorithms.fuzzy_c_means.update_partition_matrix(feat_vects, partition_matrix_before, best_k, feat_num, centers, settings.get("m"))
                    dist_after = numpy.linalg.norm(partition_matrix_after)
                    #Convergence is calculated and checked against the threshold.
                    convergence = abs(dist_after - dist_before)
                #Data is labelled with cluster information.
                labelled_data = algorithms.fuzzy_c_means.label_data(feat_vects, partition_matrix_after, best_k)
                colours = ["#FF0000", "#0000FF", "#00FF00", "#00CCFF", "#FF33CC", "#FFFF00", "#000000", "#CCCCCC", "#522900", "#FFFFCC",  "#99CC00", "#FF9999", "#336699", "#006600", "#CCFFFF", "#CCCC00", "#FFCCFF", "#9900FF", "#006666", "#993366"]
                for i in xrange(0, best_k):
                    for j in xrange(0, len(labelled_data)):
                        if(labelled_data[j][label_pos] == i):
                            plt.scatter(labelled_data[j][0], labelled_data[j][1], s=10, color=colours[i])
                plt.show()          
        
        if (settings.get("algorithm") == "spectralclustering"):
            print "\nMaximum number of clusters:", settings.get("max_k")
            print "Population size:", settings.get("pop_size")
            print "Number of generations:", settings.get("max_iter")
            print "Mutation Threshold:", settings.get("mutation_threshold")
            print "Similarity Threshold:", settings.get("similarity_threshold")
            print "-------------------------------"
            #Iterating up to the maximum generation.
            for a in xrange(0, settings.get("max_iter")):
                #Intitializing the fitnesses.
                for i in xrange(0, settings.get("pop_size")):
                    fitnesses.append(0)      

                print "\nGeneration", a + 1
                i = 0
                #Iterating over each chromosome.
                while i < settings.get("pop_size"):
                    #Intitialize the feature vectors.
                    if (debug_flag == True):
                        print "Initializing data vectors...\n"
                    if (settings.get("dataset") == "iris"):
                        all_vects = datasets.iris.get_feat_vects()
                    if (settings.get("dataset") == "wisconsin"):
                        all_vects = datasets.wisconsin.get_feat_vects()
                    if (settings.get("dataset") == "s1"):
                        all_vects = datasets.s1.get_feat_vects()
                    if (settings.get("dataset") == "dim3"):
                        all_vects = datasets.dim3.get_feat_vects()
                    if (settings.get("dataset") == "spiral"):
                        all_vects = datasets.spiral.get_feat_vects()
                    if (settings.get("dataset") == "flame"):
                        all_vects = datasets.flame.get_feat_vects()
                    if (i == 0) & (a == 0):
                        feat_vects = random.sample(all_vects, settings.get("num_data"))
                        label_pos = settings.get("num_data")
                        feat_num = settings.get("num_data")
                        #Initializing the required matrices for spectral clustering.
                        dist_matrix = algorithms.spectral_clustering.get_distance_matrix(feat_vects)
                        adj_matrix = algorithms.spectral_clustering.get_adjacency_matrix(dist_matrix, settings.get("similarity_threshold"))
                        deg_matrix = algorithms.spectral_clustering.get_degree_matrix_simple(adj_matrix)
                        lap_matrix = algorithms.spectral_clustering.get_unnormalized_laplacian_matrix(deg_matrix, adj_matrix)
                        eigenvalues, eigenvectors = numpy.linalg.eig(lap_matrix)
                        clust_matrix = algorithms.spectral_clustering.get_cluster_matrix(eigenvectors)
                    #Initialize the chromosomes if it is the first iteration of the program.
                    if (i == 0) & (a == 0): 
                        if (debug_flag == True): 
                            print "Initializing chromosomes..."
                        chromosomes = genetic_algorithm.init_ga(settings.get("pop_size"), clust_matrix, settings.get("max_k"), debug_flag)
                    #If the chromosome is invalid, reinitialize it.
                    if (invalid_flag == True):
                        chromosomes = genetic_algorithm.reinit_chrom(clust_matrix, settings.get("max_k"), chromosomes, i, debug_flag)
                        invalid_flag = False
                    if (debug_flag == True):
                        print "\nAnalyzing chromosome", i + 1, "..."
                        print chromosomes[i]
                    #Decode the centers in the current chromosome.
                    centers = genetic_algorithm.get_center(chromosomes[i])
                    #Decoding the k value.
                    k = len(centers)
                    if (debug_flag == True):
                        print "Clustering data..."
                    #If there is less than one cluster, the chromosome is invalid.
                    if (k <= 1):
                        if (debug_flag == True):
                            print "Chromosome", i + 1, "is invalid."
                        invalid_flag = True
                        continue
                    #Clustering the data using the decoded centers.
                    clustered_data = algorithms.k_means.cluster_data(centers, clust_matrix, True, feat_num)
                    if (debug_flag == True):
                        print "Counting cluster members..."
                    #Counting the cluster members.
                    count = algorithms.k_means.count_cluster_items(clustered_data, k, label_pos)
                    converge_flag = False
                    counter = 1
                    #This code repeatedly executes until the algorithm converges (i.e cluster member count remains stable between iterations).
                    while not converge_flag:
                        if (debug_flag == True):
                            print "\nIteration", counter, "( Chromosome", i + 1, ")"
                        old_count = list(count)
                        if (debug_flag == True):
                            print "\nCalculating new cluster centers..."
                        #Calculating the new centers.
                        centers = algorithms.k_means.calculate_centroids(clustered_data, k, label_pos, feat_num)
                        #Checking chromosome validity.
                        if (centers == True or k <= 1):
                            if (debug_flag == True):
                                print "Chromosome", i + 1, "is invalid."
                            invalid_flag = True
                            break
                        if (debug_flag == True):    
                            print "Clustering data..."
                        #Clustering the data.
                        clustered_data = algorithms.k_means.cluster_data(centers, clust_matrix, False, feat_num)
                        if (debug_flag == True):
                            print "Counting cluster members..."
                        #Counting the cluster members.
                        count = algorithms.k_means.count_cluster_items(clustered_data, k, label_pos)
                        if (debug_flag == True):
                            print count
                        #Check if the algorithm has converged. While loop terminates if it has.
                        converge_flag = algorithms.k_means.compare_counts(old_count, count, k)
                        counter += 1    

                    #If the chromosome is valid and the algorithm has converged, this executes.
                    if not invalid_flag:
                        if (debug_flag == True):
                            print "\nCount has stabilized, clusters have converged to an optima."
                            print  "Evaluating clusters..."
                        #The cluster fitness is calculated and saved.
                        if (settings.get("fitness_func") == "dbi"):
                            dbi = stats.calc_dbi(clust_matrix, centers, k, label_pos, label_pos)                            
                            fitnesses[i] = dbi
                            if (debug_flag == True):
                                print "Chromosome's fitness (lower is better):", dbi
                                print "Generation's fitnesses:", fitnesses
                            #If it is the first iteration for the algorithm, set the chromosome as the best.
                            if (a == 0):
                                best = chromosomes[i]
                                best_fitness = dbi
                                best_it = a + 1
                                best_chrom = i + 1
                            #If the DBI is better than the saved best fitness, set the chromosome as the new best.
                            if (dbi < best_fitness):
                                best = chromosomes[i]
                                best_fitness = dbi
                                best_it = a + 1
                                best_chrom = i + 1
                        if (settings.get("fitness_func") == "silhouette"):
                            silhouette = stats.calc_silhouette(clust_matrix, k, label_pos, feat_num)   
                            fitnesses[i] = silhouette
                            if (debug_flag == True):
                                print "Chromosome's fitness (closer to one is better):", silhouette
                                print "Generation's fitnesses:", fitnesses
                            #If it is the first iteration for the algorithm, set the chromosome as the best.
                            if (a == 0):
                                best = chromosomes[i]
                                best_fitness = silhouette
                                best_it = a + 1
                                best_chrom = i + 1
                            #If the silhouette index is better than the saved best fitness, set the chromosome as the new best.
                            if (silhouette > best_fitness):
                                best = chromosomes[i]
                                best_fitness = silhouette
                                best_it = a + 1
                                best_chrom = i + 1
                        #A successful chromosomal iteration allows the counter to be incremented.
                        i = i + 1
                        if (debug_flag == True):
                            #Printing a line for spacing issues.
                            print
                #If it is not the final generation, this executes. Final generation chromosomes do not need to be crossed or mutated.
                if (a != settings.get("max_iter") - 1):
                    #The chromosomes are ordered by fitness.
                    if (settings.get("fitness_func") == "dbi"):
                        ordered_chromosomes = genetic_algorithm.order_chroms(chromosomes, fitnesses, "minimise", debug_flag)
                    if (settings.get("fitness_func") == "silhouette"):
                        ordered_chromosomes = genetic_algorithm.order_chroms(chromosomes, fitnesses, "maximise", debug_flag)
                    #The chromosomes are crossed and mutated to produce children.
                    children = genetic_algorithm.crossover(ordered_chromosomes, 1, debug_flag)
                    children = genetic_algorithm.mutation(settings.get("mutation_threshold"), children, clust_matrix, label_pos, settings.get("max_k"), debug_flag)
                    #Children are saved into the chromosome array to be used as the next generation.
                    chromosomes = children
                    if (debug_flag == True):
                        print "\nAfter Generation", a + 1, ", the best chromosome is:"
                        print best, "with a fitness of " + str(best_fitness) + ". (Chromosome " + str(best_chrom) + ", Generation " + str(best_it) + ".)" 

            #After the algorithm finishes, final results are printed.
            if (debug_flag == False):
                print
            print "With a fitness of " + str(best_fitness) + ", the best chromosome found is:"
            print best
            best_centers = genetic_algorithm.get_center(best)
            best_k = len(best_centers)
            print "\nThis translates to a K-value of: ", best_k
            print "Using the centroids: "
            for i in xrange(0, best_k):
                print
                print best_centers[i]
               
            #When the program has finished, stop the timer and display runtime.
            stop_time = timeit.default_timer()
            runtime_mins = int(math.trunc(((stop_time - start_time) / 60)))
            runtime_secs = int(round(((stop_time - start_time) % 60), 0))

            print "\n\nRuntime:", runtime_mins, "minute(s)", runtime_secs, "second(s)"   
               
            if (settings.get("plot") == "on"):
                print "\nPlotting data."
                #Clustering the data using the decoded centers.
                clustered_data = algorithms.k_means.cluster_data(best_centers, clust_matrix, True, feat_num)
                #Counting the cluster members.
                count = algorithms.k_means.count_cluster_items(clustered_data, best_k, label_pos)
                converge_flag = False
                counter = 1
                #This code repeatedly executes until the algorithm converges (i.e cluster member count remains stable between iterations).
                while not converge_flag:
                    old_count = count
                    #Calculating the new centers.
                    centers = algorithms.k_means.calculate_centroids(clustered_data, best_k, label_pos, feat_num)
                    #Checking chromosome validity.
                    #Clustering the data.
                    clustered_data = algorithms.k_means.cluster_data(centers, clust_matrix, False, feat_num)
                    #Counting the cluster members.
                    count = algorithms.k_means.count_cluster_items(clustered_data, best_k, label_pos)
                    #Check if the algorithm has converged. While loop terminates if it has.
                    converge_flag = algorithms.k_means.compare_counts(old_count, count, best_k)
                    counter += 1
                colours = ["#FF0000", "#0000FF", "#00FF00", "#00CCFF", "#FF33CC", "#FFFF00", "#000000", "#CCCCCC", "#522900", "#FFFFCC",  "#99CC00", "#FF9999", "#336699", "#006600", "#CCFFFF", "#CCCC00", "#FFCCFF", "#9900FF", "#006666", "#993366"]
                for i in xrange(0, best_k):
                    for j in xrange(0, len(clustered_data)):
                        if(clustered_data[j][label_pos] == i):
                            plt.scatter(clustered_data[j][0], clustered_data[j][1], s=10, color=colours[i])
                plt.show()
    
################################################################################
########################### Differential Evolution #############################
################################################################################    
    
    if (settings.get("function") == "differentialevolution"):
        invalid_flag = False
        #Housekeeping variables for DE
        fitnesses = []
        best = []
        best_fitness = 99999
        best_part = 0
        best_chrom = 0

        if (settings.get("algorithm") == "kmeans"):
            print "\nMaximum number of clusters:", settings.get("max_k")
            print "Population size:", settings.get("pop_size")
            print "Number of generations:", settings.get("max_iter")
            print "Mutation Threshold:", settings.get("mutation_threshold")
            print "Activation Threshold:", settings.get("activation_threshold")
            print "-------------------------------"
            #Iterating up to the maximum generation.
            for a in xrange(0, settings.get("max_iter")):
                #Initializing the fitnesses.
                for i in xrange(0, settings.get("pop_size")):
                    fitnesses.append(0) 

                print "\nGeneration", a + 1
                i = 0
                #Iterating over each chromosome.
                while (i < settings.get("pop_size")):
                    invalid_flag = False
                    #Initialize the feature vectors.
                    if (debug_flag == True):
                        print "Initializing data vectors...\n"
                    if (settings.get("dataset") == "iris"):
                        all_vects = datasets.iris.get_feat_vects()
                    if (settings.get("dataset") == "wisconsin"):
                        all_vects = datasets.wisconsin.get_feat_vects()
                    if (settings.get("dataset") == "s1"):
                        all_vects = datasets.s1.get_feat_vects()
                    if (settings.get("dataset") == "dim3"):
                        all_vects = datasets.dim3.get_feat_vects()
                    if (settings.get("dataset") == "spiral"):
                        all_vects = datasets.spiral.get_feat_vects()
                    if (settings.get("dataset") == "flame"):
                        all_vects = datasets.flame.get_feat_vects()
                    feat_vects = random.sample(all_vects, settings.get("num_data"))
                    #Initialize the chromosomes if it is the program's first iteration.
                    if (i == 0) & (a == 0):           
                        if (debug_flag == True):
                            print "Initializing chromosomes..."  
                        #Initializing the required matrices for spectral clustering.
                        chromosomes = differential_evolution.init_de(feat_vects, feat_num, settings.get("max_k"), settings.get("pop_size"), settings.get("activation_threshold"))
                    if (debug_flag == True):
                        print "\nAnalyzing chromosome", i + 1, "..."
                        print chromosomes[i]
                    #Decode the centers from the chromosome.
                    centers = differential_evolution.get_center(chromosomes[i], settings.get("max_k"), settings.get("activation_threshold"), feat_num)
                    #Decode the k-value from the chromosome.
                    k = len(centers)
                    if (debug_flag == True):
                        print "\nClustering data...\n"
                    #Cluster the data according to the centers in the chromosome.
                    clustered_data = algorithms.k_means.cluster_data(centers, feat_vects, True, feat_num)
                    if (debug_flag == True):
                        print "Counting cluster members...\n"
                    #Count the cluster members.
                    new_count = algorithms.k_means.count_cluster_items(clustered_data, k, label_pos)
                    if (debug_flag == True):
                        print new_count
                    converge_flag = False
                    counter = 1
                    #Until the algorithm converges, this loop executes..
                    while not bool(converge_flag):
                        if (debug_flag == True):
                            print "\nIteration", counter, "( Chromosome", i + 1, ")"
                            print "\nCalculating new cluster centers...\n"
                        old_count = new_count   
                        #Calculate the new centers.
                        centers = algorithms.k_means.calculate_centroids(clustered_data, k, label_pos, feat_num)
                        if((centers == True) or (0 in old_count)):
                            #If the function returns true or there is a 0 count, the chromosome is invalid.
                            invalid_flag = True
                            #Reinitialize the chromosome.
                            chromosomes[i] = differential_evolution.reinit_chrom(chromosomes[i], feat_vects, feat_num, settings.get("max_k"), settings.get("activation_threshold"), debug_flag)
                            break
                        if (debug_flag == True):
                            print "Clustering data...\n"
                        #Cluster the data according to the new centers.
                        clustered_data = algorithms.k_means.cluster_data(centers, feat_vects, False, feat_num)
                        if (debug_flag == True):
                            print "Counting cluster members...\n"
                        #Re-count the cluster members.
                        new_count = algorithms.k_means.count_cluster_items(clustered_data, k, label_pos)
                        if (debug_flag == True):
                            print new_count
                        #Check to see if the algorithm has converged.
                        converge_flag = algorithms.k_means.compare_counts(old_count, new_count, k)
                        counter += 1    
                    #This executes for valid chromosomes that converge;
                    if not invalid_flag:
                        if (debug_flag == True):
                            print "\nCount has remained unchanged, clusters have converged to an optima."
                            print  "Evaluating clusters..."
                        #Calculate cluster fitness.
                        if (settings.get("fitness_func") == "dbi"):
                            dbi = stats.calc_dbi(feat_vects, centers, k, label_pos, feat_num)
                            fitnesses[i] = dbi
                            if (debug_flag == True):
                                print "Saving fitness:", dbi
                                print "fitnesses:", fitnesses
                            #If it is the first iteration for the algorithm, set the chromosome as the best.
                            if (a == 0):
                                best = chromosomes[i]
                                best_fitness = dbi
                                best_it = a + 1
                                best_chrom = i + 1
                            #If the DBI is better than the saved best fitness, set the chromosome as the new best.
                            if (dbi < best_fitness):
                                best = chromosomes[i]
                                best_fitness = dbi
                                best_it = a + 1
                                best_chrom = i + 1
                        if (settings.get("fitness_func") == "silhouette"):
                            silhouette = stats.calc_silhouette(feat_vects, k, label_pos, feat_num)
                            fitnesses[i] = silhouette
                            if (debug_flag == True):
                                print "Saving fitness:", silhouette
                                print "fitnesses:", fitnesses
                            #If it is the first iteration for the algorithm, set the chromosome as the best.
                            if (a == 0):
                                best = chromosomes[i]
                                best_fitness = silhouette
                                best_it = a + 1
                                best_chrom = i + 1
                            #If the silhouette index is better than the saved best fitness, set the chromosome as the new best.
                            if (silhouette > best_fitness):
                                best = chromosomes[i]
                                best_fitness = silhouette
                                best_it = a + 1
                                best_chrom = i + 1
                        i = i + 1
                #If it is not the final generation, this executes. Final generation chromosomes do not need to be crossed or mutated.
                if(a < (settings.get("max_iter") - 1)):
                    #Order the chromosomes by fitness, perform crossover and mutation.
                    if (settings.get("fitness_func") == "dbi"):
                        ordered_chromosomes = differential_evolution.order_chroms(chromosomes, fitnesses, "minimise", debug_flag)
                    if (settings.get("fitness_func") == "silhouette"):
                        ordered_chromosomes = differential_evolution.order_chroms(chromosomes, fitnesses, "maximise", debug_flag)
                    children = differential_evolution.crossover(ordered_chromosomes, 1, debug_flag)
                    children = differential_evolution.mutation(settings.get("activation_threshold"), children, settings.get("max_k"), settings.get("mutation_threshold"), debug_flag)
                    chromosomes = children
                    if (debug_flag == True):
                        print "\nAfter Generation", a + 1, ", the best chromosome is:"
                        print best, "with a fitness of " + str(best_fitness) + ". (Chromosome " + str(best_chrom) + ", Generation " + str(best_it) + ".)" 

            #If the algorithm finishes, final results are printed.
            if (debug_flag == False):
                print
            print "With a fitness of " + str(best_fitness) + ", The best chromosome found is:"
            print best
            best_centers = differential_evolution.get_center(best, settings.get("max_k"), settings.get("activation_threshold"), feat_num)
            best_k = len(best_centers)
            print "\nThis translates to a K-value of: ", best_k
            print "Using the centroids: "
            for i in xrange(0, best_k):
                print best_centers[i]
            #Runtime is calculated.
            stop_time = timeit.default_timer()
            runtime_mins = int(math.trunc(((stop_time - start_time) / 60)))
            runtime_secs = int(round(((stop_time - start_time) % 60), 0))
            print "\n\nRuntime:", runtime_mins, "minute(s)", runtime_secs, "second(s)"
            
            #The best chromosome is then run through the algorithm and results are plotted.
            if (settings.get("plot") == "on"):
                print "\nPlotting data."
                #Clustering the data
                clustered_data = algorithms.k_means.cluster_data(best_centers, feat_vects, False, feat_num)
                #Counting the cluster members.
                count = algorithms.k_means.count_cluster_items(clustered_data, best_k, label_pos)
                converge_flag = False
                counter = 1
                #This code repeatedly executes until the algorithm converges (i.e cluster member count remains stable between iterations).
                while not converge_flag:
                    old_count = count
                    #Calculating the new centers.
                    centers = algorithms.k_means.calculate_centroids(clustered_data, best_k, label_pos, feat_num)
                    #Clustering the data.
                    clustered_data = algorithms.k_means.cluster_data(centers, feat_vects, False, feat_num)
                    #Counting the cluster members.
                    count = algorithms.k_means.count_cluster_items(clustered_data, best_k, label_pos)
                    #Check if the algorithm has converged. While loop terminates if it has.
                    converge_flag = algorithms.k_means.compare_counts(old_count, count, best_k)
                    counter += 1
                colours = ["#FF0000", "#0000FF", "#00FF00", "#00CCFF", "#FF33CC", "#FFFF00", "#000000", "#CCCCCC", "#522900", "#FFFFCC",  "#99CC00", "#FF9999", "#336699", "#006600", "#CCFFFF", "#CCCC00", "#FFCCFF", "#9900FF", "#006666", "#993366"]
                for i in xrange(0, best_k):
                    for j in xrange(0, len(clustered_data)):
                        if(clustered_data[j][label_pos] == i):
                            plt.scatter(clustered_data[j][0], clustered_data[j][1], s=10, color=colours[i])
                plt.show()

        if (settings.get("algorithm") == "fuzzycmeans"):
            invalid_flag = False
            fitnesses = []
            print "\nMaximum number of clusters:", settings.get("max_k")
            print "Population size:", settings.get("pop_size")
            print "Number of generations:", settings.get("max_iter")
            print "Mutation Threshold:", settings.get("mutation_threshold")
            print "Activation Threshold:", settings.get("activation_threshold")
            print "Convergence Threshold:", settings.get("convergence_threshold")
            print "Fuzziness:", settings.get("m")
            print "-------------------------------"
            #Iterating up to the maximum generation.
            for a in xrange(0, settings.get("max_iter")):
                #Initializing the fitnesses.
                for i in xrange(0, settings.get("pop_size")):
                    fitnesses.append(0) 

                print "\nGeneration", a + 1
                i = 0
                #Iterating over each chromosome.
                while (i < settings.get("pop_size")):
                    #Initialize the feature vectors.
                    if (debug_flag == True):
                        print "Initializing data vectors...\n"
                    if (settings.get("dataset") == "iris"):
                        all_vects = datasets.iris.get_feat_vects()
                    if (settings.get("dataset") == "wisconsin"):
                        all_vects = datasets.wisconsin.get_feat_vects()
                    if (settings.get("dataset") == "s1"):
                        all_vects = datasets.s1.get_feat_vects()
                    if (settings.get("dataset") == "dim3"):
                        all_vects = datasets.dim3.get_feat_vects()
                    if (settings.get("dataset") == "spiral"):
                        all_vects = datasets.spiral.get_feat_vects()
                    if (settings.get("dataset") == "flame"):
                        all_vects = datasets.flame.get_feat_vects()
                    feat_vects = random.sample(all_vects, settings.get("num_data"))
                    if (i == 0) & (a == 0):         
                        if (debug_flag == True):  
                            print "Initializing chromosomes..."
                        #Initialize the chromosomes if it is the program's first iteration.
                        chromosomes = differential_evolution.init_de(feat_vects, feat_num, settings.get("max_k"), settings.get("pop_size"), settings.get("activation_threshold"))
                    if (debug_flag == True):
                        print "\nAnalyzing chromosome", i + 1, "..."
                        print chromosomes[i]
                    #Decode the centers from the chromosome.
                    centers = differential_evolution.get_center(chromosomes[i], settings.get("max_k"), settings.get("activation_threshold"), feat_num)
                    #Decode the k-value from the chromosome.
                    k = len(centers)
                    if (debug_flag == True):
                        print "Generating partition matrix..."
                    #Randomly generating a partition matrix and calculating its L2 norm.
                    partition_matrix_before = algorithms.fuzzy_c_means.get_partition_matrix(feat_vects, k)
                    dist_before = numpy.linalg.norm(partition_matrix_before)
                    if (debug_flag == True):
                        print "Updating partition matrix..."
                    #Using the chromosomal centers to update the partition matrix and calculating its L2 norm.
                    partition_matrix_after = algorithms.fuzzy_c_means.update_partition_matrix(feat_vects, partition_matrix_before, k, feat_num, centers, settings.get("m"))
                    dist_after = numpy.linalg.norm(partition_matrix_after)
                    #Calculating convergence based on the L2 norms.
                    convergence = abs(dist_after - dist_before)
                    if (debug_flag == True):
                        print "Convergence (threshold of " + str(settings.get("convergence_threshold")) + "):", convergence
                    counter = 1
                    #This executes until the convergence becomes smaller than the given threshold.
                    while (convergence > settings.get("convergence_threshold")):
                        if (debug_flag == True):
                            print "\nIteration", counter, "( Chromosome", i + 1, ")"
                            print "Updating partition matrix..."
                        partition_matrix_before = partition_matrix_after
                        dist_before = dist_after
                        if (debug_flag == True):
                            print "Calcuating centers..."
                        #Calculating new centers.
                        centers = algorithms.fuzzy_c_means.calculate_centroids(feat_vects, partition_matrix_before, k, feat_num, settings.get("m"))
                        #Using them to update the partition matrix and calculate its L2 norm.
                        partition_matrix_after = algorithms.fuzzy_c_means.update_partition_matrix(feat_vects, partition_matrix_before, k, feat_num, centers, settings.get("m"))
                        dist_after = numpy.linalg.norm(partition_matrix_after)
                        #Checking convergence.
                        convergence = abs(dist_after - dist_before)
                        if (debug_flag == True):
                            print "Convergence (threshold of " + str(settings.get("convergence_threshold")) + "):", convergence
                        counter += 1                    
                    if (debug_flag == True):
                        print "\nAlgorithm has converged on Generation " + str(counter) + "."
                        print "Counting cluster items..."
                    #On convergence, cluster members are counted.
                    count = algorithms.fuzzy_c_means.count_cluster_items(partition_matrix_after, k)
                    if (debug_flag == True):
                        print count
                        print "Checking chromosome validity..."
                    #Count is checked for zeros. A zero value means the chromosome is invalid.
                    invalid_flag = algorithms.fuzzy_c_means.check_count(count, k)
                    #If the chromosome is valid, the algorithm continues.
                    if (invalid_flag == False):
                        #The data is labelled using the clusters.
                        labelled_data = algorithms.fuzzy_c_means.label_data(feat_vects, partition_matrix_after, k)
                        if (debug_flag == True):
                            print "Calculating fitness..."
                        #The cluster fitness is calculated (evaluates how good the clusters are).
                        if (settings.get("fitness_func") == "dbi"):
                            dbi = stats.calc_dbi(labelled_data, centers, k, label_pos, feat_num)
                            fitnesses[i] = dbi
                            if (debug_flag == True):
                                print "saving fitness:", dbi
                                print "fitnesses:", fitnesses
                            #If it is the first iteration, the chromosome is set as the best.
                            if (a == 0):
                                best = chromosomes[i]
                                best_fitness = dbi
                                best_it = a + 1
                                best_chrom = i + 1
                            #If the DBI is better than the saved best fitness, set the chromosome as the new best.
                            if (dbi < best_fitness):
                                best = chromosomes[i]
                                best_fitness = dbi
                                best_it = a + 1
                                best_chrom = i + 1
                        if (settings.get("fitness_func") == "silhouette"):
                            silhouette = stats.calc_silhouette(labelled_data, k, label_pos, feat_num)
                            fitnesses[i] = silhouette
                            if (debug_flag == True):
                                print "saving fitness:", silhouette
                                print "fitnesses:", fitnesses
                            #If it is the first iteration, the chromosome is set as the best.
                            if (a == 0):
                                best = chromosomes[i]
                                best_fitness = silhouette
                                best_it = a + 1
                                best_chrom = i + 1
                            #If the DBI is better than the saved best fitness, set the chromosome as the new best.
                            if (silhouette > best_fitness):
                                best = chromosomes[i]
                                best_fitness = silhouette
                                best_it = a + 1
                                best_chrom = i + 1                            
                        i = i + 1
                    else:
                        #Invalid chromosomes are reinitialized and the iteration begins again.
                        chromosomes[i] = differential_evolution.reinit_chrom(chromosomes[i], feat_vects, feat_num, settings.get("max_k"), settings.get("activation_threshold"), debug_flag)


                #If it is not the final generation, this executes. Final generation chromosomes do not need to be crossed or mutated.
                if(a < (settings.get("max_iter") - 1)):
                    #Order the chromosomes by fitness, perform crossover and mutation.
                    if (settings.get("fitness_func") == "dbi"):
                        ordered_chromosomes = differential_evolution.order_chroms(chromosomes, fitnesses, "minimise", debug_flag)
                    if (settings.get("fitness_func") == "silhouette"):
                        ordered_chromosomes = differential_evolution.order_chroms(chromosomes, fitnesses, "maximise", debug_flag)
                    children = differential_evolution.crossover(ordered_chromosomes, 1, debug_flag)
                    children = differential_evolution.mutation(settings.get("activation_threshold"), children, settings.get("max_k"), settings.get("mutation_threshold"), debug_flag)
                    chromosomes = children
                    if (debug_flag == True):
                        print "\nAfter Generation", a + 1, ", the best chromosome is:"
                        print best, "with a fitness of " + str(best_fitness) + ". (Chromosome " + str(best_chrom) + ", Generation " + str(best_it) + ".)" 
            #If the algorithm finishes, final results are printed.
            if (debug_flag == False):
                print
            print "With a fitness of " + str(best_fitness) + ", The best chromosome found is:"
            print best
            best_centers = differential_evolution.get_center(best, settings.get("max_k"), settings.get("activation_threshold"), feat_num)
            best_k = len(best_centers)
            print "\nThis translates to a K-value of: ", best_k
            print "Using the centroids: "
            for i in xrange(0, best_k):
                print best_centers[i]
            #Runtime is calculated.
            stop_time = timeit.default_timer()
            runtime_mins = int(math.trunc(((stop_time - start_time) / 60)))
            runtime_secs = int(round(((stop_time - start_time) % 60), 0))
            print "\n\nRuntime:", runtime_mins, "minute(s)", runtime_secs, "second(s)"
            
            if (settings.get("plot") == "on"):
                print "\nPlotting data."
                #Randomly generating a partition matrix and calculating its L2 norm.
                partition_matrix_before = algorithms.fuzzy_c_means.get_partition_matrix(feat_vects, best_k)
                dist_before = numpy.linalg.norm(partition_matrix_before)
                #Updating the partition matrix using the centers from the chromosome and calculating its L2 norm.
                partition_matrix_after = algorithms.fuzzy_c_means.update_partition_matrix(feat_vects, partition_matrix_before, best_k, feat_num, best_centers, settings.get("m"))
                dist_after = numpy.linalg.norm(partition_matrix_after)
                #Calculating the convergence using the L2 norms.
                convergence = abs(dist_after - dist_before)
                #Executes until the algorithm converges.
                while (convergence > settings.get("convergence_threshold")):
                    partition_matrix_before = partition_matrix_after
                    dist_before = dist_after
                    #Centers are calculated and used to update the partition matrix and L2 norm.
                    centers = algorithms.fuzzy_c_means.calculate_centroids(feat_vects, partition_matrix_before, best_k, feat_num, settings.get("m"))
                    partition_matrix_after = algorithms.fuzzy_c_means.update_partition_matrix(feat_vects, partition_matrix_before, best_k, feat_num, centers, settings.get("m"))
                    dist_after = numpy.linalg.norm(partition_matrix_after)
                    #Convergence is calculated and checked against the threshold.
                    convergence = abs(dist_after - dist_before)
                #Data is labelled with cluster information.
                labelled_data = algorithms.fuzzy_c_means.label_data(feat_vects, partition_matrix_after, best_k)
                colours = ["#FF0000", "#0000FF", "#00FF00", "#00CCFF", "#FF33CC", "#FFFF00", "#000000", "#CCCCCC", "#522900", "#FFFFCC",  "#99CC00", "#FF9999", "#336699", "#006600", "#CCFFFF", "#CCCC00", "#FFCCFF", "#9900FF", "#006666", "#993366"]
                for i in xrange(0, best_k):
                    for j in xrange(0, len(labelled_data)):
                        if(labelled_data[j][label_pos] == i):
                            plt.scatter(labelled_data[j][0], labelled_data[j][1], s=10, color=colours[i])
                plt.show()
            
        if (settings.get("algorithm") == "spectralclustering"):
            print "\nMaximum number of clusters:", settings.get("max_k")
            print "Population size:", settings.get("pop_size")
            print "Number of generations:", settings.get("max_iter")
            print "Mutation Threshold:", settings.get("mutation_threshold")
            print "Activation Threshold:", settings.get("activation_threshold")
            print "Similarity Threshold:", settings.get("similarity_threshold")
            print "-------------------------------"
            #Iterating up to the maximum generation.
            for a in xrange(0, settings.get("max_iter")):
                #Initializing the fitnesses.
                for i in xrange(0, settings.get("pop_size")):
                    fitnesses.append(0) 

                print "\nGeneration", a + 1
                i = 0
                #Iterating over each chromosome.
                while (i < settings.get("pop_size")):
                    invalid_flag = False
                    #Initialize the feature vectors.
                    if (debug_flag == True):
                        print "Initializing data vectors...\n"
                    if (settings.get("dataset") == "iris"):
                        all_vects = datasets.iris.get_feat_vects()
                    if (settings.get("dataset") == "wisconsin"):
                        all_vects = datasets.wisconsin.get_feat_vects()
                    if (settings.get("dataset") == "s1"):
                        all_vects = datasets.s1.get_feat_vects()
                    if (settings.get("dataset") == "dim3"):
                        all_vects = datasets.dim3.get_feat_vects()
                    if (settings.get("dataset") == "spiral"):
                        all_vects = datasets.spiral.get_feat_vects()
                    if (settings.get("dataset") == "flame"):
                        all_vects = datasets.flame.get_feat_vects()
                    if (i == 0) & (a == 0):
                        feat_vects = random.sample(all_vects, settings.get("num_data"))
                        label_pos = settings.get("num_data")
                        feat_num = settings.get("num_data")
                        #Initializing the required matrices for spectral clustering.
                        dist_matrix = algorithms.spectral_clustering.get_distance_matrix(feat_vects)
                        adj_matrix = algorithms.spectral_clustering.get_adjacency_matrix(dist_matrix, settings.get("similarity_threshold"))
                        deg_matrix = algorithms.spectral_clustering.get_degree_matrix_simple(adj_matrix)
                        lap_matrix = algorithms.spectral_clustering.get_unnormalized_laplacian_matrix(deg_matrix, adj_matrix)
                        eigenvalues, eigenvectors = numpy.linalg.eig(lap_matrix)
                        clust_matrix = algorithms.spectral_clustering.get_cluster_matrix(eigenvectors)
                    #Initialize the chromosomes if it is the program's first iteration.
                    if (i == 0) & (a == 0):           
                        if (debug_flag == True):
                            print "Initializing chromosomes..."  
                        chromosomes = differential_evolution.init_de(clust_matrix, feat_num, settings.get("max_k"), settings.get("pop_size"), settings.get("activation_threshold"))
                    if (debug_flag == True):
                        print "\nAnalyzing chromosome", i + 1, "..."
                        print chromosomes[i]
                    #Decode the centers from the chromosome.
                    centers = differential_evolution.get_center(chromosomes[i], settings.get("max_k"), settings.get("activation_threshold"), feat_num)
                    #Decode the k-value from the chromosome.
                    k = len(centers)
                    if (debug_flag == True):
                        print "\nClustering data...\n"
                    #Cluster the data according to the centers in the chromosome.
                    clustered_data = algorithms.k_means.cluster_data(centers, clust_matrix, True, feat_num)
                    if (debug_flag == True):
                        print "Counting cluster members...\n"
                    #Count the cluster members.
                    new_count = algorithms.k_means.count_cluster_items(clustered_data, k, label_pos)
                    if (debug_flag == True):
                        print new_count
                    converge_flag = False
                    counter = 1
                    #Until the algorithm converges, this loop executes..
                    while not bool(converge_flag):
                        if (debug_flag == True):
                            print "\nIteration", counter, "( Chromosome", i + 1, ")"
                            print "\nCalculating new cluster centers...\n"
                        old_count = new_count[:]
                        #Calculate the new centers.
                        centers = algorithms.k_means.calculate_centroids(clustered_data, k, label_pos, feat_num)
                        if((centers == True) or (0 in old_count)):
                            #If the function returns true or there is a 0 count, the chromosome is invalid.
                            invalid_flag = True
                            #Reinitialize the chromosome.
                            chromosomes[i] = differential_evolution.reinit_chrom(chromosomes[i], clust_matrix, feat_num, settings.get("max_k"), settings.get("activation_threshold"), debug_flag)
                            break
                        if (debug_flag == True):
                            print "Clustering data...\n"
                        #Cluster the data according to the new centers.
                        clustered_data = algorithms.k_means.cluster_data(centers, clust_matrix, False, feat_num)
                        if (debug_flag == True):
                            print "Counting cluster members...\n"
                        #Re-count the cluster members.
                        new_count = algorithms.k_means.count_cluster_items(clustered_data, k, label_pos)
                        if (debug_flag == True):
                            print new_count
                        #Check to see if the algorithm has converged.
                        converge_flag = algorithms.k_means.compare_counts(old_count, new_count, k)
                        counter += 1    
                    #This executes for valid chromosomes that converge;
                    if not invalid_flag:
                        if (debug_flag == True):
                            print "\nCount has remained unchanged, clusters have converged to an optima."
                            print  "Evaluating clusters..."
                        #Calculate the cluster fitness.
                        if (settings.get("fitness_func") == "dbi"):
                            dbi = stats.calc_dbi(clust_matrix, centers, k, label_pos, feat_num)
                            fitnesses[i] = dbi
                            if (debug_flag == True):
                                print "Saving fitness:", dbi
                                print "fitnesses:", fitnesses
                            #If it is the first iteration for the algorithm, set the chromosome as the best.
                            if (a == 0):
                                best = chromosomes[i]
                                best_fitness = dbi
                                best_it = a + 1
                                best_chrom = i + 1
                            #If the DBI is better than the saved best fitness, set the chromosome as the new best.
                            if (dbi < best_fitness):
                                best = chromosomes[i]
                                best_fitness = dbi
                                best_it = a + 1
                                best_chrom = i + 1
                        if (settings.get("fitness_func") == "silhouette"):
                            silhouette = stats.calc_silhouette(clust_matrix, k, label_pos, feat_num)
                            fitnesses[i] = silhouette
                            if (debug_flag == True):
                                print "Saving fitness:", silhouette
                                print "fitnesses:", fitnesses 
                            #If it is the first iteration for the algorithm, set the chromosome as the best.
                            if (a == 0):
                                best = chromosomes[i]
                                best_fitness = silhouette
                                best_it = a + 1
                                best_chrom = i + 1
                            #If the DBI is better than the saved best fitness, set the chromosome as the new best.
                            if (silhouette > best_fitness):
                                best = chromosomes[i]
                                best_fitness = silhouette
                                best_it = a + 1
                                best_chrom = i + 1
                        i = i + 1
                #If it is not the final generation, this executes. Final generation chromosomes do not need to be crossed or mutated.
                if(a < (settings.get("max_iter") - 1)):
                    #Order the chromosomes by fitness, perform crossover and mutation.
                    if (settings.get("fitness_func") == "dbi"):
                        ordered_chromosomes = differential_evolution.order_chroms(chromosomes, fitnesses, "minimise", debug_flag)
                    if (settings.get("fitness_func") == "silhouette"):
                        ordered_chromosomes = differential_evolution.order_chroms(chromosomes, fitnesses, "maximise", debug_flag)
                    children = differential_evolution.crossover(ordered_chromosomes, 1, debug_flag)
                    children = differential_evolution.mutation(settings.get("activation_threshold"), children, settings.get("max_k"), settings.get("mutation_threshold"), debug_flag)
                    chromosomes = children
                    if (debug_flag == True):
                        print "\nAfter Generation", a + 1, ", the best chromosome is:"
                        print best, "with a fitness of " + str(best_fitness) + ". (Chromosome " + str(best_chrom) + ", Generation " + str(best_it) + ".)" 

            #If the algorithm finishes, final results are printed.
            if (debug_flag == False):
                print
            print "With a fitness of " + str(best_fitness) + ", The best chromosome found is:"
            print best
            best_centers = differential_evolution.get_center(best, settings.get("max_k"), settings.get("activation_threshold"), feat_num)
            best_k = len(best_centers)
            print "\nThis translates to a K-value of: ", best_k
            print "Using the centroids: "
            for i in xrange(0, best_k):
                print
                print best_centers[i]
            #Runtime is calculated.
            stop_time = timeit.default_timer()
            runtime_mins = int(math.trunc(((stop_time - start_time) / 60)))
            runtime_secs = int(round(((stop_time - start_time) % 60), 0))
            print "\n\nRuntime:", runtime_mins, "minute(s)", runtime_secs, "second(s)"
            
            if (settings.get("plot") == "on"):
                print "\nPlotting data."
                #Clustering the data using the decoded centers.
                clustered_data = algorithms.k_means.cluster_data(best_centers, clust_matrix, True, feat_num)
                #Counting the cluster members.
                count = algorithms.k_means.count_cluster_items(clustered_data, best_k, label_pos)
                converge_flag = False
                counter = 1
                #This code repeatedly executes until the algorithm converges (i.e cluster member count remains stable between iterations).
                while not converge_flag:
                    old_count = count
                    #Calculating the new centers.
                    centers = algorithms.k_means.calculate_centroids(clustered_data, best_k, label_pos, feat_num)
                    #Checking chromosome validity.
                    #Clustering the data.
                    clustered_data = algorithms.k_means.cluster_data(centers, clust_matrix, False, feat_num)
                    #Counting the cluster members.
                    count = algorithms.k_means.count_cluster_items(clustered_data, best_k, label_pos)
                    #Check if the algorithm has converged. While loop terminates if it has.
                    converge_flag = algorithms.k_means.compare_counts(old_count, count, best_k)
                    counter += 1
                colours = ["#FF0000", "#0000FF", "#00FF00", "#00CCFF", "#FF33CC", "#FFFF00", "#000000", "#CCCCCC", "#522900", "#FFFFCC",  "#99CC00", "#FF9999", "#336699", "#006600", "#CCFFFF", "#CCCC00", "#FFCCFF", "#9900FF", "#006666", "#993366"]
                for i in xrange(0, best_k):
                    for j in xrange(0, len(clustered_data)):
                        if(clustered_data[j][label_pos] == i):
                            plt.scatter(clustered_data[j][0], clustered_data[j][1], s=10, color=colours[i])
                plt.show()
 
################################################################################
######################## Particle Swarm Optimization ###########################
################################################################################ 
 
    if (settings.get("function") == "particleswarmoptimization"):
        invalid_flag = False
        
        #PSO housekeeping variables.
        scores = []
        p_best_particles = []
        p_best_scores = []
        g_best = []
        g_best_score = 99999
        g_best_part = 0
        g_best_it = 0

        if (settings.get("algorithm") == "kmeans"):
            print "\nMaximum number of clusters:", settings.get("max_k")
            print "Population size:", settings.get("pop_size")
            print "Number of generations:", settings.get("max_iter")
            print "Activation Threshold:", settings.get("activation_threshold")
            print "Omega range: " + str(settings.get("omega_min")) + " - " + str(settings.get("omega_max"))
            print "Velocity Weight:", settings.get("weight")
            print "-------------------------------"
            #Iterating up to the maximum number of iterations.
            for a in xrange(0, settings.get("max_iter")):
                #Initializing the fitness scores.
                for i in xrange(0, settings.get("pop_size")):
                    scores.append(9999) 

                print "\nGeneration", a + 1
                i = 0
                #Iterating over each particle.
                while (i < settings.get("pop_size")):
                    invalid_flag = False
                    #Initializing the feature vectors.
                    if (debug_flag == True):
                        print "Initializing data vectors...\n"
                    if (settings.get("dataset") == "iris"):
                        all_vects = datasets.iris.get_feat_vects()
                    if (settings.get("dataset") == "wisconsin"):
                        all_vects = datasets.wisconsin.get_feat_vects()
                    if (settings.get("dataset") == "s1"):
                        all_vects = datasets.s1.get_feat_vects()
                    if (settings.get("dataset") == "dim3"):
                        all_vects = datasets.dim3.get_feat_vects()
                    if (settings.get("dataset") == "spiral"):
                        all_vects = datasets.spiral.get_feat_vects()
                    if (settings.get("dataset") == "flame"):
                        all_vects = datasets.flame.get_feat_vects()
                    feat_vects = random.sample(all_vects, settings.get("num_data"))
                    #If it is the first iteration of the program, initiate particles and velocities.
                    if((i == 0) & (a == 0)):
                        if (debug_flag == True):
                            print "Initializing particles and velocities..."
                        particles = particle_swarm_optimization.init_pso(feat_vects, label_pos, settings.get("max_k"), settings.get("pop_size"), settings.get("activation_threshold"))
                        velocities = particle_swarm_optimization.init_velocities(feat_vects, label_pos, settings.get("max_k"), len(particles[0]), settings.get("pop_size"))                    
                    if (debug_flag == True):    
                        print "\nAnalyzing particle", i + 1, "..."
                        print particles[i]
                    #Decode the centers from the particle.
                    centers = particle_swarm_optimization.get_center(particles[i], settings.get("max_k"), settings.get("activation_threshold"), label_pos)
                    #Decode the k-value from the particle.
                    k = len(centers)
                    #If the number of centers is less than 2, the particle is invalid.
                    if (k < 2):
                        invalid_flag = True
                        #Reinitialize the invalid particle.
                        particles[i] = particle_swarm_optimization.reinit_particle(particles[i], feat_vects, label_pos, settings.get("max_k"), settings.get("activation_threshold"), debug_flag)
                        continue
                    if (debug_flag == True):
                        print "Centers:", centers
                        print "\nClustering data...\n"
                    #Cluster the data according to the centers in the particle.
                    clustered_data = algorithms.k_means.cluster_data(centers, feat_vects, True, feat_num)
                    if (debug_flag == True):
                        print "Counting cluster members...\n"
                    #Count the cluster members.
                    new_count = algorithms.k_means.count_cluster_items(clustered_data, k, label_pos)
                    if (debug_flag == True):
                        print new_count
                    converge_flag = False
                    counter = 1
                    #This loop executes until the algorithm converges.
                    while not bool(converge_flag):
                        if (debug_flag == True):
                            print "\nIteration", counter, "( Particle", i + 1, ")"
                        old_count = new_count
                        if (debug_flag == True):
                            print "\nCalculating new cluster centers...\n"
                        #Calculate the new centers.
                        centers = algorithms.k_means.calculate_centroids(clustered_data, k, label_pos, feat_num)
                        #If the function returns true or there is a 0 count, the particle is invalid.
                        if((centers == True) or (0 in old_count)):
                            invalid_flag = True
                            #Reinitialize the invalid particle.
                            particles[i] = particle_swarm_optimization.reinit_particle(particles[i], feat_vects, label_pos, settings.get("max_k"), settings.get("activation_threshold"), debug_flag)
                            break
                        if (debug_flag == True):
                            print "Clustering data...\n"
                        #Cluster the data
                        clustered_data = algorithms.k_means.cluster_data(centers, feat_vects, False, feat_num)
                        if (debug_flag == True):
                            print "Counting cluster members...\n"
                        #Count the members of the clusters.
                        new_count = algorithms.k_means.count_cluster_items(clustered_data, k, label_pos)
                        if (debug_flag == True):
                            print new_count
                        #Check if the algorithm has converged.
                        converge_flag = algorithms.k_means.compare_counts(old_count, new_count, k)
                        counter += 1 
                    #If the particle is valid and the algorithm has converged, this block executes.
                    if not invalid_flag:
                        if (debug_flag == True):
                            print "\nCount has remained unchanged, clusters have converged to an optima."
                            print  "Evaluating clusters..."
                        #Calculate the cluster fitness.
                        if (settings.get("fitness_func") == "dbi"):
                            dbi = stats.calc_dbi(feat_vects, centers, k, label_pos, label_pos)
                            if (debug_flag == True):
                                print "Calculating score:", dbi
                            #If it is the first iteration for the particle, set the particle as its personal best.
                            if (a == 0):
                                temp_particle = particles[i][:]
                                p_best_particles.append(temp_particle)
                                p_best_scores.append(dbi)
                            #If it is the first iteration for the program, set the particle as the global best
                            if ((a == 0) & (i == 0)):
                                g_best = particles[i][:]
                                g_best_score = dbi
                                g_best_it = a + 1
                                g_best_part = i + 1
                            #If the DBI is better than the saved personal best, set the particle as its personal best.
                            if (dbi < p_best_scores[i]):
                                temp_particle = particles[i][:]
                                p_best_particles[i] = temp_particle
                                p_best_scores[i] = dbi
                            #If the DBI is better than the saved global best, set the particle as the global best.
                            if (dbi < g_best_score):
                                g_best = particles[i][:]
                                g_best_score = dbi
                                g_best_it = a + 1
                                g_best_part = i + 1
                        if (settings.get("fitness_func") == "silhouette"):
                            silhouette = stats.calc_silhouette(feat_vects, k, label_pos, label_pos)
                            if (debug_flag == True):
                                print "Calculating score:", silhouette
                            #If it is the first iteration for the particle, set the particle as its personal best.
                            if (a == 0):
                                temp_particle = particles[i][:]
                                p_best_particles.append(temp_particle)
                                p_best_scores.append(silhouette)
                            #If it is the first iteration for the program, set the particle as the global best
                            if ((a == 0) & (i == 0)):
                                g_best = particles[i][:]
                                g_best_score = silhouette
                                g_best_it = a + 1
                                g_best_part = i + 1
                            #If the silhouette index is better than the saved personal best, set the particle as its personal best.
                            if (silhouette > p_best_scores[i]):
                                temp_particle = particles[i][:]
                                p_best_particles[i] = temp_particle
                                p_best_scores[i] = silhouette
                            #If the silhouette index is better than the saved global best, set the particle as the global best.
                            if (silhouette > g_best_score):
                                g_best = particles[i][:]
                                g_best_score = silhouette
                                g_best_it = a + 1
                                g_best_part = i + 1                            
                        counter = counter + 1
                        if (debug_flag == True):
                            print "\nAfter Generation", a + 1, "the personal best for particle", i + 1, " is:"
                            print p_best_particles[i], p_best_scores[i]
                            print "\nMoving particle", i + 1, "in solution space..."
                        #Update the particle's position in the solution space using it's velocity.
                        particles[i] = particle_swarm_optimization.update_position(particles[i], velocities[i], settings.get("max_k"), label_pos, feat_vects, debug_flag)
                        #Update the velocity.
                        velocities[i] = particle_swarm_optimization.update_velocity(settings.get("omega_min"), settings.get("omega_max"), particles[i], velocities[i], settings.get("weight"), p_best_particles[i], g_best, debug_flag)
                        i = i + 1
                if((debug_flag == True) & (a != (settings.get("max_iter") - 1))):
                    print "\nAfter Generation", a + 1, ", the swarm best is:"
                    print g_best, g_best_score, " (Particle " + str(g_best_part) + ", Generation " + str(g_best_it) + ".)" 
            #Once the algorithm finishes, final results are printed.
            print "\nWith a fitness of " + str(g_best_score) + ", The best particle found is:"
            print g_best
            best_centers = particle_swarm_optimization.get_center(g_best, settings.get("max_k"), settings.get("activation_threshold"), feat_num)
            best_k = len(best_centers)
            print "\nThis translates to a K-value of: ", best_k
            print "Using the centroids: "
            for i in xrange(0, best_k):
                print best_centers[i]
            #Calculate runtime.
            stop_time = timeit.default_timer()
            runtime_mins = int(math.trunc(((stop_time - start_time) / 60)))
            runtime_secs = int(round(((stop_time - start_time) % 60), 0))
            print "\n\nRuntime:", runtime_mins, "minute(s)", runtime_secs, "second(s)"
            
            #The best chromosome is then run through the algorithm and results are plotted.
            if (settings.get("plot") == "on"):
                print "\nPlotting data."
                #Clustering the data
                clustered_data = algorithms.k_means.cluster_data(best_centers, feat_vects, False, feat_num)
                #Counting the cluster members.
                count = algorithms.k_means.count_cluster_items(clustered_data, best_k, label_pos)
                converge_flag = False
                counter = 1
                #This code repeatedly executes until the algorithm converges (i.e cluster member count remains stable between iterations).
                while not converge_flag:
                    old_count = count
                    #Calculating the new centers.
                    centers = algorithms.k_means.calculate_centroids(clustered_data, best_k, label_pos, feat_num)
                    #Clustering the data.
                    clustered_data = algorithms.k_means.cluster_data(centers, feat_vects, False, feat_num)
                    #Counting the cluster members.
                    count = algorithms.k_means.count_cluster_items(clustered_data, best_k, label_pos)
                    #Check if the algorithm has converged. While loop terminates if it has.
                    converge_flag = algorithms.k_means.compare_counts(old_count, count, best_k)
                    counter += 1
                colours = ["#FF0000", "#0000FF", "#00FF00", "#00CCFF", "#FF33CC", "#FFFF00", "#000000", "#CCCCCC", "#522900", "#FFFFCC",  "#99CC00", "#FF9999", "#336699", "#006600", "#CCFFFF", "#CCCC00", "#FFCCFF", "#9900FF", "#006666", "#993366"]
                for i in xrange(0, best_k):
                    for j in xrange(0, len(clustered_data)):
                        if(clustered_data[j][label_pos] == i):
                            plt.scatter(clustered_data[j][0], clustered_data[j][1], s=10, color=colours[i])
                plt.show()

        if (settings.get("algorithm") == "fuzzycmeans"):
            invalid_flag = False
            print "\nMaximum number of clusters:", settings.get("max_k")
            print "Population size:", settings.get("pop_size")
            print "Number of generations:", settings.get("max_iter")
            print "Activation Threshold:", settings.get("activation_threshold")
            print "Convergence Threshold:", settings.get("convergence_threshold")
            print "Omega range: " + str(settings.get("omega_min")) + " - " + str(settings.get("omega_max"))
            print "Velocity Weight:", settings.get("weight")
            print "Fuzziness:", settings.get("m")
            print "-------------------------------"
            #PSO housekeeping variables.
            scores = []
            p_best_particles = []
            p_best_scores = []
            g_best = []
            g_best_score = 99999
            g_best_part = 0
            g_best_it = 0

            #Iterating up to the maximum generation.
            for a in xrange(0, settings.get("max_iter")):
                #Initializing the fitnesses.
                for i in xrange(0, settings.get("pop_size")):
                    scores.append(9999)  

                print "\nGeneration", a + 1
                i = 0
                #Iterating over each chromosome.
                while (i < settings.get("pop_size")):
                    invalid_flag = False
                    #Initialize the feature vectors.
                    if (debug_flag == True):
                        print "Initializing data vectors...\n"
                    if (settings.get("dataset") == "iris"):
                        all_vects = datasets.iris.get_feat_vects()
                    if (settings.get("dataset") == "wisconsin"):
                        all_vects = datasets.wisconsin.get_feat_vects()
                    if (settings.get("dataset") == "s1"):
                        all_vects = datasets.s1.get_feat_vects()
                    if (settings.get("dataset") == "dim3"):
                        all_vects = datasets.dim3.get_feat_vects()
                    if (settings.get("dataset") == "spiral"):
                        all_vects = datasets.spiral.get_feat_vects()
                    if (settings.get("dataset") == "flame"):
                        all_vects = datasets.flame.get_feat_vects()
                    feat_vects = random.sample(all_vects, settings.get("num_data"))
                    if (i == 0) & (a == 0):  
                        if (debug_flag == True):         
                            print "Initializing particles..."
                        #Initialize the particles/velocities if it is the program's first iteration.
                        particles = particle_swarm_optimization.init_pso(feat_vects, label_pos, settings.get("max_k"), settings.get("pop_size"), settings.get("activation_threshold"))
                        velocities = particle_swarm_optimization.init_velocities(feat_vects, label_pos, settings.get("max_k"), len(particles[0]), settings.get("pop_size"))                    
                    if (debug_flag == True):
                        print "\nAnalyzing chromosome", i + 1, "..."
                        print particles[i]
                    #Decode the centers from the particle.
                    centers = particle_swarm_optimization.get_center(particles[i], settings.get("max_k"), settings.get("activation_threshold"), label_pos)
                    #Decode the k-value from the particle.
                    k = len(centers)
                    #If there are fewer than 2 centers, the particle is invalid
                    if (k < 2):
                        invalid_flag = True
                        #Reinitialize the invalid particle.
                        particles[i] = particle_swarm_optimization.reinit_particle(particles[i], feat_vects, label_pos, settings.get("max_k"), settings.get("activation_threshold"), debug_flag)
                        continue
                    if (debug_flag == True):
                        print "Generating partition matrix..."
                    #Randomly generate a partition matrix and calculate its L2 norm.
                    partition_matrix_before = algorithms.fuzzy_c_means.get_partition_matrix(feat_vects, k)
                    dist_before = numpy.linalg.norm(partition_matrix_before)
                    if (debug_flag == True):
                        print "Updating partition matrix..."
                    #Update the partition matrix using the particle's centers and calculate its L2 norm.
                    partition_matrix_after = algorithms.fuzzy_c_means.update_partition_matrix(feat_vects, partition_matrix_before, k, feat_num, centers, settings.get("m"))
                    dist_after = numpy.linalg.norm(partition_matrix_after)
                    #Calculate convergence using the L2 norms.
                    convergence = abs(dist_after - dist_before)
                    if (debug_flag == True):
                        print "Convergence (threshold of " + str(settings.get("convergence_threshold")) + "):", convergence
                    counter = 1
                    #This executes until the convergence is less than the threshold.
                    while (convergence > settings.get("convergence_threshold")):
                        if (debug_flag == True):
                            print "\nIteration", counter, "( Chromosome", i + 1, ")"
                            print "Updating partition matrix..."
                        partition_matrix_before = partition_matrix_after
                        dist_before = dist_after
                        if (debug_flag == True):
                            print "Calcuating centers..."
                        #New centers are calculated.
                        centers = algorithms.fuzzy_c_means.calculate_centroids(feat_vects, partition_matrix_before, k, feat_num, settings.get("m"))
                        #Partition matrix is updated and its L2 norm is calculated.
                        partition_matrix_after = algorithms.fuzzy_c_means.update_partition_matrix(feat_vects, partition_matrix_before, k, feat_num, centers, settings.get("m"))
                        dist_after = numpy.linalg.norm(partition_matrix_after)
                        #Convergence is checked.
                        convergence = abs(dist_after - dist_before)
                        if (debug_flag == True):
                            print "Convergence (threshold of " + str(settings.get("convergence_threshold")) + "):", convergence
                        counter += 1                    
                    if (debug_flag == True):    
                        print "\nAlgorithm has converged on Generation " + str(counter) + "."
                        print "Counting cluster items..."
                    #Once the algorithm has converged, cluster membership is counted.
                    count = algorithms.fuzzy_c_means.count_cluster_items(partition_matrix_after, k)
                    if (debug_flag == True):
                        print count
                        print "Checking chromosome validity..."
                    #Particle is checked for validity.
                    invalid_flag = algorithms.fuzzy_c_means.check_count(count, k)
                    if (invalid_flag == False):
                        labelled_data = algorithms.fuzzy_c_means.label_data(feat_vects, partition_matrix_after, k)
                        if (settings.get("fitness_func") == "dbi"):
                            #Calculate particle's fitness
                            dbi = stats.calc_dbi(labelled_data, centers, k, label_pos, feat_num)
                            if (debug_flag == True):
                                print "Calculating score...", dbi
                            #If it is the first iteration for the particle, set the particle as its personal best.
                            if (a == 0):
                                p_best_particles.append(particles[i][:])
                                p_best_scores.append(dbi)
                            #If it is the first iteration for the program, set the particle as the global best
                            if ((a ==0) & (i == 0)):
                                g_best = particles[i][:]
                                g_best_score = dbi
                                g_best_it = a + 1
                                g_best_part = i + 1
                            #If the DBI is better than the saved personal best, set the particle as its personal best.
                            if (dbi < p_best_scores[i]):
                                p_best_particles[i] = particles[i][:]
                                p_best_scores[i] = dbi
                            #If the DBI is better than the saved global best, set the particle as its personal best.
                            if (dbi < g_best_score):
                                g_best = particles[i][:]
                                g_best_score = dbi
                                g_best_it = a + 1
                                g_best_part = i + 1
                        if (settings.get("fitness_func") == "silhouette"):
                            #Calculate particle's fitness.
                            silhouette = stats.calc_silhouette(labelled_data, k, label_pos, feat_num) 
                            if (debug_flag == True):
                                print "Calculating score...", silhouette
                            #If it is the first iteration for the particle, set the particle as its personal best.
                            if (a == 0):
                                p_best_particles.append(particles[i][:])
                                p_best_scores.append(silhouette)
                            #If it is the first iteration for the program, set the particle as the global best
                            if ((a ==0) & (i == 0)):
                                g_best = particles[i][:]
                                g_best_score = silhouette
                                g_best_it = a + 1
                                g_best_part = i + 1
                            #If the silhouette index is better than the saved personal best, set the particle as its personal best.
                            if (silhouette > p_best_scores[i]):
                                p_best_particles[i] = particles[i][:]
                                p_best_scores[i] = silhouette
                            #If the silhouette index is better than the saved global best, set the particle as its personal best.
                            if (silhouette > g_best_score):
                                g_best = particles[i][:]
                                g_best_score = silhouette
                                g_best_it = a + 1
                                g_best_part = i + 1
                        counter = counter + 1

                        if (debug_flag == True):
                            print "\nAfter Generation", a + 1, "the personal best for particle", i + 1, " is:"
                            print p_best_particles[i], p_best_scores[i]
                            print "\nMoving particle", i + 1, "in solution space..."
                        #Update the particle's position in the solution space using it's velocity.
                        particles[i] = particle_swarm_optimization.update_position(particles[i], velocities[i], settings.get("max_k"), label_pos, feat_vects, debug_flag)
                        #Update the velocity.
                        velocities[i] = particle_swarm_optimization.update_velocity(settings.get("omega_min"), settings.get("omega_max"), particles[i], velocities[i], settings.get("weight"), p_best_particles[i], g_best, debug_flag)
                        i = i + 1
                    else:
                        #Reinitialize invalid chromosomes and begin the iteration again.
                        particles[i] = particle_swarm_optimization.reinit_particle(particles[i], feat_vects, label_pos, settings.get("max_k"), settings.get("activation_threshold"), debug_flag)                
                if((debug_flag == True) & (a != (settings.get("max_iter") - 1))):
                    print "\nAfter Generation", a + 1, ", the swarm best is:"
                    print g_best, g_best_score, " (Particle " + str(g_best_part) + ", Generation " + str(g_best_it) + ".)" 
            #Once the algorithm has converged, cluster membership is counted.
            print "\nWith a fitness of " + str(g_best_score) + ", The best particle found is:"
            print g_best
            best_centers = particle_swarm_optimization.get_center(g_best, settings.get("max_k"), settings.get("activation_threshold"), feat_num)
            best_k = len(best_centers)
            print "\nThis translates to a K-value of: ", best_k
            print "Using the centroids: "
            for i in xrange(0, best_k):
                print best_centers[i]

            #Calculate runtime.
            stop_time = timeit.default_timer()
            runtime_mins = int(math.trunc(((stop_time - start_time) / 60)))
            runtime_secs = int(round(((stop_time - start_time) % 60), 0))
            print "\n\nRuntime:", runtime_mins, "minute(s)", runtime_secs, "second(s)"
            
            if (settings.get("plot") == "on"):
                print "\nPlotting data."
                #Randomly generating a partition matrix and calculating its L2 norm.
                partition_matrix_before = algorithms.fuzzy_c_means.get_partition_matrix(feat_vects, best_k)
                dist_before = numpy.linalg.norm(partition_matrix_before)
                #Updating the partition matrix using the centers from the chromosome and calculating its L2 norm.
                partition_matrix_after = algorithms.fuzzy_c_means.update_partition_matrix(feat_vects, partition_matrix_before, best_k, feat_num, best_centers, settings.get("m"))
                dist_after = numpy.linalg.norm(partition_matrix_after)
                #Calculating the convergence using the L2 norms.
                convergence = abs(dist_after - dist_before)
                #Executes until the algorithm converges.
                while (convergence > settings.get("convergence_threshold")):
                    partition_matrix_before = partition_matrix_after
                    dist_before = dist_after
                    #Centers are calculated and used to update the partition matrix and L2 norm.
                    centers = algorithms.fuzzy_c_means.calculate_centroids(feat_vects, partition_matrix_before, best_k, feat_num, settings.get("m"))
                    partition_matrix_after = algorithms.fuzzy_c_means.update_partition_matrix(feat_vects, partition_matrix_before, best_k, feat_num, centers, settings.get("m"))
                    dist_after = numpy.linalg.norm(partition_matrix_after)
                    #Convergence is calculated and checked against the threshold.
                    convergence = abs(dist_after - dist_before)
                #Data is labelled with cluster information.
                labelled_data = algorithms.fuzzy_c_means.label_data(feat_vects, partition_matrix_after, best_k)
                colours = ["#FF0000", "#0000FF", "#00FF00", "#00CCFF", "#FF33CC", "#FFFF00", "#000000", "#CCCCCC", "#522900", "#FFFFCC",  "#99CC00", "#FF9999", "#336699", "#006600", "#CCFFFF", "#CCCC00", "#FFCCFF", "#9900FF", "#006666", "#993366"]
                for i in xrange(0, best_k):
                    for j in xrange(0, len(labelled_data)):
                        if(labelled_data[j][label_pos] == i):
                            plt.scatter(labelled_data[j][0], labelled_data[j][1], s=10, color=colours[i])
                plt.show()
            
        if (settings.get("algorithm") == "spectralclustering"):      
            print "\nMaximum number of clusters:", settings.get("max_k")
            print "Population size:", settings.get("pop_size")
            print "Number of generations:", settings.get("max_iter")
            print "Activation Threshold:", settings.get("activation_threshold")
            print "Similarity Threshold:", settings.get("similarity_threshold")
            print "Omega range: " + str(settings.get("omega_min")) + " - " + str(settings.get("omega_max"))
            print "Velocity Weight:", settings.get("weight")
            print "-------------------------------"
            #Iterating up to the maximum number of iterations.
            for a in xrange(0, settings.get("max_iter")):
                #Initializing the fitness scores.
                for i in xrange(0, settings.get("pop_size")):
                    scores.append(9999) 

                print "\nGeneration", a + 1
                i = 0
                #Iterating over each particle.
                while (i < settings.get("pop_size")):
                    invalid_flag = False
                    #Initializing the feature vectors.
                    if (debug_flag == True):
                        print "Initializing data vectors...\n"
                    if (settings.get("dataset") == "iris"):
                        all_vects = datasets.iris.get_feat_vects()
                    if (settings.get("dataset") == "wisconsin"):
                        all_vects = datasets.wisconsin.get_feat_vects()
                    if (settings.get("dataset") == "s1"):
                        all_vects = datasets.s1.get_feat_vects()
                    if (settings.get("dataset") == "dim3"):
                        all_vects = datasets.dim3.get_feat_vects()
                    if (settings.get("dataset") == "spiral"):
                        all_vects = datasets.spiral.get_feat_vects()
                    if (settings.get("dataset") == "flame"):
                        all_vects = datasets.flame.get_feat_vects()
                    if (i == 0) & (a == 0):
                        feat_vects = random.sample(all_vects, settings.get("num_data"))
                        label_pos = settings.get("num_data")
                        feat_num = settings.get("num_data")
                        #Initializing the required matrices for spectral clustering.
                        dist_matrix = algorithms.spectral_clustering.get_distance_matrix(feat_vects)
                        adj_matrix = algorithms.spectral_clustering.get_adjacency_matrix(dist_matrix, settings.get("similarity_threshold"))
                        deg_matrix = algorithms.spectral_clustering.get_degree_matrix_simple(adj_matrix)
                        lap_matrix = algorithms.spectral_clustering.get_unnormalized_laplacian_matrix(deg_matrix, adj_matrix)
                        eigenvalues, eigenvectors = numpy.linalg.eig(lap_matrix)
                        clust_matrix = algorithms.spectral_clustering.get_cluster_matrix(eigenvectors)
                    #If it is the first iteration of the program, initiate particles and velocities.
                    if((i == 0) & (a == 0)):
                        if (debug_flag == True):
                            print "Initializing particles and velocities..."
                        particles = particle_swarm_optimization.init_pso(clust_matrix, label_pos, settings.get("max_k"), settings.get("pop_size"), settings.get("activation_threshold"))
                        velocities = particle_swarm_optimization.init_velocities(clust_matrix, label_pos, settings.get("max_k"), len(particles[0]), settings.get("pop_size"))                    
                    if (debug_flag == True):    
                        print "\nAnalyzing particle", i + 1, "..."
                        print particles[i]
                    #Decode the centers from the particle.
                    centers = particle_swarm_optimization.get_center(particles[i], settings.get("max_k"), settings.get("activation_threshold"), label_pos)
                    #Decode the k-value from the particle.
                    k = len(centers)
                    #If the number of centers is less than 2, the particle is invalid.
                    if (k < 2):
                        invalid_flag = True
                        #Reinitialize the invalid particle.
                        particles[i] = particle_swarm_optimization.reinit_particle(particles[i], clust_matrix, label_pos, settings.get("max_k"), settings.get("activation_threshold"), debug_flag)
                        continue
                    if (debug_flag == True):
                        print "Centers:", centers
                        print "\nClustering data...\n"
                    #Cluster the data according to the centers in the particle.
                    clustered_data = algorithms.k_means.cluster_data(centers, clust_matrix, True, feat_num)
                    if (debug_flag == True):
                        print "Counting cluster members...\n"
                    #Count the cluster members.
                    new_count = algorithms.k_means.count_cluster_items(clustered_data, k, label_pos)
                    if (debug_flag == True):
                        print new_count
                    converge_flag = False
                    counter = 1
                    #This loop executes until the algorithm converges.
                    while not bool(converge_flag):
                        if (debug_flag == True):
                            print "\nIteration", counter, "( Particle", i + 1, ")"
                        old_count = new_count
                        if (debug_flag == True):
                            print "\nCalculating new cluster centers...\n"
                        #Calculate the new centers.
                        centers = algorithms.k_means.calculate_centroids(clustered_data, k, label_pos, feat_num)
                        #If the function returns true or there is a 0 count, the particle is invalid.
                        if((centers == True) or (0 in old_count)):
                            invalid_flag = True
                            #Reinitialize the invalid particle.
                            particles[i] = particle_swarm_optimization.reinit_particle(particles[i], clust_matrix, label_pos, settings.get("max_k"), settings.get("activation_threshold"), debug_flag)
                            break
                        if (debug_flag == True):
                            print "Clustering data...\n"
                        #Cluster the data
                        clustered_data = algorithms.k_means.cluster_data(centers, clust_matrix, False, feat_num)
                        if (debug_flag == True):
                            print "Counting cluster members...\n"
                        #Count the members of the clusters.
                        new_count = algorithms.k_means.count_cluster_items(clustered_data, k, label_pos)
                        if (debug_flag == True):
                            print new_count
                        #Check if the algorithm has converged.
                        converge_flag = algorithms.k_means.compare_counts(old_count, new_count, k)
                        counter += 1 
                    #If the particle is valid and the algorithm has converged, this block executes.
                    if not invalid_flag:
                        if (debug_flag == True):
                            print "\nCount has remained unchanged, clusters have converged to an optima."
                            print  "Evaluating clusters..."
                        #Calculate the cluster fitness (evaluates how good the clusters are).
                        if (settings.get("fitness_func") == "dbi"):
                            dbi = stats.calc_dbi(clust_matrix, centers, k, label_pos, label_pos)
                            if (debug_flag == True):
                                print "Calculating score:", dbi
                            #If it is the first iteration for the particle, set the particle as its personal best.
                            if (a == 0):
                                p_best_particles.append(particles[i][:])
                                p_best_scores.append(dbi)
                            #If it is the first iteration for the program, set the particle as the global best
                            if ((a ==0) & (i == 0)):
                                g_best = particles[i][:]
                                g_best_score = dbi
                                g_best_it = a + 1
                                g_best_part = i + 1
                            #If the DBI is better than the saved personal best, set the particle as its personal best.
                            if (dbi < p_best_scores[i]):
                                p_best_particles[i] = particles[i][:]
                                p_best_scores[i] = dbi
                            #If the DBI is better than the saved global best, set the particle as the global best.
                            if (dbi < g_best_score):
                                g_best = particles[i][:]
                                g_best_score = dbi
                                g_best_it = a + 1
                                g_best_part = i + 1
                        if (settings.get("fitness_func") == "silhouette"):
                            silhouette = stats.calc_silhouette(clust_matrix, k, label_pos, label_pos)
                            if (debug_flag == True):
                                print "Calculating score:", silhouette
                            #If it is the first iteration for the particle, set the particle as its personal best.
                            if (a == 0):
                                p_best_particles.append(particles[i][:])
                                p_best_scores.append(silhouette)
                            #If it is the first iteration for the program, set the particle as the global best
                            if ((a ==0) & (i == 0)):
                                g_best = particles[i][:]
                                g_best_score = silhouette
                                g_best_it = a + 1
                                g_best_part = i + 1
                            #If the silhouette index is better than the saved personal best, set the particle as its personal best.
                            if (silhouette > p_best_scores[i]):
                                p_best_particles[i] = particles[i][:]
                                p_best_scores[i] = silhouette
                            #If the silhouette index is better than the saved global best, set the particle as the global best.
                            if (silhouette > g_best_score):
                                g_best = particles[i][:]
                                g_best_score = silhouette
                                g_best_it = a + 1
                                g_best_part = i + 1
                            
                        counter = counter + 1
                        if (debug_flag == True):
                            print "\nAfter Generation", a + 1, "the personal best for particle", i + 1, " is:"
                            print p_best_particles[i], p_best_scores[i]
                            print "\nMoving particle", i + 1, "in solution space..."
                        #Update the particle's position in the solution space using it's velocity.
                        particles[i] = particle_swarm_optimization.update_position(particles[i], velocities[i], settings.get("max_k"), label_pos, clust_matrix, debug_flag)
                        #Update the velocity.
                        velocities[i] = particle_swarm_optimization.update_velocity(settings.get("omega_min"), settings.get("omega_max"), particles[i], velocities[i], settings.get("weight"), p_best_particles[i], g_best, debug_flag)
                        i = i + 1
                if((debug_flag == True) & (a != (settings.get("max_iter") - 1))):
                    print "\nAfter Generation", a + 1, ", the swarm best is:"
                    print g_best, g_best_score, " (Particle " + str(g_best_part) + ", Generation " + str(g_best_it) + ".)" 
            #Once the algorithm finishes, final results are printed.
            print "\nWith a fitness of " + str(g_best_score) + ", The best particle found is:"
            print g_best
            best_centers = particle_swarm_optimization.get_center(g_best, settings.get("max_k"), settings.get("activation_threshold"), feat_num)
            best_k = len(best_centers)
            print "\nThis translates to a K-value of: ", best_k
            print "Using the centroids: "
            for i in xrange(0, best_k):
                print
                print best_centers[i]
            #Calculate runtime.
            stop_time = timeit.default_timer()
            runtime_mins = int(math.trunc(((stop_time - start_time) / 60)))
            runtime_secs = int(round(((stop_time - start_time) % 60), 0))
            print "\n\nRuntime:", runtime_mins, "minute(s)", runtime_secs, "second(s)"
            
            if (settings.get("plot") == "on"):
                print "\nPlotting data."
                #Clustering the data using the decoded centers.
                clustered_data = algorithms.k_means.cluster_data(best_centers, clust_matrix, True, feat_num)
                #Counting the cluster members.
                count = algorithms.k_means.count_cluster_items(clustered_data, best_k, label_pos)
                converge_flag = False
                counter = 1
                #This code repeatedly executes until the algorithm converges (i.e cluster member count remains stable between iterations).
                while not converge_flag:
                    old_count = count
                    #Calculating the new centers.
                    centers = algorithms.k_means.calculate_centroids(clustered_data, best_k, label_pos, feat_num)
                    #Checking chromosome validity.
                    #Clustering the data.
                    clustered_data = algorithms.k_means.cluster_data(centers, clust_matrix, False, feat_num)
                    #Counting the cluster members.
                    count = algorithms.k_means.count_cluster_items(clustered_data, best_k, label_pos)
                    #Check if the algorithm has converged. While loop terminates if it has.
                    converge_flag = algorithms.k_means.compare_counts(old_count, count, best_k)
                    counter += 1
                colours = ["#FF0000", "#0000FF", "#00FF00", "#00CCFF", "#FF33CC", "#FFFF00", "#000000", "#CCCCCC", "#522900", "#FFFFCC",  "#99CC00", "#FF9999", "#336699", "#006600", "#CCFFFF", "#CCCC00", "#FFCCFF", "#9900FF", "#006666", "#993366"]
                for i in xrange(0, best_k):
                    for j in xrange(0, len(clustered_data)):
                        if(clustered_data[j][label_pos] == i):
                            plt.scatter(clustered_data[j][0], clustered_data[j][1], s=10, color=colours[i])
                plt.show()
                
################################################################################
############################### Base Functions #################################
################################################################################             
                
    if (settings.get("function") == "none"):
        
        if (settings.get("algorithm") == "kmeans"):
            print "\nNumber of clusters:", settings.get("clusters")
            print "-------------------------------"
            #Intitialize the feature vectors.
            if (debug_flag == True):
                print "\nInitializing data vectors...\n"
            if (settings.get("dataset") == "iris"):
                all_vects = datasets.iris.get_feat_vects()
            if (settings.get("dataset") == "wisconsin"):
                all_vects = datasets.wisconsin.get_feat_vects()
            if (settings.get("dataset") == "s1"):
                all_vects = datasets.s1.get_feat_vects()
            if (settings.get("dataset") == "dim3"):
                all_vects = datasets.dim3.get_feat_vects()
            if (settings.get("dataset") == "spiral"):
                all_vects = datasets.spiral.get_feat_vects()
            if (settings.get("dataset") == "flame"):
                all_vects = datasets.flame.get_feat_vects()
            feat_vects = random.sample(all_vects, settings.get("num_data"))
            k = settings.get("clusters")
            centers = random.sample(feat_vects, k)
            if (debug_flag == True):
                print "Clustering data..."
            #Clustering the data using the decoded centers.
            clustered_data = algorithms.k_means.cluster_data(centers, feat_vects, True, feat_num)
            if (debug_flag == True):
                print "Counting cluster members..."
            #Counting the cluster members.
            count = algorithms.k_means.count_cluster_items(clustered_data, k, label_pos)
            converge_flag = False
            counter = 1
            #This code repeatedly executes until the algorithm converges (i.e cluster member count remains stable between iterations).
            while not converge_flag:
                if (debug_flag == True):
                    print "\nIteration " + str(counter) + "."
                old_count = count
                if (debug_flag == True):
                    print "\nCalculating new cluster centers..."
                #Calculating the new centers.
                centers = algorithms.k_means.calculate_centroids(clustered_data, k, label_pos, feat_num)
                if (debug_flag == True):    
                    print "Clustering data..."
                #Clustering the data.
                clustered_data = algorithms.k_means.cluster_data(centers, feat_vects, False, feat_num)
                if (debug_flag == True):
                    print "Counting cluster members..."
                #Counting the cluster members.
                count = algorithms.k_means.count_cluster_items(clustered_data, k, label_pos)
                if (debug_flag == True):
                    print count
                #Check if the algorithm has converged. While loop terminates if it has.
                converge_flag = algorithms.k_means.compare_counts(old_count, count, k)
                counter += 1    
            if (debug_flag == True):
                print "\nCount has stabilized, clusters have converged to an optima."
                print  "Evaluating clusters..."
            #The cluster fitness is calculated and saved.
            if (settings.get("fitness_func") == "dbi"):
                dbi = stats.calc_dbi(feat_vects, centers, k, label_pos, label_pos)                            
                print "\nCluster fitness (lower is better):", dbi
            if (settings.get("fitness_func") == "silhouette"):
                silhouette = stats.calc_silhouette(clustered_data, k, label_pos, feat_num)
                print "\nCluster fitness (closer to one is better):", silhouette
            print "\nCenters:"
            for i in xrange(0, k):
                print centers[i]
            #The timer is stopped and runtime is calculated.
            stop_time = timeit.default_timer()
            runtime_mins = int(math.trunc(((stop_time - start_time) / 60)))
            runtime_secs = int(round(((stop_time - start_time) % 60), 0))
            print "\nRuntime:", runtime_mins, "minute(s)", runtime_secs, "second(s)"
            
            #The best chromosome is then run through the algorithm and results are plotted.
            if (settings.get("plot") == "on"):
                colours = ["#FF0000", "#0000FF", "#00FF00", "#00CCFF", "#FF33CC", "#FFFF00", "#000000", "#CCCCCC", "#522900", "#FFFFCC",  "#99CC00", "#FF9999", "#336699", "#006600", "#CCFFFF", "#CCCC00", "#FFCCFF", "#9900FF", "#006666", "#993366"]
                for i in xrange(0, k):
                    for j in xrange(0, len(clustered_data)):
                        if(clustered_data[j][label_pos] == i):
                            plt.scatter(clustered_data[j][0], clustered_data[j][1], s=10, color=colours[i])
                plt.show()
                
        if (settings.get("algorithm") == "fuzzycmeans"):
            print "\nNumber of clusters:", settings.get("clusters")
            print "Convergence Threshold:", settings.get("convergence_threshold")
            print "Fuzziness:", settings.get("m")
            print "-------------------------------"
            #Initialize the feature vectors.
            if (debug_flag == True):
                print "\nInitializing data vectors...\n"
            if (settings.get("dataset") == "iris"):
                all_vects = datasets.iris.get_feat_vects()
            if (settings.get("dataset") == "wisconsin"):
                all_vects = datasets.wisconsin.get_feat_vects()
            if (settings.get("dataset") == "s1"):
                all_vects = datasets.s1.get_feat_vects()
            if (settings.get("dataset") == "dim3"):
                all_vects = datasets.dim3.get_feat_vects()
            if (settings.get("dataset") == "spiral"):
                all_vects = datasets.spiral.get_feat_vects()
            if (settings.get("dataset") == "flame"):
                all_vects = datasets.flame.get_feat_vects()
            feat_vects = random.sample(all_vects, settings.get("num_data"))
            k = settings.get("clusters")
            centers = random.sample(feat_vects, k)
            if (debug_flag == True):
                print "Generating partition matrix..."
            #Randomly generating a partition matrix and calculating its L2 norm.
            partition_matrix_before = algorithms.fuzzy_c_means.get_partition_matrix(feat_vects, k)
            dist_before = numpy.linalg.norm(partition_matrix_before)
            if (debug_flag == True):
                print "Updating partition matrix..."
            #Updating the partition matrix using the centers and calculating its L2 norm.
            partition_matrix_after = algorithms.fuzzy_c_means.update_partition_matrix(feat_vects, partition_matrix_before, k, feat_num, centers, settings.get("m"))
            dist_after = numpy.linalg.norm(partition_matrix_after)
            #Calculating the convergence using the L2 norms.
            convergence = abs(dist_after - dist_before)
            if (debug_flag == True):
                print "Convergence (threshold of " + str(settings.get("convergence_threshold")) + "):", convergence
            counter = 1
            #Executes until the algorithm converges.
            while (convergence > settings.get("convergence_threshold")):
                if (debug_flag == True):
                    print "\nIteration " + str(counter) + "."
                    print "Updating partition matrix..."
                partition_matrix_before = partition_matrix_after
                dist_before = dist_after
                if (debug_flag == True):
                    print "Calcuating centers..."
                #Centers are calculated and used to update the partition matrix and L2 norm.
                centers = algorithms.fuzzy_c_means.calculate_centroids(feat_vects, partition_matrix_before, k, feat_num, settings.get("m"))
                partition_matrix_after = algorithms.fuzzy_c_means.update_partition_matrix(feat_vects, partition_matrix_before, k, feat_num, centers, settings.get("m"))
                dist_after = numpy.linalg.norm(partition_matrix_after)
                #Convergence is calculated and checked against the threshold.
                convergence = abs(dist_after - dist_before)
                if (debug_flag == True):
                    print "Convergence (threshold of " + str(settings.get("convergence_threshold")) + "):", convergence
                counter += 1
            if (debug_flag == True):    
                print "\nAlgorithm has converged on iteration " + str(counter) + "."
                print "Counting cluster items..."
            #Cluster members are counted after convergence.
            count = algorithms.fuzzy_c_means.count_cluster_items(partition_matrix_after, k)
            #Data is labelled with cluster information.
            labelled_data = algorithms.fuzzy_c_means.label_data(feat_vects, partition_matrix_after, k)
            if (debug_flag == True):
                print "Calculating fitness..."
            #Cluster fitness is calculated (evaluates how good the clusters are).
            if (settings.get("fitness_func") == "dbi"):
                dbi = stats.calc_dbi(labelled_data, centers, k, label_pos, feat_num)
                print "\nCluster fitness (lower is better):", dbi
            if (settings.get("fitness_func") == "silhouette"):
                silhouette = stats.calc_silhouette(labelled_data, k, label_pos, feat_num)
                print "\nCluster fitness (closer to one is better):", silhouette
            print "\nCenters:"
            for i in xrange(0, k):
                print centers[i]
            #When the program has finished, stop the timer and display runtime.
            stop_time = timeit.default_timer()
            runtime_mins = int(math.trunc(((stop_time - start_time) / 60)))
            runtime_secs = int(round(((stop_time - start_time) % 60), 0))

            print "\n\nRuntime:", runtime_mins, "minute(s)", runtime_secs, "second(s)"

            if (settings.get("plot") == "on"):
                colours = ["#FF0000", "#0000FF", "#00FF00", "#00CCFF", "#FF33CC", "#FFFF00", "#000000", "#CCCCCC", "#522900", "#FFFFCC",  "#99CC00", "#FF9999", "#336699", "#006600", "#CCFFFF", "#CCCC00", "#FFCCFF", "#9900FF", "#006666", "#993366"]
                for i in xrange(0, k):
                    for j in xrange(0, len(labelled_data)):
                        if(labelled_data[j][label_pos] == i):
                            plt.scatter(labelled_data[j][0], labelled_data[j][1], s=10, color=colours[i])
                plt.show()
                
        if (settings.get("algorithm") == "spectralclustering"):
            print "\nNumber of clusters:", settings.get("clusters")
            print "Similarity Threshold:", settings.get("similarity_threshold")
            print "-------------------------------"
            #Intitialize the feature vectors.
            if (debug_flag == True):
                print "\nInitializing data vectors...\n"
            if (settings.get("dataset") == "iris"):
                all_vects = datasets.iris.get_feat_vects()
            if (settings.get("dataset") == "wisconsin"):
                all_vects = datasets.wisconsin.get_feat_vects()
            if (settings.get("dataset") == "s1"):
                all_vects = datasets.s1.get_feat_vects()
            if (settings.get("dataset") == "dim3"):
                all_vects = datasets.dim3.get_feat_vects()
            if (settings.get("dataset") == "spiral"):
                all_vects = datasets.spiral.get_feat_vects()
            if (settings.get("dataset") == "flame"):
                all_vects = datasets.flame.get_feat_vects()
            feat_vects = random.sample(all_vects, settings.get("num_data"))
            label_pos = settings.get("num_data")
            feat_num = settings.get("num_data")
            #Initializing the required matrices for spectral clustering.
            dist_matrix = algorithms.spectral_clustering.get_distance_matrix(feat_vects)
            adj_matrix = algorithms.spectral_clustering.get_adjacency_matrix(dist_matrix, settings.get("similarity_threshold"))
            deg_matrix = algorithms.spectral_clustering.get_degree_matrix_simple(adj_matrix)
            lap_matrix = algorithms.spectral_clustering.get_unnormalized_laplacian_matrix(deg_matrix, adj_matrix)
            eigenvalues, eigenvectors = numpy.linalg.eig(lap_matrix)
            clust_matrix = algorithms.spectral_clustering.get_cluster_matrix(eigenvectors)
            k = settings.get("clusters")
            centers = random.sample(clust_matrix, k)
            if (debug_flag == True):
                print "Clustering data..."
            #Clustering the data using the decoded centers.
            clustered_data = algorithms.k_means.cluster_data(centers, clust_matrix, True, feat_num)
            if (debug_flag == True):
                print "Counting cluster members..."
            #Counting the cluster members.
            count = algorithms.k_means.count_cluster_items(clustered_data, k, label_pos)
            converge_flag = False
            counter = 1
            #This code repeatedly executes until the algorithm converges (i.e cluster member count remains stable between iterations).
            while not converge_flag:
                if (debug_flag == True):
                    print "\nIteration" + str(counter) + "."
                old_count = count
                if (debug_flag == True):
                    print "\nCalculating new cluster centers..."
                #Calculating the new centers.
                centers = algorithms.k_means.calculate_centroids(clustered_data, k, label_pos, feat_num)
                if (debug_flag == True):    
                    print "Clustering data..."
                #Clustering the data.
                clustered_data = algorithms.k_means.cluster_data(centers, clust_matrix, False, feat_num)
                if (debug_flag == True):
                    print "Counting cluster members..."
                #Counting the cluster members.
                count = algorithms.k_means.count_cluster_items(clustered_data, k, label_pos)
                if (debug_flag == True):
                    print count
                #Check if the algorithm has converged. While loop terminates if it has.
                converge_flag = algorithms.k_means.compare_counts(old_count, count, k)
                counter += 1    

                if (debug_flag == True):
                    print "\nCount has stabilized, clusters have converged to an optima."
                    print  "Evaluating clusters..."
                #The cluster fitness is calculated and saved.
                if (settings.get("fitness_func") == "dbi"):
                    dbi = stats.calc_dbi(clust_matrix, centers, k, label_pos, label_pos)                            
                    print "\nCluster fitness (lower is better):", dbi
                if (settings.get("fitness_func") == "silhouette"):
                    silhouette = stats.calc_silhouette(clust_matrix, k, label_pos, feat_num)   
                    print "\nCluster fitness (closer to one is better):", silhouette
                if (debug_flag == True):
                    #Printing a line for spacing issues.
                    print       
            print "\nCenters:"
            for i in xrange(0, k):
                print centers[i]
            #When the program has finished, stop the timer and display runtime.
            stop_time = timeit.default_timer()
            runtime_mins = int(math.trunc(((stop_time - start_time) / 60)))
            runtime_secs = int(round(((stop_time - start_time) % 60), 0))

            print "\n\nRuntime:", runtime_mins, "minute(s)", runtime_secs, "second(s)"   
               
            if (settings.get("plot") == "on"):
                print "\nPlotting data."
                colours = ["#FF0000", "#0000FF", "#00FF00", "#00CCFF", "#FF33CC", "#FFFF00", "#000000", "#CCCCCC", "#522900", "#FFFFCC",  "#99CC00", "#FF9999", "#336699", "#006600", "#CCFFFF", "#CCCC00", "#FFCCFF", "#9900FF", "#006666", "#993366"]
                for i in xrange(0, k):
                    for j in xrange(0, len(clustered_data)):
                        if(clustered_data[j][label_pos] == i):
                            plt.scatter(clustered_data[j][0], clustered_data[j][1], s=10, color=colours[i])
                plt.show()
            

