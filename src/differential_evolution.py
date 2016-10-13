import random

#Function to initialize the chromosomes.
def init_de(data, feat_num, max_k, pop_size, threshold):
    chromosomes = []
    #Chromosome length is calculated based on the maximum K-value given by the user and the number of features in each data-point.
    chrom_len = max_k + (max_k * feat_num)
    data_len = len(data)    
    #Iterating over the population of chromosomes.
    for i in xrange(0, pop_size):
        temp_chrom = []
        #For each of the first k elements
        for j in xrange(0, max_k):
            #A random number between 0 and 1 is generated to serve as the cluster activation value.
            temp_chrom.append(random.random())
        counter = 0
        #A set is initialized to store the indexes of the data-points.
        rand_set = set()
        #Iterating from the k-th element to the end of the chromosome
        for j in xrange(max_k, chrom_len):
            #If it's the first instance of the corresponding cluster.
            if (counter == 0):
                #A random index is selected
                rand = random.randint(0, (data_len - 1))
                #If the index is in the set already, this executes
                while rand in rand_set:
                    #Random indexes are continually generated until one is found that is not in the set.
                    rand = random.randint(0, (data_len - 1))
                #The data-point of that index is saved as a center.
                center = data[rand]
                #The data-point's index is added to the set.
                rand_set.add(rand)
            temp_chrom.append(center[counter])
            counter = counter + 1
            #When the number of features is reached, the counter is reset.
            if (counter == feat_num):
                counter = 0
        #The chromosome is appended into the array.
        chromosomes.append(temp_chrom)
    #Before return, the chromosomes are checked for validity.
    chromosomes = check_chroms(chromosomes, threshold, max_k)
    return chromosomes

#Function to check the chromosomes are valid.
def check_chroms(chromosomes, threshold, max_k):    
    pop_size = len(chromosomes)
    i = 0
    #Iterating over the chromosomes.
    while i < pop_size:
        active_clusters = 0
        #Checking that a minimum of 2 activation values are greater than the user-input threshold.
        for j in xrange(0, max_k):
            if(chromosomes[i][j] >= threshold):
                active_clusters += 1
        #If there are not 2 activation values greater than threshold;
        if (active_clusters < 2):
            #2 random thresholds are generated and inserted randomly into the activation section of the chromosome.
            for l in xrange(0, 2):
                rand_thresh = random.uniform(threshold, 1)
                rand_loc = random.randint(0, (max_k - 1))
                chromosomes[i][rand_loc] = rand_thresh
        else:
            i = i + 1
    return chromosomes

#Function to reinitialize an invalid chromosome.
def reinit_chrom(chromosome, data, feat_num, max_k, threshold, debug):
    if (debug == True):
        print "Invalid centroid detected, reinitializing chromosome..."
    chrom_len = len(chromosome)
    data_len = len(data)
    temp_chrom = []
    #Iterating over the activation section of the chromosome.
    for i in xrange(0, max_k):
        #Generating random activation values and inserting them into the chromosome.
        temp_chrom.append(random.random())
    active_clusters = 0
    #Checking that there are at least 2 active clusters.
    for i in xrange(0, max_k):
        if(chromosome[i] >= threshold):
                active_clusters += 1
    #If there are less than 2 active clusters;
    if (active_clusters < 2):
            #two active thresholds are generated and inserted randomly into the activation section of the chromosome.
            for l in xrange(0, 2):
                rand_thresh = random.uniform(threshold, 1)
                rand_loc = random.randint(0, (max_k - 1))
                chromosome[rand_loc] = rand_thresh
    counter = 0
    rand_set = set()
    #Iterating over the remaining chromosomal elements.
    for i in xrange(max_k, chrom_len):
        #If it is the first element;
        if (counter == 0):
            #A random index is generated until one is found that is not in the rand_set.
            rand = random.randint(0, (data_len - 1))
            while rand in rand_set:
                rand = random.randint(0, (data_len - 1))
            #The index is used to access the data and save the corresponding point as a center.
            center = data[rand]
            #The index is added to the set to prevent repeats.
            rand_set.add(rand)
        #The center is appended to the chromosome.
        temp_chrom.append(center[counter])
        counter = counter + 1
        if (counter == feat_num):
            #The counter is reset when the counter reached the number of features.
            counter = 0
    chromosome = temp_chrom
    return chromosome            
    
#Function to decode the centers from the chromosome.
def get_center(chromosomes, max_k, threshold, feat_num):
    centers = []
    #Iterating over the activation section of the chromosome.
    for i in xrange(0, max_k):
        temp_center = []
        #If the activation value is over the threshold;
        if chromosomes[i] >= threshold:
            #The first and last elements are calculated using the index of the activation value.
            first_element = max_k + (i * feat_num)
            last_element = first_element + feat_num
            #Iterating over the center corresponding to the activation index.
            for j in xrange(first_element, last_element):
                temp_center.append(chromosomes[j])
            #Appending the center to the centers array.
            centers.append(temp_center)
    return centers

#Function to order the chromosomes based on their fitness values.
def order_chroms(chromosomes, fitnesses, function, debug):
    if (debug == True):
        print "\nOrdering chromosomes by fitness..."
    pop_size = len(chromosomes)
    ordered_chroms = []
    #Iterating over each chromosome.
    for i in xrange(0, pop_size):
        #Choosing the maximal or minimal fitness value based on what function is being optimized.
        if (function == "minimise"):
            best = min(fitnesses)
        if (function == "maximise"):
            best = max(fitnesses)
        #Finding the index of the best fitness
        best_ind = fitnesses.index(best)
        #Finding the best chromosome using the index and appending it in the ordered array.
        ordered_chroms.append(chromosomes[best_ind])
        #Removing the best fitness and corresponding chromosome from their arrays.
        fitnesses.pop(best_ind)
        chromosomes.pop(best_ind)
    return ordered_chroms

#Function to perform crossover on the chromosomes.
def crossover(chromosomes, num_cuts, debug):
    pop_size = len(chromosomes)
    #Printing the chromosomes.
    if (debug == True):
        print "\nStarting crossover..."
        print "\nSaving best 2 chromosomes..."
    children = []
    #Saving the best 2 chromosomes by appending them straight into the children array.
    for i in xrange(0, 2):
        chromosomes.pop(-1)
        children.append(chromosomes[i])
    #Iterating over the chromosomes in pairs.
    for i in xrange(1, (pop_size / 2)):
        #Choosing the parents
        parent_a = chromosomes[(2 * i) - 2]
        parent_b = chromosomes[(2 * i) - 1]
        #Calling the breed function to crossover the parents.
        offspring = breed(parent_a, parent_b, num_cuts, debug)
        #Retrieving the children
        child_a = offspring.get("child_a")
        child_b = offspring.get("child_b")
        #Appending the children in the array.
        children.append(child_a)
        children.append(child_b)
    if(debug):
        print
        for i in xrange(0, pop_size):
            print chromosomes[i]
    return children

#Function to cross two parental chromosomes.
def breed(parent_a, parent_b, num_cuts, debug):
    if (debug == True):
        print "\nBreeding..."
    chrom_len = len(parent_a)
    for i in xrange(0, num_cuts):
        #Randomly generating a point on the chromosome to cut.
        cut_point = random.randint(1, (chrom_len - 1))
        #Arranging the children based on the cut point.
        child_a = list(parent_a[0:cut_point]) + list(parent_b[cut_point:])
        child_b = list(parent_b[0:cut_point]) + list(parent_a[cut_point:])
        #Setting the children as parents for the next iteration of the loop (if more than one cut is to be performed).
        parent_a = child_a
        parent_b = child_b
    #Returning the children in a dictionary.
    return {"child_a":child_a, "child_b":child_b}

#Function to perform mutation on the chromosomes.
def mutation(threshold, chromosomes, max_k, mut_thresh, debug): 
    if (debug == True):   
        print "\nPerforming mutation (threshold of " + str(mut_thresh) + ")..."
    pop_size = len(chromosomes)
    chrom_len = len(chromosomes[0])
    #Iterating over each chromosome.
    for i in xrange(0, pop_size):
        if (debug == True):
            print "\nChromosome " + str(i + 1) + ":"
        mut_flag = False
        #Iterating over the activation section.
        for j in xrange(0, max_k):
            #Generating a random number between 0 and 1000.
            mutation = random.randint(0, 1000)
            #Check if the number is over the user-input mutation threshold.
            if (mutation >= mut_thresh):
                mut_flag = True
                active_flag = False
                inactive_flag = False
                before = chromosomes[i][j]
                #If the activation value is over the threshold, mutate to under the threshold.
                if (chromosomes[i][j] >= threshold):
                    chromosomes[i][j] = 0.0
                    active_flag = True
                #If the activation value is under the threshold, mutate to over the threshold.
                else:
                    chromosomes[i][j] = 1.0
                    inactive_flag = True
                if active_flag:
                    if (debug == True):
                        print "Element " + str(j + 1) + " has mutated from active to inactive."
                if inactive_flag:
                    if (debug == True):
                        print "Element " + str(j + 1) + " has mutated from inactive to active."
        #Iterating over the remaining chromosomal elements.
        for j in xrange(max_k, chrom_len):
            before = chromosomes[i][j]
            #Generating a random number between 0 and 1000.
            mutation = random.randint(0, 1000)
            #If the number is over the user-input mutation threshold;
            if (mutation >= mut_thresh):
                #Multiply the current center values by a random number between 0 and 1, and then multiply by 0.6
                chromosomes[i][j] = round(chromosomes[i][j] * (0.6 + random.random()), 1)
                mut_flag = True
                if (debug == True):
                    print "Element " + str(j + 1) + " has mutated from '" + str(before) + "' to '" + str(chromosomes[i][j]) + "'."
        if (debug == True):
            if(mut_flag == True):
                print chromosomes[i]
            else:
                print "No elements mutated."
        #Before returning, check the chromosomes for validity.
        chromosomes[i] = check_chromosome(chromosomes[i], threshold, max_k, debug)
    return chromosomes       
    
#Function to check a single chromosome for validity.
def check_chromosome(chromosome, threshold, max_k, debug):
    k_arr = []
    inv_flag = False
    #iterating over the activation section of the chromosome.
    for i in xrange(0, max_k):
        #Checking that at least 2 activation values are over the threshold.
        if (chromosome[i] >= threshold):
            k_arr.append(1)
    rand_set = set()
    #If less than 2 activation values are over the threshold;
    if (len(k_arr) < 2):
        inv_flag = True
        if (debug == True):
            print "Chromosome is invalid, re-initializing thresholds..."
        #Randomly set 2 activation values to over the threshold
        for i in xrange(0, 2):
            rand = random.randint(0, (max_k - 1))
            while rand in rand_set:
                rand = random.randint(0, (max_k - 1))
            chromosome[rand] = 1.0
            rand_set.add(rand)
    if inv_flag:
        if (debug == True):
            print chromosome
    return chromosome