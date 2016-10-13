import random

#Function to initialize the chromosomal population.
def init_ga(pop_size, data, k, debug):
    data_len = len(data)
    #If the population size is not even, program quits.
    if (pop_size % 2 != 0):
        print "Chromosomal population must be an even number."
        quit()
    chromosomes = []
    #Chromosomal length is three times the user provided k value.
    chrom_len = 3 * k

    #Iterating over the population.
    for i in xrange(0, pop_size):
        temp_chrom = []
        #Iterating over each base on the chromosome.
        for j in xrange(0, chrom_len):
            #Appending a 'don't care' symbol.
            temp_chrom.append('#')
        #Addding the chromosome to the array.
        chromosomes.append(temp_chrom)
    #Iterating over the population.
    for i in xrange(0, pop_size):
        #Generating a random number of clusters for the chromosome to encode (minimum of 2).
        k_rand = random.randint(2, k)
        #Iterating over each potential cluster.
        for j in xrange(0, k_rand):
            #Generating a random data point.
            rand_data_point = random.randint(0, (data_len - 1))
            #Saving the data point as a center.
            center = data[rand_data_point]
            #Generating a random location on the chromosome and putting the center there.
            rand_chrom_loc = random.randint(0, (chrom_len - 1))
            chromosomes[i][rand_chrom_loc] = center
    #Iterating over the population.
    for i in xrange(0, pop_size):
        k_temp = len(get_center(chromosomes[i]))
        #If the number of clusters is below 2, the chromosome is reinitialized.
        if (k_temp <= 1):
            chromosomes[i] = reinit_chrom(data, k, chromosomes, i, debug)
    return chromosomes
    
#Function to reinitialize a chromosome.
def reinit_chrom(data, k, chromosomes, position, debug):
    if (debug == True):
        print "Re-initializing chromosome", position + 1, "..."
    data_len = len(data)
    new_chrom = []
    chrom_len = 3 * k
    #Creating a new chromosome.
    for i in xrange(0, chrom_len):
        new_chrom.append('#')
    #Creating a random number of clusters for the chromosome to encode.
    k_rand = random.randint(2, k)
    #iterating over the clusters.
    for i in xrange(0, k_rand):
        #Generating a random datapoint and a location to insert the datapoint.
        rand_data_point = random.randint(0, (data_len - 1))
        center = data[rand_data_point]
        rand_chrom_loc = random.randint(0, (chrom_len - 1))
        new_chrom[rand_chrom_loc] = center
    #Reinserting the repaired chromosome into the population.
    chromosomes[position] = new_chrom
    return chromosomes
    
#Function to decode the centers from a chromosome.
def get_center(chromosome):
    chrom_len = len(chromosome)
    centers = []
    #Iterating over the chromosome.
    for i in xrange(0, chrom_len):
            #If a 'don't care' symbol is encountered, ignroe it.
            if (chromosome[i] == '#'):
                continue
            #If a center is encountered, append it.
            else:
                centers.append(chromosome[i])
    return centers

#Function to order the chromosome by fitness.
def order_chroms(chromosomes, fitnesses, function, debug):
    if (debug == True):
        print "\nOrdering chromosomes by fitness..."
    pop_size = len(chromosomes)
    ordered_chroms = []
    #Iterating over each chromosome.
    for i in xrange(0, pop_size):
        #Finding the best fitness value based on the function required.
        if (function == "minimise"):
            best = min(fitnesses)
        if (function == "maximise"):
            best = max(fitnesses)
        #Finding the index of the chromosome corresponding to the best fitness value.
        best_ind = fitnesses.index(best)
        #Appending the chromosome to the ordered array.
        ordered_chroms.append(chromosomes[best_ind])
        #Removing the fitness and chromosome from the old array.
        fitnesses.pop(best_ind)
        chromosomes.pop(best_ind)
    return ordered_chroms
    
#Function to crossover the chromosomes.
def crossover(chromosomes, num_cuts, debug):
    pop_size = len(chromosomes)
    if (debug == True):
        print "\nStarting crossover..."
        
        print "\nSaving best 2 chromosomes..."
    children = []
    #Saving the first 2 chromosomes as elites.
    #for i in xrange(0, 2):
    #    chromosomes.pop(-1)
    #    children.append(chromosomes[i])
    #Iterating over the population in pairs.
    for i in xrange(0, (pop_size / 2)):
        #Selecting two parents
        parent_a = chromosomes[(2 * i) - 2]
        parent_b = chromosomes[(2 * i) - 1]
        #Breeding to produce offspring.
        offspring = breed(parent_a, parent_b, num_cuts, debug)
        #Retrieving offspring from the dictionary returned.
        child_a = offspring.get("child_a")
        child_b = offspring.get("child_b")
        #Appending into the children array.
        children.append(child_a)
        children.append(child_b)
    if(debug):
        print
        for i in xrange(0, pop_size):
            print children[i]
    return children

#Function to breed two parental chromosomes to produce two child chromosomes.
def breed(parent_a, parent_b, num_cuts, debug):
    if (debug == True):
        print "\nBreeding..."
    chrom_len = len(parent_a)
    for i in xrange(0, num_cuts):
        #Randomly generating a cut point.
        cut_point = random.randint(1, (chrom_len - 1))
        #Assembling the children based on the parents and the cut point.
        child_a = list(parent_a[0:cut_point]) + list(parent_b[cut_point:])
        child_b = list(parent_b[0:cut_point]) + list(parent_a[cut_point:])
        #Setting the children as parents for the next iteration (in the case of more than one cut point).
        parent_a = child_a
        parent_b = child_b
    #Returning the children as a dictionary.
    return {"child_a":child_a, "child_b":child_b}

#Function to mutate chromosomes.
def mutation(threshold, chromosomes, data, data_len, max_k, debug):
    if (debug == True):
        print "\nPerforming mutation (threshold of " + str(threshold) + ")..."
    pop_size = len(chromosomes)
    chrom_len = len(chromosomes[0])
    #Iterating over the population
    for i in xrange(0, pop_size):
        if (debug == True):
            print "\nChromosome " + str(i + 1) + ":"
        mut_flag = False
        #Iterating over each element in the chromosome
        for j in xrange(0, chrom_len):
            #Generating a random number between 0 and 1000.
            mutation = random.randint(0, 1000)
            #If the number is greater than the threshold;
            if (mutation >= threshold):
                mut_flag = True
                before = chromosomes[i][j][:]
                #If the element is a don't care, mutate to a center.
                if (chromosomes[i][j] == '#'):
                    if (len(get_center(chromosomes[i])) < max_k):
                        rand_data_point = random.randint(0, (data_len - 1))
                        chromosomes[i][j] = data[rand_data_point]
                        if (debug == True):
                            print "Element " + str(j + 1) + " has mutated from '" + str(before) + "' to '" + str(chromosomes[i][j]) + "'."
                #If the element is a center, multiply the center elements by a random number between 0 and 1 added to 0.6 
                else:
                    for k in xrange(0, data_len):
                        mutated = 0.6 + random.random()
                        chromosomes[i][j][k] = chromosomes[i][j][k] * mutated
                    if (debug == True):
                        print "Element " + str(j + 1) + " has mutated from '" + str(before) + "' to '" + str(chromosomes[i][j]) + "'."
        if (debug == True):                
            if(mut_flag == True):
                print chromosomes[i]
            else:
                print "No elements mutated."
    return chromosomes
                    

    