#Functions for parsing settings.ini
import re
import ConfigParser
import sys

def parse_settings_file():
    
    settings = ConfigParser.ConfigParser()
    settings.read("config.ini")
    
    #Reading main settings from settings.ini.
    function = str.lower(re.sub('[^A-Za-z]+', '', settings.get('Main Settings', 'Function')))
    algorithm = str.lower(re.sub('[^A-Za-z]+', '', settings.get('Main Settings', 'Algorithm')))
    dataset = str.lower(re.sub('[^A-Za-z0-9]+', '', settings.get('Main Settings', 'Dataset')))
    num_data = int(re.sub('[^0-9]+', '', settings.get('Main Settings', 'Number of Data Points')))
    debug = str.lower(re.sub('[^A-Za-z]+', '', settings.get('Main Settings', 'Debug')))
    fitness_func = str.lower(re.sub('[^A-Za-z]+', '', settings.get('Main Settings', 'Fitness')))
    plot = str.lower(re.sub('[^A-Za-z]+', '', settings.get('Main Settings', 'Plot')))
    
    #Reading function settings from ini.
    try:
        pop_size = int(re.sub('[^0-9]+', '', settings.get('Function Settings', 'Population Size')))
    except ValueError:
        print "\nPopulation size must be an integer."
        sys.exit()
    try:
        max_iter = int(re.sub('[^0-9]+', '', settings.get('Function Settings', 'Number Of Generations')))
    except ValueError:
        print "\nNumber of generations must be an integer."
        sys.exit()
    
    #Reading algorithm settings from ini.
    try:
        clusters = int(re.sub('[^0-9]+', '', settings.get('Algorithm Settings', 'Clusters')))
    except ValueError:
        print "\nNumber of clusters must be an integer."
        sys.exit()
    try:
        max_k = int(re.sub('[^0-9]+', '', settings.get('Algorithm Settings', 'Maximum Clusters')))
    except ValueError:
        print "\nMaximum clusters must be an integer."
        sys.exit()
    try:
        mutation_threshold = int(re.sub('[^0-9]+', '', settings.get('Algorithm Settings', 'Mutation Threshold'))) 
    except ValueError:
        print "\nMutation threshold must be an integer between 0 and 1000."
        sys.exit()
    try:
        activation_threshold = float(re.sub('[^0-9.]+', '', settings.get('Algorithm Settings', 'Activation Threshold')))
    except ValueError:
        print "\nActivation threshold must be a decimal number between 0 and 1."
        sys.exit()
    try:
        convergence_threshold = float(re.sub('[^0-9.]+', '', settings.get('Algorithm Settings', 'Convergence Threshold')))
    except ValueError:
        print "\nCovnergence threshold must be a number."
        sys.exit()
    try:
        similarity_threshold = float(re.sub('[^0-9.]+', '', settings.get('Algorithm Settings', 'Similarity Threshold')))
    except ValueError:
        print "\nSimilarity threshold must be a number."
        sys.exit()
    try:
        omega_min = float(re.sub('[^0-9.]+', '', settings.get('Algorithm Settings', 'Omega Min')))
        omega_max =float(re.sub('[^0-9.]+', '', settings.get('Algorithm Settings', 'Omega Max')))
    except ValueError:
        print "\nOmega values must be numbers."
        sys.exit()
    try:
        weight = float(re.sub('[^0-9.]+', '', settings.get('Algorithm Settings', 'Velocity Weight')))
    except ValueError:
        print "\nVelocity weight must be a number."
        sys.exit()
    try:
        m = float(re.sub('[^0-9.]+', '', settings.get('Algorithm Settings', 'Fuzziness')))
    except ValueError:
        print "\nFuzziness must be a number."
        sys.exit()
    
    return {"function":function, "algorithm":algorithm, "dataset":dataset, "num_data":num_data, "debug":debug, "fitness_func":fitness_func, "clusters":clusters, 
            "max_k":max_k, "pop_size":pop_size, "max_iter":max_iter, "mutation_threshold":mutation_threshold, "activation_threshold":activation_threshold,
            "convergence_threshold":convergence_threshold, "similarity_threshold":similarity_threshold, "omega_min":omega_min, "omega_max":omega_max,
            "weight":weight, "m":m, "plot":plot}

def get_print_info():
    
    settings = ConfigParser.ConfigParser()
    settings.read("config.ini")
    
    function = settings.get('Main Settings', 'Function')
    algorithm = settings.get('Main Settings', 'Algorithm')
    dataset = settings.get('Main Settings', 'Dataset')
    num_data = settings.get('Main Settings', 'Number of Data Points')
    fitness_func = settings.get('Main Settings', 'Fitness')
    
    return {"function":function, "algorithm":algorithm, "dataset":dataset, "num_data":num_data, "fitness_func":fitness_func}

def check_valid_settings(settings):
    
    #Defining allowed values.
    functions_arr = ["geneticalgorithm", "differentialevolution", "particleswarmoptimization", "none"]
    algorithms_arr = ["kmeans", "fuzzycmeans", "spectralclustering"]
    dataset_arr = ["iris", "wisconsin", "s1", "dim3", "spiral", "flame"]
    fitness_arr = ["dbi", "silhouette"]
    
    #Checking values against allowed values.
    if (settings.get("function") not in functions_arr):
        print "\nUnrecognised evolutionary function."
        print "Check the function field of settings.ini."
        sys.exit()
    if (settings.get("algorithm") not in algorithms_arr):
        print "\nUnrecognised algorithm."
        print "Check the algorithm field of settings.ini."
        sys.exit()
    if (settings.get("dataset") not in dataset_arr):
        print "\nUnrecognised dataset."
        print "Check the dataset field of settings.ini."
        sys.exit()
    if ((settings.get("debug") != "on") and (settings.get("debug") != "off")):
        print "\nUnrecognised debug input."
        print "Check the debug field of settings.ini."
        sys.exit()
    if (settings.get("fitness_func") not in fitness_arr):
        print "\nUnrecognised fitness function."
        print "Check the fitness function field of settings.ini."
        sys.exit()
    if (settings.get("plot") != "on") and (settings.get("plot") != "off"):
        print "\nUnrecognised plot input."
        print "Check the plot field of settings.ini."
        sys.exit()

