import numpy as np


#Functions for dealing with IMMET's meteorological datadataset (monthly average from the last five years).
#NAO USAR POR ENQUANTO, DADOS NAO TRATADOS
class last5INMET():
    # Function to generate vectors from the data. (also used to return length of dataset).
    @staticmethod
    def get_feat_vects():
        feat_vects = []
        # Opening the data file
        with open("data/last5-INMET.txt") as feat_file:
            for line in feat_file:
                x = []
                # Initiating a counter to count up to the label.
                count = 1
                # Splitting feature by comma and line.
                for w in line.strip().split(','):
                    # If the counter is less than the label index (feature 5).
                    if (count < 15):
                        # Converting string representations into floats
                        x.append(float(w))
                    # When it reaches the label, ignore it.
                    else:
                        continue
                    count += 1
                feat_vects.append(x)
        return feat_vects

#Functions for dealing with IMMET's meteorological datadataset (from the last five years).
#NAO USAR POR ENQUANTO, DADOS NAO TRATADOS
class last5HINMET():
    # Function to generate vectors from the data. (also used to return length of dataset).
    @staticmethod
    def get_feat_vects():
        feat_vects = []
        # Opening the data file
        with open("data/last5-HOURS-INMET.txt") as feat_file:
            for line in feat_file:
                x = []
                # Initiating a counter to count up to the label.
                count = 1
                # Splitting feature by comma and line.
                for w in line.strip().split(','):
                    # If the counter is less than the label index (feature 5).
                    if (count < 8):
                        # Converting string representations into floats
                        x.append(float(w))
                    # When it reaches the label, ignore it.
                    else:
                        continue
                    count += 1
                feat_vects.append(x)
        return feat_vects


class Simepar:

    def __init__(self, file_path='../../weather_data_simepar_csv/simepar_curitiba_2015.csv'):
        self.file_path = file_path
        self.data_start_index = 2
        self.columns = [
            'station',
            'date',
            'pressaomedia',
            'tempmaxima',
            'tempminima',
            'tempmedia',
            'radsolarmedia',
            'umidademedia',
            'velventomedio',
            'dirventomedio',
            'velrajadamedia',
            'precipitacao',
            'rajadamaxima'
        ]
        self.used_columns = [
            # 'pressaomedia',
            # 'tempmaxima',
            # 'tempminima',
            'tempmedia',
            'radsolarmedia',
            'umidademedia',
            # 'velventomedio',
            # 'dirventomedio',
            # 'velrajadamedia',
            'precipitacao',
            # 'rajadamaxima'
        ]
        self.used_columns_idx = []
        for idx, column in enumerate(self.columns):

            if column in self.used_columns:
                self.used_columns_idx.append(idx)
        print(self.used_columns)
        print(self.used_columns_idx)


    def get_feat_vects(self):
        feat_vects = []
        # Opening the data file
        with open(self.file_path) as feat_file:
            for line in feat_file:
                splited_line = np.asarray(line.strip().split(','))
                x = []
                for c in self.used_columns_idx:
                    if c >= self.data_start_index:
                        x.append(float(splited_line[c]))
                    else:
                        x.append(splited_line[c])
                feat_vects.append(x)
        return feat_vects


#Functions for dealing with Fisher's Iris dataset.
class iris():
        
    #Function to generate vectors from the data. (also used to return length of dataset).
    @staticmethod
    def get_feat_vects():
        feat_vects = []
        #Opening the data file
        with open("data/iris.data.txt") as feat_file:
            for line in feat_file:
                x = []
                #Initiating a counter to count up to the label.
                count = 1
                #Splitting feature by comma and line.
                for w in line.strip().split(','):
                    #If the counter is less than the label index (feature 5).
                    if(count < 5):
                        #Converting string representations into floats
                        x.append(float(w))
                    #When it reaches the label, ignore it.
                    else:
                        continue
                    count += 1
                feat_vects.append(x)
        return feat_vects

    #Function to generate a corresponding label array from the data.
    @staticmethod
    def get_label_vects():
        label_vects = []
        #Opening the data file
        with open("data/iris.data.txt") as feat_file:
            for line in feat_file:
                #Initiating a counter to count up to the label.
                count = 1
                #Splitting feature by comma and line.
                for w in line.strip().split(','):
                    #When the counter reaches the label index (feature 5), append it.
                    if(count >= 5):
                        label_vects.append(w)
                    count += 1
        return label_vects

#Functions for dealing with the Wisconsin Breast Cancer dataset.
class wisconsin():
    
    #Function to generate vectors from the data. (also used to return length of dataset).
    @staticmethod
    def get_feat_vects():
        feat_vects = []
        #Opening the data file
        with open("data/wisconsin.txt") as feat_file:
            for line in feat_file:
                x = []
                #Initiating a counter to count up to the label.
                count = 1
                #Splitting feature by comma and line.
                for w in line.strip().split(','):
                    #Skip over the patient ID stored in the first element and the class label in the eleventh element, which are unecessary.
                    if((count != 1) and (count != 11)):
                        #If the feature is missing, append the median value of 5.
                        if w == '?':
                            w = 5
                        x.append(float(w))
                    count += 1
                feat_vects.append(x)
        return feat_vects
            
#Functions for dealing with the s1 dataset.
class s1():
    #Function to generate vectors from the data. (also used to return length of dataset).
    @staticmethod
    def get_feat_vects():
        feat_vects = []
        #Opening the data file
        with open("data/s1.txt") as feat_file:
            for line in feat_file:
                x = []
                #Splitting feature by spaces and lines.
                for w in line.strip().split('    '):
                    x.append(float(w))
                feat_vects.append(x)
        return feat_vects
    
#Functions for dealing with the dim3 dataset.
class dim3():
    #Function to generate vectors from the data. (also used to return length of dataset).
    @staticmethod
    def get_feat_vects():
        feat_vects = []
        #Opening the data file
        with open("data/dim3.txt") as feat_file:
            for line in feat_file:
                x = []
                #Splitting feature by spaces and lines.
                for w in line.strip().split('    '):
                    x.append(float(w))
                feat_vects.append(x)
        return feat_vects
    
#Functions for dealing with the spiral dataset.
class spiral():
    #Function to generate vectors from the data. (also used to return length of dataset).
    @staticmethod
    def get_feat_vects():
        feat_vects = []
        #Opening the data file
        with open("data/spiral.txt") as feat_file:
            for line in feat_file:
                count = 1
                x = []
                #Splitting feature by spaces and lines.
                for w in line.strip().split(' '):
                    x.append(float(w))
                feat_vects.append(x)
        return feat_vects

#Functions for dealing with the flame dataset.
class flame():
    #Function to generate vectors from the data. (also used to return length of dataset).
    @staticmethod
    def get_feat_vects():
        feat_vects = []
        #Opening the data file
        with open("data/flame.txt") as feat_file:
            for line in feat_file:
                count = 1
                x = []
                #Splitting feature by spaces and lines.
                for w in line.strip().split(' '):
                    x.append(float(w))
                feat_vects.append(x)
        return feat_vects