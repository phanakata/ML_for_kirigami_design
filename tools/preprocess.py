import numpy as np 
from helper_functions import * 
def preprocess_main():
    """ 
        this function takes a pre-made list of email texts (by default word_data.pkl)
        and the corresponding authors (by default email_authors.pkl) and performs
        a number of preprocessing steps:
            -- splits into training/testing sets (10% testing)
            -- vectorizes into tfidf matrix
            -- selects/keeps most helpful features

        after this, the feaures and labels are put into numpy arrays, which play nice with sklearn functions

        4 objects are returned:
            -- training/testing features
            -- training/testing labels

    """




def load_and_tabulate_data(filename, listBinary):
    """
    Parameters
    -----------------
    filename: strings
        name of the file to be loaded
    listBinary: list 
        list of genomes
        
    Return
    ------------------
    data: numpy ndarray(nsamples, nfeatures+nobservations)
    """
    rawData=np.loadtxt(filename)        
    #first make copies of equivalent configurations
    s = np.zeros((4*len(rawData), 4))
    for i in range(len(s)):
        s[i] = rawData[i//4]
        s[i][0]=i #use normal genome
    
    nfeatures = len(listBinary[0]) #length of 2D arrays flatten to 1D
    data = np.zeros((len(s), nfeatures+3)) # '+3' as we include 3 properties 
    
    for i in range(len(data)):
        array = toArray(listBinary[int(s[i][0])])
        data[i][0:nfeatures] = array
        data[i][nfeatures:] = s[i][1:]
    
    return data


def load_and_tabulate_data_unique(filename, listBinary):
    """
    Parameters
    -----------------
    filename: strings
        name of the file to be loaded
    listBinary: list 
        list of genomes
        
    Return
    ------------------
    data: numpy ndarray(nsamples, nfeatures+nobservations)
    """
    rawData=np.loadtxt(filename)        
    #first make copies of equivalent configurations
    s = np.zeros((len(rawData), 4))
    for i in range(len(s)):
        s[i] = rawData[i]
        s[i][0]=i*4 #use normal genome, in normal genome there are 4 duplicates! 
    
    nfeatures = len(listBinary[0]) #length of 2D arrays flatten to 1D
    data = np.zeros((len(s), nfeatures+3)) # '+3' as we include 3 properties 
    
    for i in range(len(data)):
        array = toArray(listBinary[int(s[i][0])])
        data[i][0:nfeatures] = array
        data[i][nfeatures:] = s[i][1:]
    
    return data



def create_matrix(data, discrete, prop, cutoff, nfeatures):
    """
    Create matrix X and 1D array y for data analysis 
    
    Parameters
    -----------------
    data: numpy array
        data  
    discrete: boolean
        If TRUE set y=1 for 'good' design and y=0 for 'bad' design
    prop: int 
        property to study
    cutoff: float 
        Set cutoff (e.g. fracstrain) to distinguish 'good' and 'bad' designs 
        fracture for pristine 0.25632206
    nfeatures: int 
        number of features (length of 2D grid that flatten into 1D array)
        
    Return
    -----------
    x: matrix 
        matrix X (nsamples, nfeatures)
    
    y: 1D array
        y values (nsamples,)
       
    """
    y = np.zeros(len(data))
  
    count = 0 
    for i in range (len(data)):
        if data[i][nfeatures+prop]>cutoff:
            y[i]=1
            count += 1
        else:
            y[i]=0
            
        if discrete==False:
            y[i]=data[i][nfeatures+prop]
    
    x = data[:, 0:nfeatures]
            
            
    print ("Number of good designs "+str(count)+" out of total "+str(len(y)))
    return x, y 


def split_data(x, y, percentage_training, percentage_test):
    """
    Parameters
    ----------------------------
    x: numpy matrix 
    y: numpy array
    percentage_test: float 
        percentage of data to be set for test set
    
    Return
    -------------------
    X_train, X_valid, X_test, y_train, y_valid, y_test
    
    """
    percentage_valid = 1- percentage_training - percentage_test
    ntrain = int(percentage_training * len(x))
    nvalid = int(percentage_valid * len(x))
    ntest = int(percentage_test * len(x))

    X_train = x[:ntrain]
    X_valid = x[ntrain:ntrain:ntrain+nvalid]
    X_test = x[ntrain+nvalid:]
    
    y_train = y[:ntrain]
    y_valid = y[ntrain:ntrain:ntrain+nvalid]
    y_test = y[ntrain+nvalid:]
    
    return X_train, X_valid, X_test, y_train, y_valid, y_test