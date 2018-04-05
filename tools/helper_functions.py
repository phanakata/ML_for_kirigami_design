import numpy as np

#need this function to convert strings to array 
def toArray(stringsBinary):
    size = len(stringsBinary)
    array = np.zeros(size)
    
    for i in range(size):
        array[i] = int(stringsBinary[i])
        
    return array  


def toString(cutConfigurations):
    genome_ID=""
    for i in range(len(cutConfigurations)):
        genome_ID=genome_ID+str(cutConfigurations[i])

    return genome_ID