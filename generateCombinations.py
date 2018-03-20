import numpy as np

#def generateCombination(NCcell_x, NCcell_y, listBinary, numberNoCuts, detach):

#def generateCombinations():
"""
This codes generate all possible configurations 
NCell_x, y: integer
listBinary: list
numberNoCuts: integer
"""
#generate all possible 2^(NCcell_x * NCcell_y)
def generate_binary(n, listBinary):
    if n == 0:
        return listBinary
    else:
        if len(listBinary) == 0:
            return generate_binary(n-1, ["0", "1"])
        else:
            return generate_binary(n-1, [i + "0" for i in listBinary] + [i + "1" for i in listBinary])
    
    
#def find combinations 
def findCombinations(listBinary, NCcell_x, NCcell_y, numberNoCuts):
    N=NCcell_x * NCcell_y
    combinations = []
    for  i in range(len(listBinary)):
        sum=0
        for j in range(N):
            sum = sum +  int(listBinary[i][j])
            
        if sum ==numberNoCuts:
            combinations.append(listBinary[i])

    return combinations


def findCombinations2(listBinary, NCcell_x, NCcell_y, numberNoCuts):
    N = NCcell_x * NCcell_y
    combinations = []
    for  i in range(len(listBinary)):
        sumx =[0]*NCcell_x
        for jy in range(NCcell_y):
            for jx in range(NCcell_x):
                sumx[jx] = sumx[jx] +  int(listBinary[i][jx*NCcell_y+jy])
                
        sumtotal=0
        for jx in range(NCcell_x):
            sumtotal=sumx[jx] +sumtotal
        
        if sumtotal==numberNoCuts:
            include = True
            #check if it's not detached 
            for jx in range (NCcell_x):
                if sumx[jx]==0:
                    include= False
            if include is True:
                combinations.append(listBinary[i])
            #print (sumx[0])

    return combinations
    
