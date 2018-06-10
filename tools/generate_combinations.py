import numpy as np
from helper_functions import * 


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
def findCombinations_np(listBinary, NCcell_x, NCcell_y, numberNoCuts):
    N=NCcell_x * NCcell_y
    combinations = []
    for  i in range(len(listBinary)):
        if np.sum(toArray(listBinary[i]))==numberNoCuts:
              combinations.append(listBinary[i])

    return combinations


#def find combinations 
#Using list is faster than numpy array, prob because of toArray()
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


def findCombinations_wo_detached(listBinary, NCcell_x, NCcell_y, numberNoCuts):
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


#generate combinations and create its reflection right below it 
def findCombinations_rs_yxy(listBinary, NCcell_x, NCcell_y, numberNoCuts):
    """
    This method uses symmetry. 
    Create equivalent configuration by rs_yxy reflection y, x, and y. 
    All equivalent configurations are grouped ex 0123, 4567, etc 
    With this we only need to simulate 1/4 of total possible configurations 
    
    Parameters
    -----------------------
    
    
    
    """
    
    
    N = NCcell_x * NCcell_y
    combinations = []
    for  i in range(len(listBinary)):
        sumx =[0]*NCcell_x
        #assume we include the geometry 
        #first filltering check if it's detatched in any of jx 
        include = True
        for jx in range(NCcell_x):
            for jy in range(NCcell_y):
                sumx[jx] = sumx[jx] +  int(listBinary[i][jx*NCcell_y+jy])
            
            #after summing over jy check if it's detatched 
            
            if sumx[jx]==0:
                include=False
        
        
        #after filtering the 'detatched' 
        if include is True:
            sumtotal=0
            for jx in range(NCcell_x):
                sumtotal=sumx[jx] +sumtotal
            
            #2nd filter if density is correct 
            if sumtotal==numberNoCuts and listBinary[i] not in combinations:
                
                
                combinations.append(listBinary[i])
                
                
                #reflection around y 
                reftotal =""
                for nx in range (NCcell_x):
                    string1 = (listBinary[i][nx*NCcell_y:(nx+1)*NCcell_y])
                    #print ((string1))
                    ref =""
                    for k in range(len(string1)):
                        ref = str(string1[k]) + ref
                    
                    reftotal = reftotal + ref
                
                #if reftotal != listBinary[i]:
                #to make sure we don't create double configurations! 
                #for now we include duplicates of the 'identity' but make sure delete duplicates during analysis! 
                combinations.append(reftotal)
                
                #reflection around x
                reftotal2 =""
                for nx in range (NCcell_x):
                    string1 = (reftotal[nx*NCcell_y:(nx+1)*NCcell_y])
                    reftotal2 = string1 + reftotal2 
                 
                combinations.append(reftotal2)
                
                #reflection around y AGAIN!
                reftotal =""
                for nx in range (NCcell_x):
                    string1 = (reftotal2[nx*NCcell_y:(nx+1)*NCcell_y])
                    #print ((string1))
                    ref =""
                    for k in range(len(string1)):
                        ref = str(string1[k]) + ref
                    
                    reftotal = reftotal + ref
                 
                combinations.append(reftotal)
                 
                
    return combinations


    
    
    #generate combinations and create its reflection right below it 
def findCombinations_rs_yxy_np(listBinary, NCcell_x, NCcell_y, numberNoCuts):
    """
    This method uses symmetry. 
    Create equivalent configuration by rs_yxy reflection y, x, and y. 
    All equivalent configurations are grouped ex 0123, 4567, etc 
    With this we only need to simulate 1/4 of total possible configurations 
    
    Parameters
    -----------------------
    
    
    
    """
    
    
    N = NCcell_x * NCcell_y
    combinations = []
    for  i in range(len(listBinary)):
        sumx =np.zeros(NCcell_x)
        #assume we include the geometry 
        #first filltering check if it's detatched in any of jx 
        include = True
        for jx in range(NCcell_x):
            sumx[jx] = sumx[jx] +  np.sum(toArray(listBinary[i][jx*NCcell_y:(jx+1)*NCcell_y]))
            
            #after summing over jy check if it's detatched 
            
            if sumx[jx]==0:
                include=False
        
        
        #after filtering the 'detatched' 
        if include is True:
            sumtotal=0
            for jx in range(NCcell_x):
                sumtotal=sumx[jx] +sumtotal
            
            #2nd filter if density is correct 
            if sumtotal==numberNoCuts and listBinary[i] not in combinations:
                
                
                combinations.append(listBinary[i])
                
                
                #reflection around y 
                reftotal =""
                for nx in range (NCcell_x):
                    string1 = (listBinary[i][nx*NCcell_y:(nx+1)*NCcell_y])
                    #print ((string1))
                    ref =""
                    for k in range(len(string1)):
                        ref = str(string1[k]) + ref
                    
                    reftotal = reftotal + ref
                
                #if reftotal != listBinary[i]:
                #to make sure we don't create double configurations! 
                #for now we include duplicates of the 'identity' but make sure delete duplicates during analysis! 
                combinations.append(reftotal)
                
                #reflection around x
                reftotal2 =""
                for nx in range (NCcell_x):
                    string1 = (reftotal[nx*NCcell_y:(nx+1)*NCcell_y])
                    reftotal2 = string1 + reftotal2 
                 
                combinations.append(reftotal2)
                
                #reflection around y AGAIN!
                reftotal =""
                for nx in range (NCcell_x):
                    string1 = (reftotal2[nx*NCcell_y:(nx+1)*NCcell_y])
                    #print ((string1))
                    ref =""
                    for k in range(len(string1)):
                        ref = str(string1[k]) + ref
                    
                    reftotal = reftotal + ref
                 
                combinations.append(reftotal)
                 
                
    return combinations
    

    
