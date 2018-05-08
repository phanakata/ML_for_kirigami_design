import numpy as np

def generateInnerCell(NCcell_x, NCcell_y, ncell_x, ncell_y):
    """
    Create pristine ribbon of size (ncell_x-4)*ncell_y
    """    
    inner = np.ones((ncell_x-4)*ncell_y)
    
    #for debugging
    #for j in range (ncell_x):
    #   print (kirigami[j*ncell_y:(j+1)*(ncell_y)])
    
    return inner

def makeCutsonCell(cutConfigurations, inner, NCcell_x, NCcell_y, ncell_x, ncell_y):
    """
    Make cuts in inner region 
    
    Parameters
    --------------------
    cutConfigurations: string
        N-dimensional (binary) array of cuts with size NCcell_x * NCcell_y
    inner: array
        N-dimensional (binary) array with size (ncell_x-4)*ncell_y
    return: inner (array)        
    """
    
    mx = (ncell_x-4)//NCcell_x
    my = ncell_y//NCcell_y
    #debugging: print(mx, my)
    
    #ONLY MAKE CUTS inside the INNER REGION !!
    for i in range (len(inner)):
        #first find index nx and ny 
        nx = (i)//ncell_y
        ny = (i)%ncell_y
        
        #now convert nx and ny to Nx Ny
        Nx = nx//mx
        Ny = ny//my
        #debugging: print(Nx, Ny)
        
        #now conver (Nx, Ny) to one dimensional and check whether it's 0 or 1 
        index = Nx * NCcell_y + Ny
        
        if (cutConfigurations[index]==0):
            if nx> Nx * mx +mx/3 and nx < Nx* mx +mx *2/3:
                inner[i] = 0
    
    return inner     
