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


def addEdges(inner, ncell_x, ncell_y):
    """
    Add egdes to the system
    """
    #creates egdes with no cuts!
    edge1 = np.ones(2*ncell_y)
    edge2 = np.ones(2*ncell_y)

    #add them all together
    cell = np.append(edge1, inner)
    cell = np.append(cell, edge2)

    return cell

def generateLattice(kirigami, NCcell_x, NCcell_y, ncell_x, ncell_y, cell_x, cell_y, cell_z, Lx, Ly, Lz):
    """
    Generate lattice graphene kirigami 
    There are FOUR carbon atoms per unit cell 
    The positions is shifted so that (0,0) is located at the center of the ribbon
    
    Parameters
    ----------------------------------
    
    listAtoms: list
        containts atoms x y z positions 
    basisi_j: float
        basis vector of sublattice 'i' in 'j' direction 
    """

    # there are 4 atoms per unit cell 
    # atomic basis in unit of lattice vectors 
    basis1_x  = 0.0
    basis1_y  = 0.5
    basis1_z =  0.5 

    basis2_x  = 1/6
    basis2_y  = 0
    basis2_z =  0.5

    basis3_x  = 0.5
    basis3_y  = 0.
    basis3_z =  0.5

    basis4_x  = 2/3
    basis4_y  = 0.5
    basis4_z =  0.5 
    
    listAtoms = []
    for i in range (len(kirigami)):
        #convert N-dimensional vector to 2D nx ny
        nx = i//ncell_y
        ny = i%ncell_y
        nz = 0 #we are not repeating cells in z direction
        #if unit cell is not empty create atoms! 
        if kirigami[i]>0:
            x = nx * cell_x + basis1_x * cell_x - Lx/2
            y = ny * cell_y + basis1_y * cell_y - Ly/2
            z = nz * cell_z + basis1_z * cell_z - Lz/2
            listAtoms.append([x, y, z])
        
        
            x = nx * cell_x + basis2_x * cell_x - Lx/2
            y = ny * cell_y + basis2_y * cell_y - Ly/2
            z = nz * cell_z + basis2_z * cell_z - Lz/2
            listAtoms.append([x, y, z])
        
        
            x = nx * cell_x + basis3_x * cell_x - Lx/2
            y = ny * cell_y + basis3_y * cell_y - Ly/2
            z = nz * cell_z + basis3_z * cell_z - Lz/2
            listAtoms.append([x, y, z])
        
            x = nx * cell_x + basis4_x * cell_x - Lx/2
            y = ny * cell_y + basis4_y * cell_y - Ly/2
            z = nz * cell_z + basis4_z * cell_z - Lz/2
            listAtoms.append([x, y, z])
            
    return listAtoms



def getSize(listAtoms):
    """
    Helper functions to find the boundaries
    Convert list to numpy array for easy visualizations
    """
    X = np.zeros(len(listAtoms))
    Y = np.zeros(len(listAtoms))
    Z = np.zeros(len(listAtoms))
    for i in  range(len(listAtoms)):
        X[i] = listAtoms[i][0]
        Y[i] = listAtoms[i][1]
        Z[i] = listAtoms[i][2]

    xhi, xlo = np.max(X), np.min(X)
    yhi, ylo = np.max(Y), np.min(Y)
    zhi, zlo = np.max(Z), np.min(Z)

    return (xhi, xlo, yhi, ylo, zhi, zlo)
