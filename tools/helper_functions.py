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
        genome_ID=genome_ID+str(int(cutConfigurations[i]))

    return genome_ID


def writeInput_for_LAMMPS(rd, listAtoms, filename):
    """
    Function to write data file for LAMMPS geometry input
    Boundaries need to be slightly larger to
    make sure all atoms are wrapped inside the simulation box
    """
    #f=open("geo.kirigami_d0.0_"+str(rd),"w+")
    f=open(filename+str(rd),"w+")
    f.write("\n")
    f.write("%d atoms\n" %len(listAtoms))
    f.write("1 atom types\n")
    f.write("\n")
    f.write("%f\t%f xlo xhi\n" %(xlo-1, xhi+1))
    f.write("%f\t%f ylo yhi\n" %(ylo-1, yhi+1))
    f.write("%f\t%f zlo zhi\n" %(zlo-1, zhi+1))
    f.write("\n")
    f.write("Atoms\n")
    f.write("\n")
    for i in range (len(listAtoms)):
        f.write("%d\t1\t%f\t%f\t%f\n" %(i+1, listAtoms[i][0], listAtoms[i][1], listAtoms[i][2]))
    f.close()
