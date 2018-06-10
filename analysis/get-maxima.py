import numpy as np
data = np.loadtxt('stress.dat')
i=0
maxstress=data[0, 1]
strain=data[0,0]
index=0
#find strains  
while (i<len(data)):
	if data[i, 1]>maxstress:
        	maxstress = data[i, 1]
        	strain = data[i,0]
                index = i
	i = i + 1

#need index+1!
#I have checked this on xmgrace 
toughness = np.trapz(data[0:index+1, 1], data[0:index+1,0])

print strain, toughness,  maxstress
