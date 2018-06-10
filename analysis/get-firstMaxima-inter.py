import numpy as np
data = np.loadtxt('stress-i.dat')
data2 = np.loadtxt('stress.dat')
#start at 5 after few percentage of strain 
i=2
stress_cutoff =1.5
maxstress=data[i, 1]
strain=data[i,0]
index=0
#list of peaks 
peaks = []
#find strains  
# to make sure it's not due to noise, we need to check i+2 instead of i+1
while (i<(len(data)-4)):
    if (data[i, 1]>stress_cutoff and data[i-1, 1]<data[i, 1] and data[i, 1]>data[i+1, 1] and data[i, 1]>data[i+2, 1]):
        maxstress = data[i, 1]
        strain = data[i,0]
        peaks.append(i)
                #index = i
    i = i + 1

#only find FIRST PEAK
# try also last  peak 
if peaks==[]:
    #use stress.dat instead
    #print ("RUN LONGER!")
    #use last data point for the moment 
    index = len(data2)-1
    maxstress = data2[index, 1]
    strain = data2[index, 0]
    #need index+1!
    #have checked on xmgrace
    toughness = np.trapz(data2[0:index+1, 1], data2[0:index+1,0])
else:
    index = peaks[0]
    maxstress = data[index, 1]
    strain = data[index, 0] 
    #need index+1!
    #have checked on xmgrace
    toughness = np.trapz(data[0:index+1, 1], data[0:index+1,0])
print (strain, toughness,  maxstress)
