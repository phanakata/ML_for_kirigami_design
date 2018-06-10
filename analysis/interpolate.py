import numpy as np
#from scipy.interpolate import interp1d
#from scipy import interpolate
from scipy.interpolate import UnivariateSpline
data = np.loadtxt('stress.dat')

#make sure x is strictly incereasing 
i=1
while i < (len(data)):
    if data[i-1][0]>=data[i][0]:
        #print(i)
        #print(data[i][0])
        data=np.delete(data, i,0)
        #i = i+1
    else:
        i = i+1


#import matplotlib.pyplot as plt
#start at 5 after few percentage of strain 
y = data[:, 1]
x = data[:, 0]
    


xmax = x[-1]
xs = np.linspace(0, xmax, 200)

#plot original data
#plt.plot(x, y, 'ro', ms=5)


#no smoothing
#spl = UnivariateSpline(x, y)
#plt.plot(xs, spl(xs), 'g', lw=3)
#xs2 = np.linspace(0, 1.7, 1000)

spl = UnivariateSpline(x, y)
spl.set_smoothing_factor(20)
#plt.plot(xs, spl(xs), 'b', lw=1)
#plt.show()

#we will use the smoothing factor 
np.savetxt('stress-i.dat', np.c_[xs,spl(xs)])
