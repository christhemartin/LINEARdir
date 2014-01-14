import numpy as np
import pylab as py
import Functions as func
import itertools

## searching for all stars with chi2dof > 6 and looking at
## their best fits
allData = func.read_data()

#sort object ids alongside chi2dof to identify worst fits 
sorted_index = np.argsort(allData['chi2dof'])
chi2dof_sorted = [allData['chi2dof'][i] for i in sorted_index]
id_sorted = [allData['id'][i] for i in sorted_index]

#how many are worse than 6? (64)
count = 0
for i in itertools.ifilter(lambda x: x > 6, chi2dof_sorted): count += 1
print count
print chi2dof_sorted[-100:]
print id_sorted[-100:]

#populate Bad_Fit_Plots with plots of worst fit objects
#bad_plots(id_sorted[-100:])

#how are coefficients correlated to lcType?
#func.correlateCoeffs(allData)