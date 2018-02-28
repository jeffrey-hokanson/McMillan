from __future__ import division
import numpy as np
from scipy.special import erf
from scipy.optimize import newton,root


def hankel_gaussian_real(n, p = 0.95):
	if n % 2 == 0:
		def obj(alpha):
			return erf(alpha/2)**2 * (-np.expm1(-alpha**2/2))**( n/2 - 1)
			#return erf(alpha/2)**2 * (1 - np.exp(-alpha**2/2))**( n/2 - 1)
	else:
		def obj(alpha):
			return erf(alpha/2) * (-np.expm1(-alpha**2/2))**( (n-1)/2)

	
	root_est = np.sqrt(-2*np.log(1 - p**(1.0/n)))
	sol = root(lambda a: obj(a) - p, root_est, options = {'maxfev':int(1e4)})
	return float(sol.x)*np.sqrt(n)

from scipy.stats import chi, chi2

def hankel_gaussian_complex(n, p = 0.95):
	
	def obj(alpha):
		#return chi2.logcdf(alpha**2,2)*n
		#return chi.logsf(alpha,2)/n
		return (-np.expm1(-alpha**2/2))**n

	root_est = np.sqrt(-2*np.log(1 - p**(1.0/n)))
	sol = root(lambda a: obj(a) - p, root_est, options = {'maxfev': int(1e4)})
	return np.sqrt(n)*float(sol.x)

if __name__ == '__main__':
	
	for n in [2, 5, 10,20,50,100, 200, 500, 1000, 2000, 5000,10000, 20000, 50000]:
		print n,'\t', hankel_gaussian_complex(2*n+1),'\t', hankel_gaussian_real(2*n+1)
