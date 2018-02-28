# Generate test data on random Hankel matrices
from __future__ import division
import numpy as np
from joblib import Memory
from scipy.sparse.linalg import svds
memory = Memory(cachedir = './.dat', verbose = 0)


from hankel import Hankel
from hankel_bound import hankel_gaussian_complex, hankel_gaussian_real

# This decorator caches the output to a separate file
# so that we don't have to repeat our calls to this function
@memory.cache
def random_hankel_norm(seed = 0, n = 1, m = 1, complex_ = False, tol = 1e-10):
	""" Computes the norm of a random Hankel matrix

	Given a seed, construts a Hankel matrix with iid Gaussian entires and 
	computes the corresponding L2 norm using ARPACK with a fast Hankel vector product.

	Parameters
	----------
	seed: interger
		This parameter is feed to numpy.seed to initialize the random number generator
	
	n: integer
		Number of rows

	m: integer
		Number of columns
	
	complex_: boolean
		If true, the entries of the Hankel matrix sample a circular complex
		normal distribution (i.e., independent real and imaginary parts)

	tol: float
		Tolerance for ARPACK convergence
	"""
	# set the seed so we obtain deterministic answers
	np.random.seed(seed)
	if complex_:
		c = (np.random.randn(n) + 1j * np.random.randn(n))/np.sqrt(2)
		r = (np.random.randn(m) + 1j * np.random.randn(m))/np.sqrt(2)
	else:
		c = np.random.randn(n)
		r = np.random.randn(m)
	
	H = Hankel(c,r)

	# Print notification that the code is running
	print "%05d: computing SVD of %d x %d Hankel matrix" % (seed, n,m)
	
	# compute the largest singular value (i.e, the 2-norm)
	# I've noticed increasing the number of vectors used (ncv) helps decrease 
	# occurance of convergence errors

	return H.norm()



if __name__ == '__main__':
	
	#n_vec = np.array([ 5, 10, 20, 50,  
	#				 100,       200,    500,
	#				 1000,     2000,   5000])
					 #10000,   20000,  50000])#,   20000,  50000,
					 #100000, 200000, 500000,
					 #1000000])
	n_vec = np.array(2**np.arange(2,21), dtype = np.int64)
	print n_vec
	niter = 1000
	percentile = np.array([0.01, 0.05, 0.5, 0.95, 0.99])
	norm_vec = np.zeros((n_vec.shape[0],len(percentile)))

	complex_ = False

	for k, n in enumerate(n_vec):
		# Generate array of random Hankel matrix norms.
		# By iterating through different seeds we generate a deterministic (but random) distribution.
		# To avoid recomputation, the memory.cache decorator caches the result of this function
		# so we don't repeat unnecessarly between runs
		dist = np.array([random_hankel_norm(seed = i, n = n//2+1, m = n//2, complex_ = complex_) for i in range(niter)])

		# Compute the percentiles
		norm_vec[k,:] = np.percentile(dist, percentile)

		print "n = %5d \t %dth percentile: %g" % (n, int(percentile[0]*100), norm_vec[k,0])


	# Dump the data into a PGF readible format
	from pgf import PGF
	pgf = PGF()
	pgf.add('n', n_vec.astype(int))
	for k, p in enumerate(percentile):
		pgf.add('p' + str(int(p*100)), norm_vec[:,k])

	if complex_:
		for k, p in enumerate(percentile):
			bnd = np.array([hankel_gaussian_complex(n, p = p) for n in n_vec])	
			pgf.add('bnd' + str(int(p*100)), bnd)
		pgf.write('fig_random_hankel_complex.dat')
	else:
		for k, p in enumerate(percentile):
			bnd = np.array([hankel_gaussian_real(n, p = p) for n in n_vec])	
			pgf.add('bnd' + str(int(p*100)), bnd)
		pgf.write('fig_random_hankel_real.dat')

