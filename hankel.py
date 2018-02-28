from __future__ import division
import numpy as np
from scipy.sparse.linalg import LinearOperator
from scipy.fftpack import fft, ifft
#from numpy.fft import fft, ifft
from scipy.sparse.linalg import svds
from copy import copy

class Hankel(LinearOperator):
	""" Abstract Hankel matrix class

	

	Parameters
	----------
	c: numpy.ndarray (n,)
		First column of the Hankel matrix
	r: numpy.ndarray (m,)
		Last row of the Hnake matrix (first entry is ignored)
	"""
	def __init__(self, c, r = None, Fh = None):
		if r is None:
			r = np.zeros_like(c)
		
		self.c = c
		self.r = r
		# Build vector that defines the matrix
		h = np.hstack([c,r[1:]])
		# Parameters of this matrix
		self.shape = (len(c), len(r))
		self.dtype = h.dtype

		# Precompute the FFT for fast matrix-vector products
		if Fh is None:
			self.Fh = fft(h)	
		else:
			self.Fh = Fh

		self._is_adjoint = False

	def _matmat(self, x):
		# Check dimensions
		assert x.shape[0] == self.shape[1]

		if self._is_adjoint is False:	
			xx = np.vstack([np.flipud(x), np.zeros((self.shape[0]-1,x.shape[1]))])
		else:
			xx = np.vstack([np.zeros((self.shape[0]-1,x.shape[1])), x])
	
		Fx = fft(xx, axis = 0)

		n = self.shape[0] + self.shape[1]		

		if self._is_adjoint is False:
			Hx = ifft(self.Fh[:,None] * Fx, axis = 0)[-self.shape[0]:,:]
		else:
			Hx = np.flipud(ifft(self.Fh[:,None] * Fx, axis = 0)[:self.shape[0],:])
		
		# Test to see what the output format will look like
		out_format = x[0,0]*self.c[0]
			
		# Remove the complex part if the output should be real
		if np.isreal(out_format):
			Hx = Hx.real
		# Round the output prior to conversion to an integer output
		if out_format.dtype not in [np.float16, np.float32, np.float64, np.complex64, np.complex128]:
			Hx = np.round(Hx)
		return Hx.astype(out_format.dtype)


	def _adjoint(self):
		self2 = Hankel(self.r, self.c, Fh = self.Fh.conj())
		self2.shape = (self.shape[1], self.shape[0])
		if self._is_adjoint:
			self2._is_adjoint = False
		else:
			self2._is_adjoint = True

		return self2


	def norm(self, tol = 1e-8):
		if self.Fh.shape[0] > 10:
			return float(svds(self,  k = 1, tol = tol, return_singular_vectors = False, ncv = 50 ))
		else:
			return np.linalg.norm(hankel(self.c,self.r), 2)


# Unit tests below
from scipy.linalg import hankel


def test_int(n = 5, m = 10):

	# Build Hankel matrices
	c = (10*np.random.randn(n)).astype(np.int)
	r = (10*np.random.randn(m)).astype(np.int)
	H_dense = hankel(c,r)
	H = Hankel(c,r)

	# Left and right side vectors
	x = (10*np.random.randn(m)).astype(np.int)
	X = (10*np.random.randn(m,2)).astype(np.int)
	y = (10*np.random.randn(n)).astype(np.int)

	

	# Test for equality since we are using intergers
	assert np.all(H*x == np.dot(H_dense,x))
	assert np.all(H*X == np.dot(H_dense,X))
	Hadj = H.adjoint()
	assert np.all( (Hadj*y) == np.dot(H_dense.T.conj(), y))
	# Check that we haven't changed the memory in the original when creating the adjoint
	assert np.all(H*X == np.dot(H_dense,X))

	assert (H*X).dtype == np.int


def test_float(n = 5, m = 10):

	# Build Hankel matrices
	c = (10*np.random.randn(n))
	r = (10*np.random.randn(m))
	H_dense = hankel(c,r)
	H = Hankel(c,r)

	# Left and right side vectors
	x = (10*np.random.randn(m))
	X = (10*np.random.randn(m,2))
	y = (10*np.random.randn(n))

	

	# Test for approximate equality due to imprecession of floats 
	assert np.allclose(H*x, np.dot(H_dense,x))
	assert np.allclose(H*X, np.dot(H_dense,X))
	Hadj = H.adjoint()
	assert np.allclose( (Hadj*y),  np.dot(H_dense.T.conj(), y))
	# Check that we haven't changed the memory in the original when creating the adjoint
	assert np.allclose(H*X , np.dot(H_dense,X))
	assert (H*X).dtype == np.float_


def test_float_complex(n = 5, m = 10):

	# Build Hankel matrices
	c = (10*np.random.randn(n)) + 1j*np.random.randn(n)
	r = (10*np.random.randn(m)) + 1j*np.random.randn(m)
	H_dense = hankel(c,r)
	H = Hankel(c,r)

	# Left and right side vectors
	x = (10*np.random.randn(m)) + 1j*np.random.randn(m)
	X = (10*np.random.randn(m,2)) + 1j*np.random.randn(m,2)
	y = (10*np.random.randn(n)) + 1j*np.random.randn(n)

	# Test for approximate equality due to imprecession of floats 
	print H*x
	print np.dot(H_dense, x)
	assert np.allclose(H*x, np.dot(H_dense,x))
	assert np.allclose(H*X, np.dot(H_dense,X))
	Hadj = H.adjoint()
	assert np.allclose( (Hadj*y),  np.dot(H_dense.T.conj(), y))
	# Check that we haven't changed the memory in the original when creating the adjoint
	assert np.allclose(H*X , np.dot(H_dense,X))
	assert (H*X).dtype == np.complex_

def test_norm(n = 5, m = 10):
	# Build Hankel matrices
	c = (10*np.random.randn(n)) + 1j*np.random.randn(n)
	r = (10*np.random.randn(m)) + 1j*np.random.randn(m)
	H_dense = hankel(c,r)
	H = Hankel(c,r)

	assert np.allclose(H.norm(), np.linalg.norm(H_dense,2))

