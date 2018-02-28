
from scipy.sparse.linalg import svds
from hankel import Hankel
import numpy as np

def mkV(omega, n):
	V = np.zeros((n, len(omega)), dtype = np.complex)
	for i, om in enumerate(omega):
		V[:,i] = np.exp(om * np.arange(n))
	return V

def hsvd(y, p):
	"""
	"""
	n = y.shape[0]
	l = int(np.floor(n/2))

	H = Hankel(y[:l+1],y[l:])
	U,S,VT = svds(H, k = p)
	# Moore-Penrose solution
	A = np.dot(U[:-1].T.conj(), U[1:])
	ut = U[-1]
	Z = A + np.outer(ut.T.conj(), np.dot( ut, A))/(1 - np.dot(ut, ut.conj()))
	#Z = np.linalg.lstsq(U[:-1], U[1:])[0]
	omega = np.log(np.linalg.eigvals(Z))
	# Find a by least squares
	V = mkV(omega,n)
	a = np.linalg.lstsq(V, y)[0]
	return omega, a
