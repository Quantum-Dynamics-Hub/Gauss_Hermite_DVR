#!/usr/bin/python
##!/usr/lusers/dblinger/python/Python-2.7.11/python

############################
# Code by:Erica Chong and Winston Wright
# Last modified by David Lingerfelt: Jan, 2017 
#
# Simple program to diagonalize a Hamiltonian matrix
# in the DVR (pseudo-spectral) basis via 
# Gauss-Hermite DVR, and print eigenfunction in coordinate
# representation.
#
# For more info on implementation, see John Light and Tucker Carington
# "Discrete Variable Representations and their Utiliazation"
#
############################

from __future__ import division
import math
import cmath
import sys
import numpy as np
from  mpmath import *
from scipy.linalg import *
from scipy.integrate import simps
from scipy.special import h_roots
from time import strftime

xscale = 1.0
print strftime("%Y-%m-%d %H:%M:%S")
n = 50  # sets both number of grid points for the DVR (i.e. the highest order hermite polynomial in the spectral basis) 
npts = 100 # sets the resolution for the printing of eigenfunctions and potential to file
mass = 1.0      # electron mass in au
hBar = 1.0 	# au 
PIM4 = math.pow(math.pi,(-1/4))
#mp.dps = 12 # Sets the (decimal) precision for the mp floats (set this smallest that prevents overflows/div by zero in Herms and Hermsatroot builds) 
xscale = 3.0 # factor to scale the coordinates (in AU) by.  Useful if using small n to cover a large system, and very high grid point resolution isn't needed

nele = float(sys.argv[1]) 
nprot = float(sys.argv[2])
RQD = float(sys.argv[3])
l = float(sys.argv[4])


outfile = str(int(nele))+'_elect_'+str(int(nprot))+'_prot_'+str(RQD)+'_Bohr_QD_l_'+str(int(l))
o1 = str( str(outfile) + '_output.txt')
output4 = open(o1,"w")
output4.write(strftime("%Y-%m-%d %H:%M:%S") + '\n')
output4.write('number electron, protons, and QD radius (Bohr), orbital angular momentum (l):'+str(int(nele))+str(int(nprot))+str(RQD)+str(int(l)))
output4.write("n = " + str(n) + '\n')


### DEFINE THE POTENTIAL

# Harmonic oscillator and parabolic doublewell tests
# 
#def potential(x):
#    omega = .001 		# sqrt force constant
# DBL test of coordinate rescaling
#    x = 0.1*x # convert from bohr to nm for units of x
#    return (0.5*mass*omega*omega*x*x) 
#   parabolic double well potential:
#    return 1.0/4.0*(abs(x/2.0) - 2.0)**2  


def potential(x):
# Parameters for the potential
  R = RQD #QD radius in Bohr
  x = xscale*x
#  epsOut = 7.56 # THF permitivity
  epsOut = 1.0 # Vacuum permitivity 
  epsRel =  10.0 # epsilon_r for ZnO
  epsR = 1 + (epsRel-1)/(1+(1.7)/(R**1.8)) 
  if abs(x) < R:
    return -((nele-1)*x**2)/(2*epsR*R**3)+(nele-1)/(2*epsR*R)+(l**2+l)/(2*x**2) 
#    return -((nele-1)*x**2)/(2*epsR*R**3)+(nele-1)/(2*epsR*R)    
  else:
    return -(nprot-(nele-1))/(epsOut*abs(x))+(nprot-(nele-1))/(epsOut*R)+(l**2+l)/(2*x**2) 
#    return -(nprot-(nele-1))/(epsOut*abs(x))+(nprot-(nele-1))/(epsOut*R)


# orthonormal set of Hermite polynomials (physicistsi form) with arbitrary precision                                                                                    
def hermite(n,x=0):
  if n == -1:
      return mpf(0)                                                                                                        
  if n == 0:
      return mpf(math.pow(math.pi,(-1/4)))                                                                                                        
  return mpf(x*math.sqrt(2/n)*hermite(n-1,x) - math.sqrt((n-1)/n)*hermite(n-2,x)) 


# weighting function for Hermite polynomials
def w(x):
  return mpf(math.pow(math.e,(-x*x)))

# Step 0. Setting up the DVR gridpoints and weights

print "Finding the roots of Hermite polynomials (and associated weights)"
xVals,wVals = h_roots(n,False)
print "Complete"
#xmax = xVals[-1]*xscale
#xmin = xVals[0]*xscale 
xmax = xVals[-1]
xmin = xVals[0]
dx = (xmax - xmin)/(npts-1)
print "xmin, ", xmin*xscale, " xmax, ", xmax*xscale
print "Evaluating the polynomials @ roots and the printing grid points"
# Evaluate H0 and H1 on a grid and use forward recursion with the value of the functions at those points.  
# This will prevent n-factorial scaling of the eigenfunction printing 
Herms = np.zeros((npts,n))*mpf(1.0) # initialize an arbitrary precision array
for a in range(npts):
  Herms[a,0] = hermite(0,xmin+a*dx)
  Herms[a,1] = hermite(1,xmin+a*dx)
i = 2
while i < n:
  for a in range(npts):
    ex = xmin + a*dx
    Herms[a,i] = mpf(ex*math.sqrt(2/(i))*Herms[a,i-1] - math.sqrt((i-1)/i)*Herms[a,i-2]) # need arbitrary precisions here since these can get very small (not representable by double-precision floats)
  i = i + 1
HermsAtRoots = np.zeros((n,n))*mpf(1.0) # initialize an arbitrary precision array
for a in range(n):
  HermsAtRoots[a,0] = hermite(0,xVals[a])
  HermsAtRoots[a,1] = hermite(1,xVals[a])
i = 2
while i < n:
  for a in range(n):
    HermsAtRoots[a,i] = mpf(xVals[a]*math.sqrt(2/(i))*HermsAtRoots[a,i-1] - math.sqrt((i-1)/(i))*HermsAtRoots[a,i-2])# need arbitrary precisions here since these can get very small (not representable by double-precision floats)
  i = i + 1
print "Complete"

# Step 1. Set up the uniformly spaced grid to evaluate the potential on for printing 

print "Forming potential matrix in the DVR"
Vofx = np.zeros((npts))
xpts = np.zeros((npts))
for i in range(npts):
  xpts[i] = xmin+i*dx
  Vofx[i] = potential(xpts[i])
OutputV = open(str(str(outfile) +"_potential.txt"),"w")
np.savetxt(OutputV, Vofx) # in au
OutputV.close
# Step 1.5 Build the potential matrix in the pseudo-spectral (DVR) basis
potvec = np.ndarray((n))
for i in range(n):
  potvec[i] = potential(xVals[i])
potentialDVR = np.diag(potvec)
OutputX = open(str(str(outfile) +"_xpoints.txt"),"w")
np.savetxt(OutputX, xpts*xscale) # in au
OutputX.close
print "Complete"

# Step 2. Form the kinetic energy matrix directly in pseudo-spectral basis

print "Forming kinetic matrix in the DVR"

def md2dxn(j,k,roots,n):
  if j==k:
    return (-2.0*(n-1)/3.0)-0.5+(roots[j]**2/3.0)
  elif (j+k)%2!=0:
    return -1*(0.5-2.0/((roots[k]-roots[j])**2))
  else:
    return 0.5-2.0/((roots[k]-roots[j])**2)

kineticDVR = np.zeros((n,n))
for M in range(n):
  for N in range(n):
    kineticDVR[M,N] = (-hBar*hBar/(2.0*mass))*md2dxn(M,N,xVals,n) 
print "Complete"
# unit conversion for the second derivative in the kinetic energy
kineticDVR = kineticDVR*xscale**-2

# Step 3. Form full Hamiltonian Matrix

hamiltonianDVR = potentialDVR + kineticDVR

# Step 4. Diagonalize Hamiltonain Matrix

print "Diagonalizing Hamiltonian in the DVR"

evals, evecs = eig(hamiltonianDVR)
idx = evals.argsort()
evals = evals[idx]
evecs = evecs[:,idx]

# Step 5.  Print out the results (eigenvalues and eigenfunctions)

OutputEvals = open(str(str(outfile) +"_evals.txt"),"w")
np.savetxt(OutputEvals, evals)
OutputEvals.close
print "Complete"

output4.write("Transform to pseudo-spectral basis   " + strftime("%Y-%m-%d %H:%M:%S") + '\n')
# Evaluate the eigenfunctions  on the regular spaced grid.  This requires a transformation to the spectral representation, since otherwise we only know the value of these functions at the DVR grid points, which isn't compatable with printing on an equally spaced grid.
# This is ironically the most expensive part of the program, so in future, we may want to just print out the eigenfunctions at un-equally spaced points and avoid this step.
thetarray = np.zeros((npts,n))
for i in range(n):
  print i,"/",n, "progress in building transformation matrix from DVR to spectral basis"
  wvali = math.sqrt(wVals[i])
  for a in range(npts):
    for m in range(n):
      ex = xmin+a*dx # current position
      newpart = math.sqrt(w(ex))*Herms[a,m]*wvali*HermsAtRoots[i,m]
      thetarray[a,i] = thetarray[a,i] + newpart 

## numerical integration to check normalization of pseudo-spectral basis functions
#print simps(thetarray[:,1]**2,xpts)

output4.write("Printing energy eigenfunctions in coordinate rep   " + strftime("%Y-%m-%d %H:%M:%S") + '\n')
Psix = np.dot(thetarray,evecs)
print str(simps(Psix[:,1]**2/xscale,xpts*xscale))
#for i in range(n):
#  Psix[:,i]=np.add(Psix[:,i]**2,evals[i]*27.211385) # offset e-functs by e-val (in ev)
#  Psix[:,i]=np.add(Psix[:,i],evals[i])  # offset e-funct by e-val (au)
OutputPsi = open(str(str(outfile) +"_eigenfunct.txt"),"w")
np.savetxt(OutputPsi, Psix/xscale)
OutputPsi.close
## Print a bunch of Debug info to the output file 

output4.write("Printing data  " + strftime("%Y-%m-%d %H:%M:%S") + '\n')
output4.write("xvals" + '\n' + "   ".join(str(item) for item in xVals) + '\n')
output4.write("wVals" + '\n' + "   ".join(str(item) for item in wVals) + '\n')

output4.write('\n' + strftime("%Y-%m-%d %H:%M:%S") + '\n')
output4.write("potentialDVR" + '\n')
for row in potentialDVR:
  output4.write('  '.join(str(elem) for elem in row) + '\n')

output4.write('\n' + strftime("%Y-%m-%d %H:%M:%S") + '\n')
output4.write('\n' + "kineticDVR" + '\n')
for row in kineticDVR:
  output4.write('  '.join(str(elem) for elem in row) + '\n')

output4.write('\n' + strftime("%Y-%m-%d %H:%M:%S") + '\n')
output4.write('\n' + "hamiltonianDVR" + '\n')
for row in hamiltonianDVR:
  output4.write('  '.join(str(elem) for elem in row) + '\n')
output4.write('\n' + "eigenvalues" + '\n' + "   ".join(str(item) for item in evals) + '\n')

output4.write('\n' + strftime("%Y-%m-%d %H:%M:%S") + '\n')
output4.write('\n' + "eigenvectors" + '\n')
for row in evecs:
  output4.write('  '.join(str(elem) for elem in row) + '\n')

output4.write("Finished!!  " + strftime("%Y-%m-%d %H:%M:%S") + '\n')
output4.close

print "Finished!!"
print strftime("%Y-%m-%d %H:%M:%S")
