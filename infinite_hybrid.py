#from fns import Measurement
import time
import math
import mpmath
import scipy
import numpy as np
import scipy.special
import matplotlib
import matplotlib.pyplot as plt
import os
import numpy.linalg
pi=3.14159265359
eps0= 8.854187817*10**-12
############### old code
t0=time.time()
# single_recursive_images
'''def fP(n,z):
	#print(scipy.special.legendre(n)(z) )
	return(scipy.special.lpn(n,z) )# legrende polynomial'''
def fF(phi,k,accuracyLimit):
	# This is the trigonometric form of the integral; substituting t = sin theta and x = sin phi, one obtains Jacobi's form
	# single_recursive_images uses Jacobis form
	# need to implement own function to handle complex arguments
	# out=scipy.special.ellipkinc(phi.real, k*k)
	# print(phi, out)
	# return out
	outreal=scipy.integrate.quad(lambda x: fFargRe(k,phi,x), 0, 1,epsrel=accuracyLimit)
	outimag=scipy.integrate.quad(lambda x: fFargIm(k,phi,x), 0, 1,epsrel=accuracyLimit)
	#print(outreal[0]*abs(phi),outimag[0]*abs(phi))
	return (outreal[0]+1j*outimag[0])*phi

def fF_imag(phi,k,accuracyLimit):
	# This is the trigonometric form of the integral; substituting t = sin theta and x = sin phi, one obtains Jacobi's form
	# single_recursive_images uses Jacobis form
	# need to implement own function to handle complex arguments
	# out=scipy.special.ellipkinc(phi.real, k*k)
	# print(phi, out)
	# return out
	outimag=scipy.integrate.quad(lambda x: fFargIm(k,phi,x), 0, 1,epsrel=accuracyLimit)
	#print(outreal[0]*abs(phi),outimag[0]*abs(phi))
	return outimag[0]*phi
def fFargRe(k,phi, x):
	theta=phi*x
	return (1/np.sqrt(1-k*k*np.sin(theta)**2)).real
def fFargIm(k,phi, x):
	theta=phi*x
	return (1/np.sqrt(1-k*k*np.sin(theta)**2)).imag
def fdtDdz(k_Iinf,lamda,z):
	return (2*pi/(k_Iinf*lamda))*np.cos(2*pi*z/lamda)
def fdwDdt(k_I,k_Iinf,t):
	return -1/k_I/np.sqrt(	(t-1)*(t+1)*(t-1/k_Iinf)*(t+1/k_Iinf)	)
def fdwDdz(k_I,k_Iinf,t,lamda,z):
	return fdwDdt(k_I,k_Iinf,t)*fdtDdz(k_Iinf,lamda,z)

class layer:
	def __init__(self,eps11,eps33,eps13,t):
		self.eps11=eps11
		self.eps33=eps33
		self.eps13=eps13
		if self.eps33>0:
			self.epsr=(self.eps11/self.eps33-self.eps13**2/self.eps33**2)**0.5
		else:
			self.epsr=1
		self.t=t
		self.tEff=t*self.epsr
		self.iPosDir=None
		self.iNegDir=None
class interface:
	def __init__(self,layerPosDir,layerNegDir,z,maxsum):
		eps1=np.sqrt(layerPosDir.eps33*layerPosDir.eps11)
		eps2=np.sqrt(layerNegDir.eps33*layerNegDir.eps11)
		self.RPosDir=(eps2-eps1)/(eps1+eps2)
		self.TPosDir=self.RPosDir+1
		self.RNegDir=-self.RPosDir
		self.TNegDir=self.RNegDir+1
		self.z=z
		self.RPosDirEff=np.full((maxsum), None)#RPosDir)
		self.TPosDirEff=np.full((maxsum), None)#TPosDir)
		self.RNegDirEff=np.full((maxsum), None)#RNegDir)
		self.TNegDirEff=np.full((maxsum), None)#TNegDir)
		self.lPosDir=layerPosDir
		self.lNegDir=layerNegDir
	def getRPosDirEff(self,n):
		if self.RPosDirEff[n]==None:
			if self.lPosDir.iPosDir==None:
				self.RPosDirEff[n]=self.RPosDir
				self.TPosDirEff[n]=self.TPosDir
			else:
				rup=self.lPosDir.iPosDir.getRPosDirEff(n)
				t=self.lPosDir.tEff
				exp=np.exp(-np.pi*2*t*(2*n+1))
				factor=1/(1-self.RNegDir*rup*exp)
				self.RPosDirEff[n]=self.RPosDir+self.TPosDir*rup*self.TNegDir*exp*factor
				self.TPosDirEff[n]=self.TPosDir+self.TPosDir*rup*self.RNegDir*exp*factor
		return self.RPosDirEff[n]
	def getTPosDirEff(self,n):
		if self.TPosDirEff[n]==None:
			self.getRPosDirEff(n)
		return self.TPosDirEff[n]
	def getRNegDirEff(self,n):
		if self.RNegDirEff[n]==None:
			if self.lNegDir.iNegDir==None:
				self.RNegDirEff[n]=self.RNegDir
				self.TNegDirEff[n]=self.TNegDir
			else:
				rne=self.lNegDir.iNegDir.getRNegDirEff(n)
				t=self.lNegDir.tEff
				exp=np.exp(-np.pi*2*t*(2*n+1))
				factor=1/(1-self.RPosDir*rne*exp)
				self.RNegDirEff[n]=self.RNegDir+self.TNegDir*rne*self.TPosDir*exp*factor
				self.TNegDirEff[n]=self.TNegDir+self.TNegDir*rne*self.RPosDir*exp*factor
		return self.RNegDirEff[n]
	def getTNegDirEff(self,n):
		if self.TNegDirEff[n]==None:
			self.getRNegDirEff(n)
		return self.TNegDirEff[n]

class single_recursive_images:
	_aditcts=dict()#adict[k][n]=A_{2n+1}
	_Vdicts=dict()#Vdict[k][x]=V_{I,inf}(x,0)
	def __init__(self,eta,EleInt,t,epsx,epsy,maxsum=0,accuracyLimit=10**-7, inherit=None):
		if inherit==None:
			self.Layers=[]
			for e11,e33,T in zip(epsx,epsy,[np.inf]+list(t)+[np.inf]):
				self.Layers.append(layer(e11,e33,0,T))
			self.Interfaces=[]
			for i, _ in enumerate(self.Layers[0:-1]):
				if i==0:
					z=0
				else:
					z=sum(t[0:i])
				self.Interfaces.append(interface(self.Layers[i+1],self.Layers[i],z,maxsum))
				self.Layers[i].iPosDir=self.Interfaces[-1]
				self.Layers[i+1].iNegDir=self.Interfaces[-1]
		else:
			self.Layers=inherit.Layers
			self.Interfaces=inherit.Interfaces
		self.eta=np.array(eta)
		self.EleInt=EleInt
		self.layers=len(self.Layers)
		self.maxsum=maxsum
		self.accuracyLimit=accuracyLimit
		self.k=np.sin(pi/2*self.eta)
		self.Kk= scipy.special.ellipk(float(self.k**2))
		self.Kpk= scipy.special.ellipk(1-float(self.k**2))
		if not self.k in self._aditcts:
			self._aditcts[self.k]=dict()
		self.adict=self._aditcts[self.k] #dict()#adict[n]=A_{2n+1}
		if not self.k in self._Vdicts:
			self._Vdicts[self.k]=dict()
		self.Vdict=self._Vdicts[self.k] #dict()#adict[n]=A_{2n+1}
		self.tree=[]
		self.pFieldPos=np.full((maxsum), None)#RPosDir)
		self.pFieldNeg=np.full((maxsum), None)#TPosDir)
	def getPFieldPos(self,n):
		if self.pFieldPos[n]==None:
			if not self.pFieldPos[n-1]==1: # if self.pFieldPos[n-1]==1, we conclude that it will be 1 for all further components
				if len(self.Interfaces)>self.EleInt+1:
					rpo=self.Interfaces[self.EleInt+1].getRPosDirEff(n)
					rne=self.Interfaces[self.EleInt].getRNegDirEff(n)
					t=self.Layers[self.EleInt+1].tEff
					self.pFieldPos[n]=1/(1-rpo*rne*np.exp(-np.pi*2*t*(2*n+1)))
				else:
					self.pFieldPos[n]=1
				if self.EleInt>0:
					rpo=self.Interfaces[self.EleInt].getRPosDirEff(n)
					tpo=self.Interfaces[self.EleInt].getTPosDirEff(n)
					rne=self.Interfaces[self.EleInt-1].getRNegDirEff(n)
					t=self.Layers[self.EleInt].tEff
					self.pFieldPos[n]+=1  *rne*tpo*np.exp(-np.pi*2*t*(2*n+1))  /(1-rpo*rne*np.exp(-np.pi*2*t*(2*n+1)))
				if abs(self.pFieldPos[n]-1)<self.accuracyLimit: # mark that we are done calculating components, by setting self.pFieldPos[n]=1
					self.pFieldPos[n]=1
			else:
				self.pFieldPos[n]=1
		return self.pFieldPos[n]
	def getPFieldNeg(self,n):
		if self.pFieldNeg[n]==None:
			if not self.pFieldNeg[n-1]==1: # if self.pFieldNeg[n-1]==1, we conclude that it will be 1 for all further components
				if self.EleInt>0:
					rpo=self.Interfaces[self.EleInt].getRPosDirEff(n)
					rne=self.Interfaces[self.EleInt-1].getRNegDirEff(n)
					t=self.Layers[self.EleInt].tEff
					self.pFieldNeg[n]=1/(1-rpo*rne*np.exp(-np.pi*2*t*(2*n+1)))
				else:
					self.pFieldNeg[n]=1
				if len(self.Interfaces)>self.EleInt+1:
					rpo=self.Interfaces[self.EleInt+1].getRPosDirEff(n)
					rne=self.Interfaces[self.EleInt].getRNegDirEff(n)
					tne=self.Interfaces[self.EleInt].getTNegDirEff(n)
					t=self.Layers[self.EleInt+1].tEff
					self.pFieldNeg[n]+=1  *rpo*tne*np.exp(-np.pi*2*t*(2*n+1))  /(1-rpo*rne*np.exp(-np.pi*2*t*(2*n+1)))
				if abs(self.pFieldNeg[n]-1)<self.accuracyLimit: # mark that we are done calculating components, by setting self.pFieldPos[n]=1
					self.pFieldNeg[n]=1
			else:
				self.pFieldNeg[n]=1
		return self.pFieldNeg[n]
	def getC(self):
		return (self.Layers[self.EleInt].eps33*self.Layers[self.EleInt].epsr+self.Layers[self.EleInt+1].eps33*self.Layers[self.EleInt+1].epsr)*self.Kk/self.Kpk*eps0/2
	def getCintEx(self):
		if self.Layers[0].eps11>0:
			G,error=scipy.integrate.quad(lambda y: self.getEx(0,y), -np.inf, 0)
			C= G*self.Layers[0].eps11*eps0
		else:
			C=0
		for i in range(self.layers-2):
			if self.Layers[i+1].eps11>0:
				G,error=scipy.integrate.quad(lambda y: self.getEx(0,y), self.Interfaces[i].z, self.Interfaces[i+1].z)
				C+= G*self.Layers[i+1].eps11*eps0
		if self.Layers[-1].eps11>0:
			G,error=scipy.integrate.quad(lambda y: self.getEx(0,y), self.Interfaces[-1].z, np.inf)
			C+= G*self.Layers[-1].eps11*eps0
		return C
	def getA(self,n):
		if not n in self.adict:
			Pn=scipy.special.legendre(n)(2*self.k**2-1)
			self.adict[n]=pi/self.Kpk*Pn #=A_{2n+1}
		return self.adict[n]


	def getVExEy(self,x,z,getV=1,getEx=1,getEy=1): # accepts 'x' as a list, but 'y' must be single value
		if z==self.Interfaces[self.EleInt].z and getV==1 and getEx==0 and getEy==0:
			V=self.get_V_electrodes(x)
			return V, 0,0
		x=np.array(x)
		V=np.zeros(x.size)
		Ex=np.zeros(x.size)
		Ey=np.zeros(x.size)
		layer=0
		while layer<len(self.Interfaces) and self.Interfaces[layer].z <= z:
			layer+=1
		distPrior=0
		R=0
		if self.Interfaces[self.EleInt].z <= z: # we are above electrodes
			direction=1
			getPfield=self.getPFieldPos
			for i in range(self.EleInt+1,layer):
				distPrior+=self.Layers[i].tEff
			zEff=(z-self.Interfaces[layer-1].z)*self.Layers[layer].epsr+distPrior
			if len(self.Interfaces) > layer:
				zEffReverse=(self.Interfaces[layer].z-z)*self.Layers[layer].epsr+self.Layers[layer].tEff+distPrior
				R=1
		else: # we are below electrodes
			direction=-1
			getPfield=self.getPFieldNeg
			for i in range(self.EleInt,layer,direction):
				distPrior+=self.Layers[i].tEff
			zEff=direction*(z-self.Interfaces[layer].z)*self.Layers[layer].epsr+distPrior
			if layer>0:
				zEffReverse=direction*(self.Interfaces[layer-1].z-z)*self.Layers[layer].epsr+self.Layers[layer].tEff+distPrior
				R=1
		expf=-pi*zEff
		expMulFac=np.exp(expf*2)
		expM=np.exp(expf)
		N=1
		sincosf=pi*x
		if R==1:
			expfRev=-pi*zEffReverse
			expMulFacRev=np.exp(expfRev*2)
			expMRev=np.exp(expfRev)

		for n in range(self.maxsum):
			A2np1=self.getA(n)*getPfield(n)
			T=1
			if direction==1:
				for i in range(self.EleInt+1,layer):
					T=T*self.Interfaces[i].getTPosDirEff(n)
			else:
				for i in range(self.EleInt-1,layer-1,direction):
					T=T*self.Interfaces[i].getTNegDirEff(n)
			exp=expM*T
			if getV or getEy:
				sin=np.sin(N*sincosf)
				if getV:
					V+=-1/pi/N*A2np1*sin*exp
				if getEy:
					Ey+=direction*self.Layers[layer].epsr*A2np1*sin*exp
			if getEx:
				cos=np.cos(N*sincosf)
				Ex+=A2np1*cos*exp
			if exp/N<self.accuracyLimit: #this is a bit sketchy
				#print(N)
				break
			expM*=expMulFac
			R=0
			if direction == 1:
				if len(self.Interfaces) > layer:
					R=self.Interfaces[layer].getRPosDirEff(n) #pos dir
			else:
				if layer>0:
					R=self.Interfaces[layer-1].getRNegDirEff(n) #neg dir
			if not R==0:
				expRev=expMRev*T*R
				if getV or getEy:
					sin=np.sin(N*sincosf)
					if getV:
						V+=-1/pi/N*A2np1*sin*expRev
					if getEy:
						Ey-=direction*self.Layers[layer].epsr*A2np1*sin*expRev
				if getEx:
					cos=np.cos(N*sincosf)
					Ex+=A2np1*cos*expRev
				expMRev*=expMulFacRev
			N+=2
		return V, Ex, Ey

	def get_V_inf(self,x):
		if not x in self.Vdict:
			x=x+0.5
			if x%2>0.5:
				sign=-1
				x=1-x%2
			else:
				sign=1
				x=x%2
			z=x+1j*0
			t=1/self.k*np.sin(pi*z)
			F=fF(np.arcsin(t),self.k,self.accuracyLimit)
			V=sign*(1-(1/self.Kpk)*F.imag)*0.5
			self.Vdict[x]=V
		return self.Vdict[x]

	def get_V_electrodes(self,x): # accepts 'x' as a list
		x=np.array(x)
		V=np.zeros(x.size)
		for i, XX in enumerate(x):
			V[i]+=self.get_V_inf(XX)
		layer=self.EleInt+1
		#z=np.sum([o.z for o in self.Interfaces[:layer]])
		# we are above electrodes
		direction=1
		getPfield=self.getPFieldPos
		R=0
		if len(self.Interfaces) > layer:
			zEffReverse=2*self.Layers[layer].tEff#+distPrior
			R=1
		sincosf=pi*x
		if R==1:
			expfRev=-pi*zEffReverse
			expMulFacRev=np.exp(expfRev*2)
			expMRev=np.exp(expfRev)
		for n in range(self.maxsum):
			N=2*n+1
			P=getPfield(n)
			if (P-1)/N<self.accuracyLimit: #this is a bit sketchy
				#print(N)
				break
			A2np1=self.getA(n)
			sin=np.sin(N*sincosf)
			V+=-1/pi/N*A2np1*sin*(P-1)
			R=0
			if len(self.Interfaces) > layer:
				R=self.Interfaces[layer].getRPosDirEff(n) #pos dir
			if not R==0:
				expRev=expMRev*R
				sin=np.sin(N*sincosf)
				V+=-1/pi/N*A2np1*sin*expRev*P
				expMRev*=expMulFacRev
		return V

	def getV(self,x,y):
		V,Ex,Ey=self.getVExEy(x,y,1,0,0)
		return V
	def getEx(self,x,y):
		V,Ex,Ey=self.getVExEy(x,y,0,1,0)
		return Ex
	def getEy(self,x,y):
		V,Ex,Ey=self.getVExEy(x,y,0,0,1)
		return Ey


class multiple_recursive_images:
	def __init__(self,etas,t,epsx,epsy,LAcomp=8,maxsumelemtary=0, voltages = None,accuracyLimit=10**-10):
		# eta = vector of floats, t = vector of floats, epsx = vector of floats, epsy = vector of floats, LAcomp = int, maxsumelemtary = int
		# epsx and epsy refer to materials, the length must therefore be at least 2
		# eta refers to interfaces, the length must therefore be at least 1
		# t refers to the thicknes of layers of finite thickness. This vector may have 0 elements.
		# LAcomp must be an int, or a vector of same length as eta
		# voltages must be 'None' or a vector of same length as eta, and is uset to set voltage ratios if multiple sets of electrodes are used
		self.accuracyLimit=accuracyLimit
		self.etas=np.array(etas)
		#self.a=1-etas[0]
		#self.b=etas[0]
		self.t=np.array(t)
		self.epsx=np.array(epsx)
		self.epsy=np.array(epsy)
		self.LAcomp=LAcomp
		self.maxsumelemtary=maxsumelemtary #==0->single_recursive_imagess model
		self.voltages=voltages
		if self.voltages == None: # create voltages vector, all electrode sets should have V=0.5, unless they are continious, then they should have 0. If no electrodes are present at the interface, then the voltage is also set to 0
			self.voltages = []
			for eta in self.etas:
				if eta == 0:
					self.voltages.append(0)
				elif eta == 1:
					self.voltages.append(0)
				else:
					self.voltages.append(0.5)
		self.electrodesteps=[]
		if isinstance(LAcomp, (int,)):
			for eta in self.etas:
				self.electrodesteps.append(eta/self.LAcomp)
		else:
			for eta, LA in zip(self.etas, self.LAcomp):
				if LA>0:
					self.electrodesteps.append(eta/LA)
				else:
					self.electrodesteps.append(0)

		self.xpoints=[] #<- list of list of points of interest
		self.xpointVs=[]
		self.CaseEtas=[] #<- list of lists of eta of individual cases
		for eta, step, V in zip(self.etas, self.electrodesteps,self.voltages):
			self.xpoints.append([])
			self.CaseEtas.append([])
			l=step*0.5
			while l<eta and l<1-step:
				self.xpoints[-1].append(l/2.0-0.5)
				self.CaseEtas[-1].append(l+step*0.5)
				self.xpointVs.append(V)
				l+=step
		self.Cs=[]
		self.comp=[]
		self.unphys=[]
		self.Vs=[]
		self.C=-1
	def getC(self):
		if self.C==-1:
			for interface, _ in enumerate(self.CaseEtas):
				for  caseEta in self.CaseEtas[interface]:
					if len(self.unphys)==0:
						self.unphys.append(single_recursive_images(caseEta,interface,self.t,self.epsx,self.epsy,self.maxsumelemtary,self.accuracyLimit))
					else:
						self.unphys.append(single_recursive_images(caseEta,interface,self.t,self.epsx,self.epsy,self.maxsumelemtary,self.accuracyLimit, inherit=self.unphys[0]))
					self.Vs.append(np.array([]))
					z=0
					for interfaceOfXpoint, _ in enumerate(self.CaseEtas):
						if len(self.xpoints[interfaceOfXpoint])>0:
							self.Vs[-1]=np.concatenate((self.Vs[-1], self.unphys[-1].getV(self.xpoints[interfaceOfXpoint],z)))
						#for xpoint in self.xpoints[interfaceOfXpoint]:
						#	self.Vs[-1].append(self.unphys[-1].getV(xpoint,z))
						if interfaceOfXpoint<len(self.t):
							z+=self.t[interfaceOfXpoint]
					self.Cs.append(self.unphys[-1].getC())
			#print(np.array(self.Vs))
			A=np.array(self.Vs).transpose()
			B=np.array(self.xpointVs)
			self.comp=numpy.linalg.solve(A,B)
			self.C=np.dot(np.array(self.Cs),np.array(self.comp))
		return self.C
	def getVExEy(self,x,y,getV=1,getEx=1,getEy=1):
		if self.C==-1:
			self.getC()
		V=0
		Ex=0
		Ey=0
		for case, component in zip(self.unphys, self.comp):
			Vi,Exi,Eyi=case.getVExEy(x,y,getV,getEx,getEy)
			V+=Vi*component
			#print(Vi, component)
			Ex+=Exi*component
			Ey+=Eyi*component
		return(V,Ex,Ey)
	def getV(self,x,y):
		V,Ex,Ey=self.getVExEy(x,y,1,0,0)
		return V
	def getEx(self,x,y):
		V,Ex,Ey=self.getVExEy(x,y,0,1,0)
		return Ex
	def getEy(self,x,y):
		V,Ex,Ey=self.getVExEy(x,y,0,0,1)
		return Ey


from scipy.optimize import curve_fit
def getC_fit(x,xi):
	#print(x,xi)
	case=multiple_recursive_images(x[0],x[1],[x[2]],[0,xi,1],[0,x[3],1],int(x[4]),int(x[5]),int(x[6]))
	C=case.getC()
	print(C,xi)
	return(C)
def refineepsx1(C,a,b,t,epsx,epsy,LAcomp,maxsumelemtary):
	#(self,alpha,beta,C,xi=2,maxn=8,steps=5):
	x=[a,b,t,epsy[1],LAcomp,maxsumelemtary]
	#print(x)
	popt,pcov=curve_fit(getC_fit,x,C,p0=epsx[1])
	eps=popt[0]
	return eps
def getC_fitX(x,xi):
	#print(x,xi)
	case=multiple_recursive_images(x[0],x[1],[x[2]],[0,x[3],1],[0,xi,1],int(x[4]),int(x[5]),int(x[6]))
	C=case.getC()
	print(C,xi)
	return(C)
def refineepsy1(C,a,b,t,epsx,epsy,LAcomp,maxsumelemtary):
	#(self,alpha,beta,C,xi=2,maxn=8,steps=5):
	x=[a,b,t,epsx[1],LAcomp,maxsumelemtary]
	#print(x)
	popt,pcov=curve_fit(getC_fitX,x,C,p0=epsx[1])
	eps=popt[0]
	return eps
