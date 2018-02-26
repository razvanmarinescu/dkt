import numpy as np
import scipy as sp

class GPModel:

  def __init__(self, params):
    self.etaB = params['etaB']
    self.lB = params['lB']
    self.epsB = params['epsB']
    self.sigmaSB = params['sigmaSB']

    self.sigmaGfunc = params['sigmaGfunc']
    self.nrBiomk = len(params['etaB'])


  def predPopBiomk(self, dps, b):
    '''
    Predict population-level effect f_b (X) evaluated at dps (X) for biomarker b

    :param dps: 1D disease progression scores for all subjects and visits, or X
    :param b: biomarker nr
    :return: Y = f_b (X)
    '''

    sigmaGdps = self.sigmaGfunc(dps, nuB[b], lB[b])
    print(sigmaGdps.shape)
    meanGP = np.zeros(sigmaGdps.shape[0], float) # set mean of GP to zero
    fDps = np.random.multivariate_normal(meanGP, sigmaGdps)

    return fDps

  def predPop(self, dps):
    '''
    Predict population-level effect f(X) = [f_1(X) .. f_n(X)] evaluated at dps (X) for all biomarkers

    :param dps: 1D disease progression scores for all subjects and visits, or X
    :return: Y = f(X)
    '''

    fDpsCrossSB = np.zeros((dps.shape[0], self.nrBiomk), float)

    for b in range(self.nrBiomk):
      fDpsCrossSB = self.predPopBiomk(dps, b)

    return fDpsCrossSB

  def predIndiv(self, dps):
    '''
    Predict individual-level effect nu(X)  evaluated at dps (X) for all biomarkers

    :param dps: 1D disease progression scores for all subjects and visits, or X
    :return: Y = f(X)
    '''

    return 0

  def pred(self, dps):
    '''
    Predict full model f(X)+nu(X) evaluated at dps (X), for all biomarkers

    :param dps: 1D disease progression scores for all subjects and visits, or X
    :return: Y = f(X) + nu(X)
    '''

    return self.predPop(dps) + self.predIndiv(dps)


def genSigmaG(ts, nuB, lb):

  dists = sp.spatial.distance.squareform(sp.spatial.distance.pdist(ts.reshape(-1,1), 'euclidean'))
  print('dists.shape',dists.shape)

  return nuB * np.exp(dists / (2* (lb**2)))

