import numpy as np
import scipy as sp

class ParHierModel:

  def __init__(self, dysfuncParams, thetas, mapBiomkToFuncUnits, parFunc, sigmaB):
    self.dysfuncParams = dysfuncParams
    self.thetas = thetas
    self.mapBiomkToFuncUnits = mapBiomkToFuncUnits
    self.parFunc = parFunc # parametric function
    self.sigmaB = sigmaB
    self.nrBiomk = thetas.shape[0]
    self.nrFuncUnits = np.unique(mapBiomkToFuncUnits).shape[0]

    # self.subShiftsLongTrue = subShiftsLongTrue
    # self.subShiftsCrossTrue = subShiftsLongTrue[long2crossInd]


  def timeShiftFuncCross(self, ageAtVisitCross):

    return ageAtVisitCross + self.subShiftsCrossTrue


  def predPopBiomk(self, dpsCross, b):
    '''
    Predict population-level effect f_b (X) evaluated at dps (X) for biomarker b

    :param dpsCross: 1D disease progression scores for all subjects and visits, or X
    :param b: biomarker nr
    :return: Y = f_b (X)
    '''


    # find dysfunction scores for each subject
    dysScoresS = self.parFunc(dpsCross, self.dysfuncParams[self.mapBiomkToFuncUnits[b],:])
    modelPredS = self.parFunc(dysScoresS, self.thetas[b,:])

    # print('self.dysfuncParams', self.dysfuncParams)
    # print('dpsCross', dpsCross)
    # print('dysScoresS', dysScoresS)
    # print('params', self.dysfuncParams[self.mapBiomkToFuncUnits[b],:])
    # print('modelPredS', modelPredS)
    # print(ads)

    return modelPredS

  def predPopDys(self, dpsCross):
    '''
    Predict population-level effect f_b (X) evaluated at dps (X) for biomarker b

    :param dpsCross: 1D disease progression scores for all subjects and visits, or X
    :param b: biomarker nr
    :return: Y = f_b (X)
    '''

    dysScoresSU = np.zeros((dpsCross.shape[0], self.nrFuncUnits), float)

    # find dysfunction scores for each subject
    for u in range(self.nrFuncUnits):
      dysScoresSU[:,u] = self.parFunc(dpsCross, self.dysfuncParams[u,:])

    assert dysScoresSU.shape[0] == dpsCross.shape[0]
    assert dysScoresSU.shape[1] == self.nrFuncUnits

    return dysScoresSU

  def predPopFromDysfunc(self, dysScoresS):
    '''
    Predict population-level effect f_b (X) evaluated at dps (X) for biomarker b

    :param dpsCross: 1D disease progression scores for all subjects and visits, or X
    :param b: biomarker nr
    :return: Y = f_b (X)
    '''

    modelPredSB = np.zeros((dysScoresS.shape[0], self.nrBiomk), float)

    # find dysfunction scores for each subject
    for b in range(self.nrBiomk):
      modelPredSB[:,b] = self.parFunc(dysScoresS, self.thetas[b, :])

    return modelPredSB

  def predPop(self, dpsCross):
    '''
    Predict population-level effect f(X) = [f_1(X) .. f_n(X)] evaluated at dps (X) for all biomarkers

    :param dpsCross: 1D disease progression scores for all subjects and visits, or X
    :return: Y = f(X)
    '''

    modelPredSB = np.zeros((dpsCross.shape[0], self.nrBiomk), float)

    dysScoresSF = self.predPopDys(dpsCross)

    for b in range(self.nrBiomk):
      f = self.mapBiomkToFuncUnits[b]
      print(dysScoresSF.shape)
      print(self.thetas[b, :].shape)
      modelPredSB[:,b] = self.parFunc(dysScoresSF[:,f], self.thetas[b, :])

    assert modelPredSB.shape[0] == dpsCross.shape[0]
    assert modelPredSB.shape[1] == self.nrBiomk

    return modelPredSB

  def genDataIID(self, dpsCross):
    dataCrossSB = np.zeros((dpsCross.shape[0], self.nrBiomk), float)

    for b in range(self.nrBiomk):
      dataCrossSB[:,b] = self.predPopBiomk(dpsCross, b) + np.random.normal(0,self.sigmaB[b], dpsCross.shape[0])

    assert dataCrossSB.shape[0] == dpsCross.shape[0]
    assert dataCrossSB.shape[1] == self.nrBiomk

    return dataCrossSB

  @staticmethod
  def makeLongArray(array, scanTimepts, partCode, uniquePartCode):
    # place data in a longitudinal format
    longArray = [] # longArray can be data, diag, ageAtScan,scanTimepts, etc .. both 1D or 2D
    nrParticipants = len(uniquePartCode)

    longCounter = 0

    for p in range(nrParticipants):
      # print('Participant %d' % uniquePartCode[p])
      currPartIndices = np.where(partCode == uniquePartCode[p])[0]
      currPartTimepoints = scanTimepts[currPartIndices]
      currPartTimeptsOrdInd = np.argsort(currPartTimepoints)
      # print uniquePartCode[p], currPartIndices, currPartTimepoints, currPartTimeptsOrdInd
      currPartIndicesOrd = currPartIndices[currPartTimeptsOrdInd]
      # print(uniquePartCode[p], currPartIndicesOrd)

      # assert(len(currPartTimeptsOrdInd) >= 2) # 2 for PET, 3 for MRI

      # if len(currPartTimeptsOrdInd) > 1:
      longArray += [array[currPartIndicesOrd]]

    return longArray