from env import *
import numpy as np
import ParHierModel
import sklearn.metrics
import pandas as pd

def convert_csv(file):
  table = pd.read_csv(file)

  print('table', table)

  return convert_table_marco(table)

def convert_table_marco(table, biomkStartCol=3, list_biomarkers=None):

    # list of individuals
    list_RID = np.unique(table[['RID']])
    print(len(list_RID))
    # list of biomarkers
    if list_biomarkers is None:
      list_biomarkers = table.columns[range(biomkStartCol, len(table.columns))]

    RID = []
    X = [[] for _ in range(len(list_biomarkers))]
    Y = [[] for _ in range(len(list_biomarkers))]
    visitIndices = [[0 for _ in range(len(list_RID))] for _ in range(len(list_biomarkers))]

    # Parsing every biomarker and assigning to the list
    print('list_RID', list_RID)
    diagLong = []
    # print(adsas)
    for id_sub, sub in enumerate(list_RID):
      indices = np.where(np.in1d(table.loc[:, 'RID'], sub))[0]
      for id_biom, biomarker in enumerate(list_biomarkers):
        X[id_biom].append(np.array(table[['Month_bl']])[indices].flatten())
        Y[id_biom].append(np.array(table[[biomarker]])[indices].flatten())

        idx_to_keep = ~np.isnan(Y[id_biom][id_sub])
        visitIndices[id_biom][id_sub] = np.array(range(Y[id_biom][id_sub].shape[0]))[idx_to_keep]

        Y[id_biom][id_sub] = Y[id_biom][id_sub][idx_to_keep]
        X[id_biom][id_sub] = X[id_biom][id_sub][idx_to_keep]


      diagCurrSub = np.array(table['diag'][indices])
      diagNNind = np.logical_not(np.isnan(diagCurrSub))
      # print(diagCurrSub)
      monthsSinceBlCurrSub = np.array(table['Month_bl'])[indices][diagNNind]


      diagExistsForAtLeastOneVisit = monthsSinceBlCurrSub.shape[0] > 0
      if diagExistsForAtLeastOneVisit:
        RID.append(sub)
        currDiag = diagCurrSub[diagNNind][np.argmin(monthsSinceBlCurrSub)]
        diagLong.append(currDiag)

    Xtrain = []
    Ytrain = []
    visitIndicesTrain = []

    for id_biom, biomarker in enumerate(list_biomarkers):
      Xtrain.append([])
      Ytrain.append([])
      visitIndicesTrain.append([])

    for id_sub, sub in enumerate(list_RID):
      # print('sub', sub, 'RID', RID)
      # print(asda)
      if np.in1d(sub, RID)[0]:
        for id_biom, biomarker in enumerate(list_biomarkers):
          Xtrain[id_biom].append(X[id_biom][id_sub])
          Ytrain[id_biom].append(Y[id_biom][id_sub])
          visitIndicesTrain[id_biom].append(visitIndices[id_biom][id_sub])
    # print('visitIndicesTrain', visitIndicesTrain)
    # print(asdasd)

    # # some entries with duplicate Xs are in the tadpole dataset. Remove them.
    # for b in range(len(Xtrain)):
    #   for s in range(len(Xtrain[0])):
    #     unqVals, unqInd = np.unique(Xtrain[b][s], return_index=True)
    #     if unqVals.shape[0] < Xtrain[b][s].shape[0]:
    #       Xtrain[b][s] = Xtrain[b][s][unqInd]
    #       Ytrain[b][s] = Ytrain[b][s][unqInd]
    #       visitIndicesTrain[b][s] = visitIndicesTrain[b][s][unqInd]
    #       print('RID', RID[s])
    #       print('Xtrain[b][s]', Xtrain[b][s])
    #       print('unqInd', unqInd)
    #       print('unqVals', unqVals)
    #       print('visitIndicesTrain[b][s]', visitIndicesTrain[b][s])
    #       visitIndicesTrain[b][s] = np.argsort(np.argsort(visitIndicesTrain[b][s]))
    #       print('visitIndicesTrain[b][s]', visitIndicesTrain[b][s])
    #       assert Xtrain[b][s].shape[0] == Ytrain[b][s].shape[0]
    #       raise ValueError('duplicate XS found')
    #
    # for b in range(len(Xtrain)):
    #   for s in range(len(Xtrain[0])):
    #     unqVals, unqInd = np.unique(Xtrain[b][s], return_index=True)
    #     if unqVals.shape[0] < Xtrain[b][s].shape[0]:
    #       raise ValueError('duplicate XS found')


    # print(asda)

    return Xtrain, Ytrain, np.array(RID), list_biomarkers, np.array(diagLong), visitIndicesTrain

def makeShiftsIdentif(subShiftsCross, ageAtVisitCross, crossDiag, ctlDiagNr, patDiagNr):
  # set origin t=0 as best threshold that separates the two diagnostic histograms

  dpsCross = subShiftsCross + ageAtVisitCross
  nrSegments = 100
  dpsList = np.linspace(np.min(dpsCross), np.max(dpsCross),nrSegments)
  accuraciesPerDPS = np.zeros(nrSegments)

  predDiagCurr = np.zeros(dpsCross.shape[0])

  for d in range(nrSegments):
    predDiagCurr[dpsCross < dpsList[d]] = ctlDiagNr
    predDiagCurr[dpsCross > dpsList[d]] = patDiagNr

    accuraciesPerDPS[d] = sklearn.metrics.accuracy_score(predDiagCurr, crossDiag)

  # find the disease progression scores that best separated the two diagnostic histograms
  diagSeparatingDPS = dpsList[np.argmax(accuraciesPerDPS)]
  print('diagSeparatingDPS', diagSeparatingDPS)
  # print(asas)
  dpsCrossNew = dpsCross - diagSeparatingDPS
  subShiftsCrossNew = dpsCrossNew - ageAtVisitCross

  shiftTransform = subShiftsCrossNew[0] - subShiftsCross[0]

  print('diff', subShiftsCrossNew-subShiftsCross )
  print('subShiftsCross', subShiftsCross)
  assert ((subShiftsCrossNew - subShiftsCross) - (subShiftsCrossNew[0] - subShiftsCross[0]) < 0.001).all()

  return subShiftsCrossNew, shiftTransform

def makeLongFromCross(array, cross2longInd):
  return [array[idx] for idx in cross2longInd]

def sigmoidFunc(s, theta):
  """
  sigmoidal function for trectory with params [a,b,c,d] with
  minimum d, maximum a+d, slope a*b/4 and slope
  maximum attained at center c
  f(s|theta = [a,b,c,d]) = a/(1+exp(-b(s-c)))+d

  :param s: the inputs and can be an array of dim N x 1
  :param theta: parameters as np.array([a b c d])
  :return: values of the sigmoid function at the inputs s
  """

  return theta[0] * np.power((1 + np.exp(-theta[1] * (s - theta[2]))), -1) + theta[3]

def createLongData(data, diag, scanTimepts, partCode, yearsSinceBlScan):

  uniquePartCode = np.unique(partCode)

  longData = ParHierModel.ParHierModel.makeLongArray(data, scanTimepts, partCode, uniquePartCode)
  longDiagAllTmpts = ParHierModel.ParHierModel.makeLongArray(diag, scanTimepts, partCode, uniquePartCode)
  longDiag = np.array([x[0] for x in longDiagAllTmpts])
  longScanTimepts = ParHierModel.ParHierModel.makeLongArray(scanTimepts, scanTimepts, partCode, uniquePartCode)
  longPartCodeAllTimepts = ParHierModel.ParHierModel.makeLongArray(partCode, scanTimepts, partCode, uniquePartCode)
  longPartCode = np.array([x[0] for x in longPartCodeAllTimepts])
  longAgeAtScan = ParHierModel.ParHierModel.makeLongArray(yearsSinceBlScan, scanTimepts, partCode, uniquePartCode)
  uniquePartCodeFiltIndices = np.in1d(partCode, np.array(longPartCode))

  # filter cross-sectional data, keep only subjects with at least 2 visits
  filtData = data[uniquePartCodeFiltIndices,:]
  filtDiag = diag[uniquePartCodeFiltIndices]
  filtScanTimetps = scanTimepts[uniquePartCodeFiltIndices]
  filtPartCode = partCode[uniquePartCodeFiltIndices]
  filtAgeAtScan = yearsSinceBlScan[uniquePartCodeFiltIndices]
  inverseMap = np.squeeze(np.array([np.where(longPartCode == p) for p in filtPartCode])) # maps from longitudinal space
  #  to cross-sectional space

  assert(np.max(inverseMap) == len(longData)-1) # inverseMap indices should be smaller than the size of longData as they take elements from longData
  assert(len(inverseMap) == filtData.shape[0]) # length of inversemap should be the same as the cross-sectional data

  #print(np.max(inverseMap), len(longData), len(inverseMap), inverseMap.shape)
  #print(test)

  return longData, longDiagAllTmpts, longDiag, longScanTimepts, longPartCode, longAgeAtScan, inverseMap, filtData, filtDiag, filtScanTimetps, filtPartCode, filtAgeAtScan


def filterDataListFormat(params, dataIndices):
  dataIndNonBinary = np.where(dataIndices)[0]
  nrBiomk = len(params['X'])

  Xfilt = [0 for x in range(nrBiomk)]
  Yfilt = [0 for x in range(nrBiomk)]
  visitIndicesFilt = [0 for x in range(nrBiomk)]

  for b in range(nrBiomk):
    Xfilt[b] = [params['X'][b][i] for i in dataIndNonBinary]
    Yfilt[b] = [params['Y'][b][i] for i in dataIndNonBinary]
    visitIndicesFilt[b] = [params['visitIndices'][b][i] for i in dataIndNonBinary]

  return Xfilt, Yfilt, visitIndicesFilt


def getGammShapeRateFromTranTime(transitionTimePriorMean, transitionTimePriorMin, transitionTimePriorMax):

  bPriorMean = 16 / (1 * transitionTimePriorMean)
  bPriorStd = np.abs(16 / (1 * transitionTimePriorMax) - 16 / (1 * transitionTimePriorMin))

  bPriorShape = (bPriorMean ** 2) / (bPriorStd ** 2)
  bPriorRate = bPriorMean / (bPriorStd ** 2)

  return bPriorShape, bPriorRate

def getMeanStdBFromTranTime(transitionTimePriorMean, transitionTimePriorMin, transitionTimePriorMax):

  bPriorMean = 16 / (1 * transitionTimePriorMean)
  bPriorStd = np.abs(16 / (1 * transitionTimePriorMax) - 16 / (1 * transitionTimePriorMin))


  return bPriorMean, bPriorStd

def applyScalingToBiomk(dataCrossSB, scalingBiomk2B):
  scaledData = dataCrossSB * scalingBiomk2B[1,:][None, :] + scalingBiomk2B[0,:][None, :]
  return scaledData

def applyInverseScalingToBiomk(dataCrossSB, scalingBiomk2B):
  scaledData = (dataCrossSB - scalingBiomk2B[0,:][None, :]) / scalingBiomk2B[1,:][None, :]
  return scaledData

def findOptimalRowsCols(nrBiomk):
  potentialLayout = [(2, 2), (2, 3), (3, 3), (3, 4), (4, 4), (4, 5), (4, 6), (5, 6), (5,7), (6,7), (7,8), (7,9), (7,10), (8,11)]

  optimalRows, optimalCols = ([potentialLayout[i] for i in range(len(potentialLayout)) if
                               potentialLayout[i][0] * potentialLayout[i][1] >= nrBiomk])[0]

  return optimalRows, optimalCols
