
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
    X = []
    Y = []

    # list of individuals
    list_RID = np.unique(table[['RID']])
    print(len(list_RID))
    # list of biomarkers
    if list_biomarkers is None:
      list_biomarkers = table.columns[range(biomkStartCol, len(table.columns))]

    RID = []

    for id_biom, biomarker in enumerate(list_biomarkers):
        X.append([])
        Y.append([])

    # Parsing every biomarker and assigning to the list
    print('list_RID', list_RID)
    diagLong = []
    # print(adsas)
    for id_sub, sub in enumerate(list_RID):
        flag_missing = 0
        indices = np.where(np.in1d(table.loc[:, 'RID'], sub))[0]
        for id_biom, biomarker in enumerate(list_biomarkers):
            X[id_biom].append(np.array(table[['Month_bl']])[indices].flatten())
            Y[id_biom].append(np.array(table[[biomarker]])[indices].flatten())

            idx_to_keep = ~np.isnan(Y[id_biom][id_sub])

            Y[id_biom][id_sub] = Y[id_biom][id_sub][idx_to_keep]
            X[id_biom][id_sub] = X[id_biom][id_sub][idx_to_keep]

            if len(Y[id_biom][id_sub]) < 1:
                flag_missing = flag_missing + 1


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

    for id_biom, biomarker in enumerate(list_biomarkers):
        Xtrain.append([])
        Ytrain.append([])

    for id_sub, sub in enumerate(list_RID):
        if np.in1d(sub, RID)[0]:
            for id_biom, biomarker in enumerate(list_biomarkers):
                Xtrain[id_biom].append(X[id_biom][id_sub])
                Ytrain[id_biom].append(Y[id_biom][id_sub])
    # print(len(RID), RID)
    # print(asdasd)
    return Xtrain, Ytrain, np.array(RID), list_biomarkers, np.array(diagLong)

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

  for b in range(nrBiomk):
    Xfilt[b] = [params['X'][b][i] for i in dataIndNonBinary]
    Yfilt[b] = [params['Y'][b][i] for i in dataIndNonBinary]

  return Xfilt, Yfilt


def applyScalingToBiomk(dataCrossSB, scalingBiomk2B):
  scaledData = dataCrossSB * scalingBiomk2B[1,:][None, :] + scalingBiomk2B[0,:][None, :]
  return scaledData

def applyInverseScalingToBiomk(dataCrossSB, scalingBiomk2B):
  scaledData = (dataCrossSB - scalingBiomk2B[0,:][None, :]) / scalingBiomk2B[1,:][None, :]
  return scaledData

def findOptimalRowsCols(nrBiomk):
  potentialLayout = [(2, 2), (2, 3), (3, 3), (3, 4), (4, 4), (4, 5), (4, 6), (5, 6), (5,7), (6,7)]

  optimalRows, optimalCols = ([potentialLayout[i] for i in range(len(potentialLayout)) if
                               potentialLayout[i][0] * potentialLayout[i][1] >= nrBiomk])[0]

  return optimalRows, optimalCols
