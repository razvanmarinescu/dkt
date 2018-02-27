import numpy as np
import os
import pickle
from GPModel import *
from env import *
import scipy
import scipy.stats
from auxFunc import *
import ParHierModel
import pandas as pd

import MarcoModel
import auxFunc


def generateDataJMD(nrSubjLong, nrBiomk, nrTimepts, shiftsLowerLim, shiftsUpperLim, model,
  outFolder, fileName, forceRegenerate, localParams, scalingBiomk2B, ctlDiagNr, patDiagNr):
  ''' generates data from a hierarchical model of disease '''


  if os.path.isfile(fileName) and not forceRegenerate:
    dataStruct = pickle.load(open(fileName, 'rb'))
    localParams = dataStruct['localParams']

  else:
    np.random.seed(1)
    # generate subject data
    subShiftsLongTrue = np.random.uniform(shiftsLowerLim, shiftsUpperLim, (nrSubjLong,))
    nrSubjCross = nrTimepts * nrSubjLong

    # ageAtBlScanLong = np.random.uniform(lowerAgeLim,upperAgeLim, (nrSubjLong,))
    # ageAtScanCross = np.zeros(nrSubjCross, float)
    # ageAtBlScanCross = np.zeros(nrSubjCross, float)
    yearsSinceBlScanCross = np.zeros(nrSubjCross, float)

    dataCrossSB = np.zeros((nrSubjCross, nrBiomk), float)
    subShiftsCrossTrue = np.zeros(nrSubjCross, float)

    partCodeCross = np.zeros(nrSubjCross, float)
    partCodeLong = np.array(range(nrSubjLong)) # unique id for every participant
    scanTimeptsCross = np.zeros(nrSubjCross, float)

    counter = 0
    long2crossInd = np.zeros(nrSubjCross, int)
    cross2longInd = [0 for s in range(nrSubjLong)]

    for s in range(nrSubjLong):

      cross2longInd[s] = np.array(range(counter,counter+nrTimepts))

      for tp in range(nrTimepts):
        # get currTimept, age at curr Timepints, and partCodeCross
        partCodeCross[counter] = partCodeLong[s]
        scanTimeptsCross[counter] = tp
        yearsSinceBlScanCross[counter] = tp
        # ageAtScanCross[counter] = ageAtBlScanLong[s] + tp # add one year at each timepoint
        # ageAtBlScanCross[counter] = ageAtBlScanLong[s] # keep baseline age even for followups
        subShiftsCrossTrue[counter] = subShiftsLongTrue[s]
        long2crossInd[counter] = s

        counter += 1

    # yearsSinceBlScan = ageAtScanCross - ageAtBlScanCross

    # generate data - find dps from age
    dpsCross = yearsSinceBlScanCross + subShiftsCrossTrue  # disease progression score
    # dpsLongSV = [dpsCross[cross2longInd[s]] for s in range(nrSubjLong)]

    diagCross = generateDiag(dpsCross, ctlDiagNr=ctlDiagNr, patDiagNr=patDiagNr)
    print('diagCross', diagCross)
    assert np.unique(diagCross).shape[0] >= 2
    # print(adsa)

    print('subShiftsCrossTrue', subShiftsCrossTrue)
    # make the shifts identifiable. set origin dps=0 as the line that best separates CTL vs AD
    subShiftsCrossTrue, shiftTransform = makeShiftsIdentif(
      subShiftsCrossTrue, yearsSinceBlScanCross, diagCross, ctlDiagNr=ctlDiagNr, patDiagNr=patDiagNr)

    print('subShiftsCrossTrue', subShiftsCrossTrue)
    # print(asa)

    subShiftsLongTrue = makeLongFromCross(subShiftsCrossTrue, cross2longInd)
    yearsSinceBlScanLong = makeLongFromCross(yearsSinceBlScanCross, cross2longInd)
    dpsCross = yearsSinceBlScanCross + subShiftsCrossTrue
    dpsLongSV = makeLongFromCross(dpsCross, cross2longInd)

    # print('dpsCross', dpsCross)
    # print('dpsRange', np.min(dpsCross), np.max(dpsCross))
    # print('subShiftsLongTrue', subShiftsLongTrue)
    # print(asds)

    # trueParams = dict(subShiftsLongTrue=subShiftsLongTrue,
    #   subShiftsCrossTrue=subShiftsCrossTrue, dpsLongSV=dpsLongSV, dpsCross=dpsCross)

    #### now generate the actual biomarker data #######
    dataCrossSB = model.genDataIID(dpsCross)
    dataCrossSB = auxFunc.applyScalingToBiomk(dataCrossSB, scalingBiomk2B)

    assert (not np.any(np.isnan(dataCrossSB)))

    labels = ['biomk %d' % d for d in range(nrBiomk)]

    longData, longDiagAllTmpts, longDiag, longScanTimepts, longPartCode, longAgeAtScan, \
      inverseMap, filtData, filtDiag, filtScanTimetps, filtPartCode, filtYearsSinceBlScanCross, \
      = createLongData(dataCrossSB, diagCross, scanTimeptsCross,
      partCodeCross, yearsSinceBlScanCross)


    # localParams['data'] = dataCrossSB
    # localParams['scanTimepts'] = scanTimeptsCross
    # localParams['partCode'] = partCodeCross
    # localParams['ageAtScan'] = ageAtScanCross
    # localParams['ageAtBlScan'] = ageAtBlScanCross


    # localParams['longData'] = longData
    # localParams['longDiag'] = longDiag
    # localParams['longScanTimepts'] = longScanTimepts
    # localParams['longPartCode'] = longPartCode
    # localParams['longAgeAtScan'] = longAgeAtScan
    # localParams['inverseMap'] = inverseMap

    # localParams['trueParamsMarcoFormat'] = trueParamsMarcoFormat

    # monthsSinceBlScan = 12*(ageAtScanCross - ageAtBlScanCross)


    # put everything in Marco's format.
    # X - list of length NR_BIOMK.  X[b] - list of NR_SUBJ_LONG   X[b][s] - list of visit months for subject b and biomarker s
    # Y - list of length NR_BIOMK.  Y[b] - list of NR_SUBJ_LONG   Y[b][s] - list of biomarker values for subject b and biomarker s
    # RID - list of length NR_SUBJ_LONG
    X, Y, RID = convertToMarcoFormat(dataCrossSB, labels, yearsSinceBlScanCross, partCodeCross, diagCross)

    localParams['X'] = X
    localParams['Y'] = Y
    localParams['RID'] = RID
    localParams['labels'] = labels

    print('RID', RID)
    print('X[0][1]', X[0][1])
    diagMarcoFormat = np.zeros(RID.shape)
    subShiftsTrueMarcoFormatS = np.zeros(RID.shape)
    subShiftsLongTrue1D = np.array([x[0] for x in subShiftsLongTrue])

    for r in range(len(RID)):
      print(partCodeCross == RID[r])
      print(diagCross[partCodeCross == RID[r]])
      diagMarcoFormat[r] = diagCross[partCodeCross == RID[r]][0]
      subShiftsTrueMarcoFormatS[r] = subShiftsLongTrue1D[partCodeLong == RID[r]]

    localParams['diag'] = diagMarcoFormat

    # disease agnostic
    trueDysfuncXsX = np.linspace(0, 1, num=50)
    trueTrajFromDysXB = model.predPopFromDysfunc(trueDysfuncXsX)
    trueTrajFromDysXB = auxFunc.applyScalingToBiomk(trueTrajFromDysXB, scalingBiomk2B)
    trueLineSpacedDPSsX = np.linspace(np.min(dpsCross), np.max(dpsCross), num=50)


    # disease specific
    trueTrajPredXB = model.predPop(trueLineSpacedDPSsX)
    trueTrajPredXB = auxFunc.applyScalingToBiomk(trueTrajPredXB, scalingBiomk2B)
    trueDysTrajFromDpsXU = model.predPopDys(trueLineSpacedDPSsX)
    trueSubjDysfuncScoresSU = model.predPopDys(subShiftsTrueMarcoFormatS)


    trueParamsMarcoFormat = dict(subShiftsTrueMarcoFormatS=subShiftsTrueMarcoFormatS,
    trueSubjDysfuncScoresSU=trueSubjDysfuncScoresSU, trueLineSpacedDPSsX=trueLineSpacedDPSsX,
      trueTrajPredXB=trueTrajPredXB, trueDysTrajFromDpsXU=trueDysTrajFromDpsXU,
      trueDysfuncXsX=trueDysfuncXsX,trueTrajFromDysXB=trueTrajFromDysXB,
      scalingBiomk2B=scalingBiomk2B)
    localParams['trueParams'] = trueParamsMarcoFormat

    os.system('mkdir -p %s' % outFolder)
    outFileFull = '%s/%s' % (outFolder, fileName)
    pickle.dump(localParams, open(outFileFull, 'wb'), protocol=pickle.HIGHEST_PROTOCOL)



  return localParams

def convertToMarcoFormat(data, labels, yearsSinceBlScan, partCode, diag):
  df = pd.DataFrame(data,columns = labels)
  df.insert(0, 'diag', diag)
  df.insert(0, 'Month_bl', yearsSinceBlScan)
  df.insert(0, 'RID', partCode)
  df.insert(0, 'SUB', np.array(range(data.shape[0])))

  X,Y,RID,list_biomarkers, diag = auxFunc.convert_table_marco(df, list_biomarkers=labels)

  return X,Y,np.array(RID)

def generateDiag(dpsCross, ctlDiagNr, patDiagNr, diagPrecDef = 0.4, muScale = 1):
  nrSubjCross = dpsCross.shape[0]
  controlDiagPrec = diagPrecDef
  patientDiagPrec = diagPrecDef
  minDps = np.min(dpsCross)
  maxDps = np.max(dpsCross)
  #dpsUpperLim = upperAgeLim # after this dps limit limit almost all of diags will be patient
  # precision values they cannot be 1(perfect recision) as the exponential distribution is not well - defined anymore
  assert (controlDiagPrec != 1 and patientDiagPrec != 1)

  # multiplying the mean with nrTimepts scales perfectly to more biomk, tested on 18 / 03 / 2016
  muExpoCTL = minDps + muScale * (maxDps - minDps) * (1 - controlDiagPrec**(1 / 2))
  muExpoPAT = minDps + muScale * (maxDps - minDps) * (1 - patientDiagPrec**(1 / 2))
  diagCross = ctlDiagNr * np.ones(nrSubjCross, int)
  probControl = np.zeros(nrSubjCross, float)
  for s in range(nrSubjCross):
    # generate diag
    dpsCurr = dpsCross[s]
    probControl[s] = calcProbControlFromExpo(dpsCurr, muExpoCTL, muExpoPAT, minDps, maxDps)

    if np.random.rand(1, 1) > probControl[s]:
      diagCross[s] = patDiagNr

  # plot probControl over dps's
  nrStages = 100
  stageRange = np.linspace(minDps, maxDps, nrStages)
  probControlStages = np.zeros(nrStages, float)
  for st in range(nrStages):
    probControlStages[st] = calcProbControlFromExpo(stageRange[st], muExpoCTL, muExpoPAT, minDps, maxDps)

  assert not np.isnan(probControl).any()

  # print('dpsCross', dpsCross)
  # print('probControl', probControl)
  # print(stageRange, probControlStages)
  # print(muExpoCTL, muExpoPAT)
  # print(minDps, maxDps)
  # pl.plot(stageRange, probControlStages)
  # pl.show()

  return diagCross

def calcProbControlFromExpo(stage, muExpoCTL, muExpoPAT, stageLowerLim, stageUpperLim):

  probControl = scipy.stats.expon.pdf(stage-stageLowerLim, scale=muExpoCTL-stageLowerLim) / \
        (scipy.stats.expon.pdf(stage-stageLowerLim, scale=muExpoCTL-stageLowerLim) +
         scipy.stats.expon.pdf(stageUpperLim - stage, scale=muExpoPAT-stageLowerLim))

  return probControl

