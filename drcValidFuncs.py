from env import *
import pandas as pd
import numpy as np
from matplotlib import pyplot as pl
import sklearn
import copy
import scipy


def addDRCValidData(validDf):
  '''perform validation on DTI data from the DRC '''

  #dtiSS = pd.read_csv('../data/DRC/DTI/DTI_summary_forRaz.xlsx')
  dtiSS = pd.read_csv('DTI_summary_forRaz.csv')
  mappingIDtoRegion = {0 : ["Unclassified", "UNC"] ,
    1: ["Middle cerebellar peduncle", "ICP"], # TBC
    2: ["Pontine Crossing tract","PCT"], # TBC
    3: ["Genu of corpus callosum", "GCC"],# cingulate or frontal
    4: ["Body of corpus callosum", "BCC"],# cingulate or None
    5: ["Splenium of corpus callosum", "SCC"],  # cingulate or occipital
    6: ["Fornix (column and body of fornix)","FX"], # hippo
    7: ["Corticospinal tract R", "CST"],# frontal or None (wiki says 80-90% go to motor ctx from brain stem)
    8: ["Corticospinal tract L", "CST"],# frontal
    9: ["Medial lemniscus R", "ML"],## TBC
    10: ["Medial lemniscus L","ML"], ## TBC
    11: ["Inferior cerebellar peduncle R", "ICP"],  ## TBC
    12: ["Inferior cerebellar peduncle L", "ICP"],  ## TBC
    13: ["Superior cerebellar peduncle R", "SCP"],  ## TBC
    14: ["Superior cerebellar peduncle L", "SCP"],  ## TBC
    15: ["Cerebral peduncle R", "CP"],  # TBC
    16: ["Cerebral peduncle L", "CP"],  # TBC
    17: ["Anterior limb of internal capsule R", "ALIC"],  # TBC or frontal
    18: ["Anterior limb of internal capsule L", "ALIC"],  # TBC or frontal
    19: ["Posterior limb of internal capsule R", "PLIC"], # TBC or parietal (or None, wiki says connected to motor ctx.)
    20: ["Posterior limb of internal capsule L", "PLIC"], # TBC or parietal (or None, wiki says connected to motor ctx.)
    21: ["Retrolenticular part of internal capsule R", "RLIC"], # TBC or occipital
    22: ["Retrolenticular part of internal capsule L", "RLIC"], # TBC or occipital
    23: ["Anterior corona radiata R", "ACR"],# Frontal
    24: ["Anterior corona radiata L", "ACR"],# Frontal
    25: ["Superior corona radiata R", "SCR"], # Frontal or None
    26: ["Superior corona radiata L", "SCR"],# Frontal or None
    27: ["Posterior corona radiata R", "PCR"],# Parietal
    28: ["Posterior corona radiata L", "PCR"],# Parietal
    29: ["Posterior thalamic radiation R", "PTR"], # Parietal
    30: ["Posterior thalamic radiation L", "PTR"], # Parietal
    31: ["Sagittal stratum R", "SS"], # Temporal (connects inf temporal with sup parietal, passes through temporal)
    32: ["Sagittal stratum L", "SS"],# Temporal
    33: ["External capsule R", "EC"], # TBC
    34: ["External capsule L", "EC"], # TBC
    35: ["Cingulum (cingulate gyrus) R", "CGC"], # Cingulate
    36: ["Cingulum (cingulate gyrus) L", "CGC"],# Cingulate
    37: ["Cingulum (hippocampus) R", "CGH"], # hippocampus
    38: ["Cingulum (hippocampus) L", "CGH"], # hippocampus
    39: ["Fornix (cres) / Stria terminalis R", "FX"], # hippocampus
    40: ["Fornix (cres) / Stria terminalis L", "FX"], # hippocampus
    41: ["Superior longitudinal fasciculus R", "SLF"], # occip - update: wrong, should be parietal
    42: ["Superior longitudinal fasciculus L", "SLF"], # occip - update: wrong, should be parietal
    43: ["Superior fronto-occipital fasciculus R", "SFO"], # TBC or occip/frontal
    44: ["Superior fronto-occipital fasciculus L", "SFO"], # TBC or occip/frontal
    45: ["Uncinate fasciculus R", "UNC"], # TBC or temporal/frontal
    46: ["Uncinate fasciculus L", "UNC"], # TBC or temporal/frontal
    47: ["Tapetum R", "TP"],# TBC or temporal (these are fibers from corpus callosum that go to temporal)
    48: ["Tapetum L", "TP"]}# TBC or temporal

  dtiBiomkStructTemplate_updated = {
          'CST':'TBC', # 'Frontal',
          'ACR':'Frontal',
          'SCR': 'TBC', #'Frontal',
          'TP': 'Temporal', # 'TBC',
          'PCR':'TBC', #Parietal',
          'PTR': 'Parietal',
          'SS': 'TBC', #'Temporal',
          'UNC':'TBC',
          'SLF':'TBC', #'Occipital',
          'SFO':'TBC',
          'CGC':'Cingulate',
          'GCC': 'Frontal', #'TBC',
          'BCC': 'TBC', #'Cingulate',
          'SCC': 'Occipital', # 'Cingulate',
          'CGH':'Hippocampus',
          'FX':'Hippocampus',
          'ALIC':'TBC',
          'PLIC':'TBC',
          'RLIC':'TBC',
          'ICP':'TBC',
          'SCP':'TBC',
          'CP':'TBC',
          'PCT':'TBC',
          'EC':'TBC',
          'ML':'TBC',
          'n':'NA'
  }

  # dtiBiomkStructTemplate = {
  #   'Frontal' : ['CST', 'ACR', 'SCR'],
  #   'Parietal' : ['PCR', 'PTR'],
  #   'Temporal' : ['SS'],
  #   'Occipital' : ['SLF'], # not only occipital, but also frontal & temporal
  #   'Cingulate' : ['CGC', 'GCC', 'BCC', 'SCC'],
  #   'Hippocampus': ['CGH', 'FX']
  # }

  ##########################

  # remove subj ID 1719, fa values too low due to presence of artifact.
  dtiSS = dtiSS[~np.in1d(dtiSS.Scan1Study, [1719, 1496, 1760])]
  dtiSS.reset_index(drop=True, inplace=True)

  print('-------------------------\n\n')
  dtiSS.replace({'Diagnosis': {'AD (PCA)':4, 'Control':5}}, inplace=True)
  print(dtiSS['Diagnosis'])
  # print(ads)
  dtiSS_grouped = dtiSS.groupby(['region', 'metric'])

  diagNrs = [4,5]
  diagLabels = ['PCA', 'Control']
  diagColors = ['r', 'g']
  nrWMregions = 48
  nrSubj = dtiSS_grouped.get_group((0,'fa')).shape[0]
  sortedIdMS = np.zeros((nrWMregions,int(nrSubj/2)))

  for r in range(nrWMregions):
    for measure in ['fa', 'md', 'ad']:
      # print('r = %s' % mappingIDtoRegion[r], 'meas=', measure)
      # print(dtiSS_grouped.get_group((r,measure))[['mean', 'Diagnosis']])
      colLabel = mappingIDtoRegion[r][0].replace("/", "")
      outFile = 'resfiles/tad-drc/validDf_%s_%d%s.png' % (measure, r, colLabel)
      currGroup = dtiSS_grouped.get_group((r,measure)).reset_index()
      measureCol = currGroup['mean']
      diagCol = currGroup['Diagnosis']
      print('colLabel', colLabel)

      # visDfCol(measureCol, colLabel, diagCol, diagNrs, diagLabels, diagColors, outFile)

      ctlIndx = np.where(diagCol == diagNrs[1])[0]
      measCtl = measureCol[diagCol == diagNrs[1]].as_matrix().reshape(-1)
      minInd = np.argmin(measCtl)
      indxSorted = np.argsort(measCtl)
      # print(measCtl)
      # print('indxSorted', indxSorted)
      # print('ctlIndx[indxSorted]', ctlIndx[indxSorted])
      sortedIdMS[r,:] = currGroup.loc[ctlIndx[indxSorted],'Scan1Study'].as_matrix()
      # print('indxSorted', currGroup.loc[ctlIndx[indxSorted],'Scan1Study'])
      # print('minInd', currGroup.loc[ctlIndx[minInd],:])
      print('sortedIdMS',colLabel, sortedIdMS[r,:])

      # import pdb
      # pdb.set_trace()

  # print('sortedIdMS', sortedIdMS)
  # remove subj ID 1719, fa values too low due to presence of artifact.
  # dtiSS = dtiSS[dtiSS.Scan1Study != 1719]
  # dtiSS.reset_index(drop=True, inplace=True)

  # print(dtiSS.Scan1Study)
  # print(ads)
  ###########################

  dtiSS['region'] = dtiSS['region'].map(lambda x: \
       'DTI FA '+dtiBiomkStructTemplate_updated[mappingIDtoRegion[x][1]])

  # print(dtiSS)
  # print(asd)
  dtiSS_means = dtiSS.groupby(['Scan1Study','region', 'metric'])['mean']\
                  .mean().reset_index()

  print(dtiSS.groupby(['Scan1Study','region', 'metric']).mean())
  # print(adsa)


  idx = dtiSS_means.metric == 'fa'
  print('idx', idx)
  # dtiSS_means.drop(idx, inplace=True)
  dtiSS_means = dtiSS_means[idx]
  dtiSS_means.reset_index(drop=True, inplace=True)


  # print(asd)

  dtiSS_pivoted = dtiSS_means.\
          pivot(index = 'Scan1Study', columns = 'region', values = 'mean')

  unqScans_dti = np.unique(dtiSS_pivoted.index)
  unqScans_tad = np.unique(validDf.scanID)

  Scan_inter = list(set(unqScans_dti) & set(unqScans_tad))

  validDf_u = validDf.set_index('scanID')
  validDf_u.update(dtiSS_pivoted)
  validDf_u = validDf_u.reset_index()


  # dtiBiomkStructTemplate = {
  #   'Frontal' : ['CST', 'ACR', 'SCR'],
  #   'Parietal' : ['PCR', 'PTR'],
  #   'Temporal' : ['SS'],
  #   'Occipital' : ['SLF'], # not only occipital, but also frontal & temporal
  #   'Cingulate' : ['CGC', 'GCC', 'BCC', 'SCC'],
  #   'Hippocampus': ['CGH', 'FX']
  # }

  # print('validDf_u', validDf_u)
  # multiply by the number of regions we averaged in the original ADNI model. double because of L+R
  # TODO: make ADNI only take the mean, so we don't need to do this anymore.
  validDf_u['DTI FA Frontal'] *= 6
  validDf_u['DTI FA Parietal'] *= 4
  validDf_u['DTI FA Temporal'] *= 2
  validDf_u['DTI FA Occipital'] *= 2
  validDf_u['DTI FA Cingulate'] *= 8
  validDf_u['DTI FA Hippocampus'] *= 4

  print('validDf_u', validDf_u)

  # print(validDf_u)
  # print(asda)

  return validDf_u


def visValidDf(validDf, outFilePrefix):
  fig = pl.figure(5)
  # print(validDf.columns.tolist())
  # print(adsa)
  dtiCols = validDf.loc[:, 'DTI FA Cingulate' : 'DTI FA Temporal'].columns.tolist()
  dtiDf = validDf[dtiCols]

  # validDfByDiag = validDf.groupby([diag])
  diagNrs = [4,5]

  ctlVals = validDf.loc[validDf.diag == 4, dtiCols[0]].dropna()
  patVals = validDf.loc[validDf.diag == 5, dtiCols[0]].dropna()

  ridSortedCtlBS = np.zeros((len(dtiCols),ctlVals.shape[0]))
  ridSortedPatBS = np.zeros((len(dtiCols), patVals.shape[0]))

  for b in range(len(dtiCols)):
    pl.clf()
    ctlVals = validDf.loc[validDf.diag == 4, dtiCols[b]].dropna()
    patVals = validDf.loc[validDf.diag == 5, dtiCols[b]].dropna()
    pl.hist(ctlVals, color='g', label='ctl', histtype='step',fill=False)
    pl.hist(patVals, color='r', label='pca', histtype='step',fill=False)
    pl.title(dtiCols[b])
    pl.legend()
    fig.show()
    outFile = 'resfiles/tad-drc/valid_%s_%d_%s.png' % (outFilePrefix, b, dtiCols[b])
    fig.savefig(outFile)
    # print(adas)

    ctlScanID = validDf.loc[np.logical_and(validDf.diag == 4, ~np.isnan(validDf[dtiCols[b]])), 'scanID'].as_matrix().reshape(-1)
    patScanID = validDf.loc[np.logical_and(validDf.diag == 5, ~np.isnan(validDf[dtiCols[b]])), 'scanID'].as_matrix().reshape(-1)
    print('ctlScanID', ctlScanID)
    print('patScanID', patScanID)
    print('~np.isnan(validDf[dtiCols[b]])', np.sum(~np.isnan(validDf[dtiCols[b]])))
    print('validDf.diag == 4 ', np.sum(validDf.diag == 4) )
    # print(ads)

    idxSortCtl = np.argsort(ctlVals)
    idxSortPat = np.argsort(patVals)
    ridSortedCtlBS[b,:] = ctlScanID[idxSortCtl]
    ridSortedPatBS[b,:] = patScanID[idxSortPat]

  print('ridSortedCtlBS', ridSortedCtlBS)
  print('ridSortedPatBS', ridSortedPatBS)
  # print(ads)

def visDfCol(dfCol, colLabel, diagCol, diagNrs, diagLabels, diagColors, outFile):
  fig = pl.figure(1)
  nrDiags = len(diagNrs)
  pl.clf()
  for d in range(nrDiags):
    # print(dfCol)
    # print(diagCol)
    # print(diagNrs[d])
    # print(diagCol == diagNrs[d])
    # print(dfCol.loc[diagCol == diagNrs[d]])
    # print(dfCol[diagCol == diagNrs[d]])
    pl.hist(dfCol.loc[diagCol == diagNrs[d]].dropna(), color=diagColors[d], label=diagLabels[d], histtype='step',fill=False)

  pl.legend()
  pl.title(colLabel)
  fig.show()
  outFile = outFile.replace(" ", "_")
  fig.savefig(outFile)
  # print(ads)



def validateDRCBiomk(dpmObj, params):
  # first predict subject DTI measures

  diag = params['diag']
  disNr = 1 # predict for DRC subjects
  indxSubjToKeep = dpmObj.getIndxSubjToKeep(disNr)

  import DPMModelGeneric
  Xfilt, Yfilt = DPMModelGeneric.DPMModelGeneric.filterXYsubjInd(params['X'], params['Y'], indxSubjToKeep)

  diagSubjCurrDis = diag[indxSubjToKeep]

  #### construct sub-shifts for each biomarker
  XshiftedDisModelBS, ysPredBS, xsOrigPred1S = dpmObj.getDataDisOverBiomk(disNr)

  for b2 in range(dpmObj.nrBiomk):
    assert len(params['X'][b2]) == len(params['Y'][b2])
    assert len(XshiftedDisModelBS[b2]) == len(Yfilt[b2])

  # now get the validation set. This is already only for the DRC subjects
  Xvalid = params['Xvalid']
  Yvalid = params['Yvalid']
  RIDvalid = params['RIDvalid']
  diagValid = params['diagValid']


  labels = params['labels']
  print('labels', labels)
  nonMriBiomksList = [i for i in range(len(labels)) if labels[i].startswith('DTI')]
  mriBiomksList = [i for i in range(len(labels)) if labels[i].startswith('Volume')]

  assert len(ysPredBS) == len(Yvalid)

  nrDtiCols = len(nonMriBiomksList)
  mse = [0 for b in nonMriBiomksList]

  # subjects who have DTI validation
  subjWithValidIndx = np.where([ys.shape[0] > 0 for ys in Yvalid[nonMriBiomksList[0]]])[0]
  nrSubjWithValid = subjWithValidIndx.shape[0]
  XvalidFilt, YvalidFilt = DPMModelGeneric.DPMModelGeneric.filterXYsubjInd(Xvalid, Yvalid, subjWithValidIndx)
  diagValidFilt = diagValid[subjWithValidIndx]
  RIDvalidFilt = RIDvalid[subjWithValidIndx]
  ridCurrDis = params['RID'][indxSubjToKeep]

  XvalidShifFilt = [[[] for s in range(nrSubjWithValid)] for b in range(dpmObj.nrBiomk)]

  ###### construct the shifts of the subjects in validation set #############
  for b2 in range(nrDtiCols):
    mseList = []
    for s in range(RIDvalidFilt.shape[0]):
      # for each validation subject
      idxCurrDis = np.where(RIDvalidFilt[s] == ridCurrDis)[0][0]
      xsOrigFromModel = xsOrigPred1S[idxCurrDis]

      assert np.where(xsOrigFromModel == XvalidFilt[nonMriBiomksList[b2]][s][0])[0].shape[0] == 1
      idxXsWithValid = np.where(xsOrigFromModel == XvalidFilt[nonMriBiomksList[b2]][s][0])[0][0]
      ysPredCurrSubj = ysPredBS[nonMriBiomksList[b2]][idxCurrDis][idxXsWithValid]

      assert YvalidFilt[nonMriBiomksList[b2]][s].shape[0] > 0

      mseList += [(ysPredCurrSubj - YvalidFilt[nonMriBiomksList[b2]][s][0]) ** 2]

      # also compose the shifted Xs for the validation subjects
      xsShiftedFromModel = XshiftedDisModelBS[0][idxCurrDis]
      XvalidShifFilt[nonMriBiomksList[b2]][s] = np.array([xsShiftedFromModel[idxXsWithValid]])

      assert XvalidShifFilt[nonMriBiomksList[b2]][s].shape[0] == YvalidFilt[nonMriBiomksList[b2]][s].shape[0]


    mse[b2] = np.mean(mseList)


  # part 2. plot the inferred dynamics for DRC data:
  # every biomarker against original DPS
  # also plot extra validation data on top
  xsTrajX = dpmObj.getXsMinMaxRange(disNr)
  predTrajXB = dpmObj.predictBiomkSubjGivenXs(xsTrajX, disNr)
  trajSamplesBXS = dpmObj.sampleBiomkTrajGivenXs(xsTrajX, disNr, nrSamples=100)


  ### build a simpler linear predictor from MR to DTI for every ROI independently.
  # Train it on ADNI MR+DTI data and use it to predict DRC-DTI from DRC-MR.

  dataDfAll = params['dataDfAll']
  colsList = dataDfAll.columns.tolist()
  mriBiomksColsInd = [i for i in range(len(colsList)) if colsList[i].startswith('Volume')]
  nonMriBiomksColsInd = [i for i in range(len(colsList)) if colsList[i].startswith('DTI')]
  nrMriBiomks = len(mriBiomksColsInd)

  dataAllTrainDf = dataDfAll.loc[dataDfAll.dataset == 1,:]

  dataAllTrain = dataAllTrainDf.values
  # print('dataAllTrain', dataAllTrain)

  nrNonMriBiomk = len(mriBiomksColsInd)

  YvalidDktNonMri = [0 for f in range(nrNonMriBiomk)]
  YvalidLinModelNonMri = [0 for f in range(nrNonMriBiomk)]
  YvalidSplineModelNonMri = [0 for f in range(nrNonMriBiomk)]
  YvalidMultivarModelNonMri = [0 for f in range(nrNonMriBiomk)]

  mseDpm = np.zeros(nrNonMriBiomk)
  mseLin = np.zeros(nrNonMriBiomk)
  mseSpline = np.zeros(nrNonMriBiomk)
  mseMultivar = np.zeros(nrNonMriBiomk)

  squaredErrorsDpm = [[] for f in range(nrNonMriBiomk)]
  squaredErrorsLin = [[] for f in range(nrNonMriBiomk)]
  squaredErrorsSpline = [[] for f in range(nrNonMriBiomk)]
  squaredErrorsMultivar = [[] for f in range(nrNonMriBiomk)]

  # select just the DTI biomarkers
  nonMriColsArrayIndx = np.array(nonMriBiomksList)
  mriColsArrayIndx = np.array(mriBiomksList)
  predTrajNonMriXB = predTrajXB[:,nonMriColsArrayIndx]
  predTrajMriXB = predTrajXB[:, mriColsArrayIndx]
  trajSamplesNonMriBXS = trajSamplesBXS[nonMriColsArrayIndx,:,:]
  XvalidShifNonMriFilt = [XvalidShifFilt[b] for b in nonMriBiomksList]
  YvalidFiltNonMri = [YvalidFilt[b] for b in nonMriBiomksList]
  YvalidFiltMriClosestToNonMri = [[[] for s in range(nrSubjWithValid) ] for b in nonMriBiomksColsInd] # only the MRI where Non-MRI exists

  nonMriValValidAll = [[] for b in range(nrNonMriBiomk)]
  nonMriPredValidDktAll = [[] for b in range(nrNonMriBiomk)]
  nonMriPredValidLinAll = [[] for b in range(nrNonMriBiomk)]
  nonMriPredValidSplineAll = [[] for b in range(nrNonMriBiomk)]
  nonMriPredValidMultivarAll = [[] for b in range(nrNonMriBiomk)]

  corrDpm = np.zeros(nrNonMriBiomk)
  pValDpm = np.zeros(nrNonMriBiomk)
  corrLin = np.zeros(nrNonMriBiomk)
  pValLin = np.zeros(nrNonMriBiomk)
  corrSpline = np.zeros(nrNonMriBiomk)
  pValSpline = np.zeros(nrNonMriBiomk)
  corrMultivar = np.zeros(nrNonMriBiomk)
  pValMultivar = np.zeros(nrNonMriBiomk)

  for b in range(nrNonMriBiomk):

    mriDataCurrCol = dataAllTrain[:, mriBiomksColsInd[b]]
    nonMriDataCurrCol = dataAllTrain[:, nonMriBiomksColsInd[b]]

    nnMask = ~np.isnan(mriDataCurrCol) & ~np.isnan(nonMriDataCurrCol)
    linModel = sklearn.linear_model.LinearRegression(fit_intercept=True)

    linModel.fit(mriDataCurrCol[nnMask].reshape(-1,1),
      nonMriDataCurrCol[nnMask].reshape(-1,1))

    from scipy.interpolate import Rbf, UnivariateSpline
    sortedXInd = np.argsort(mriDataCurrCol[nnMask])
    splineModel = UnivariateSpline(mriDataCurrCol[nnMask][sortedXInd],
                                               nonMriDataCurrCol[nnMask][sortedXInd], k=3)

    # Now do multivariate prediction. Based on all MRI values at each lobe, predict one DTI biomk.
    nnMaskMulti = (np.sum(np.isnan(dataAllTrain[:, mriBiomksColsInd]),axis=1) == 0) & ~np.isnan(nonMriDataCurrCol)
    # print('print(np.sum(nnMaskMulti))', np.sum(nnMaskMulti))
    assert np.isfinite(dataAllTrain[nnMaskMulti, :][:, mriBiomksColsInd]).all()
    assert np.isfinite(nonMriDataCurrCol[nnMaskMulti]).all()
    nnDataMri = dataAllTrain[nnMaskMulti, :][:, mriBiomksColsInd]
    # print(nnDataMri[:,0].shape[0])
    # print(nonMriDataCurrCol[nnMaskMulti].shape[0])
    assert nnDataMri[:,0].shape[0] == nnDataMri[:, 1].shape[0]
    assert nnDataMri[:,0].shape[0] == nonMriDataCurrCol[nnMaskMulti].shape[0]
    multivarModel = Rbf(nnDataMri[:,0], nnDataMri[:,1], nnDataMri[:,2], nnDataMri[:,3], nnDataMri[:,4],
               nnDataMri[:,5], nonMriDataCurrCol[nnMaskMulti], function='cubic')
    # print(ads)


    YvalidDktNonMri[b] = [[] for s in range(nrSubjWithValid)]  # Non-MRI predictions of DKT model for subj in validation set
    YvalidLinModelNonMri[b] = [[] for s in range(nrSubjWithValid) ] # Non-MRI predictions of linear model for subj in validation set
    YvalidSplineModelNonMri[b] = [[] for s in range(nrSubjWithValid)]  # Non-MRI predictions of linear model for subj in validation set
    YvalidMultivarModelNonMri[b] = [[] for s in range(nrSubjWithValid)]  # Non-MRI predictions of linear model for subj in validation set


    for s in range(nrSubjWithValid):
      mrValsValidCurrSubj = np.array(YvalidFilt[mriBiomksList[b]][s]).reshape(-1,1)
      nonMriValValidCurrSubj = YvalidFilt[nonMriBiomksList[b]][s][0]

      xMriCurr = np.array(XvalidFilt[mriBiomksList[b]][s])
      xDTICurr = XvalidFilt[nonMriBiomksList[b]][s][0]

      closestMriIdx = np.argmin(np.abs(xMriCurr - xDTICurr))

      YvalidFiltMriClosestToNonMri[b][s] = mrValsValidCurrSubj[closestMriIdx]
      nonMriPredValidLin = linModel.predict(mrValsValidCurrSubj[closestMriIdx].reshape(-1,1))
      nonMriPredValidLin = nonMriPredValidLin[0][0]
      nonMriPredValidSpline = splineModel(mrValsValidCurrSubj[closestMriIdx].astype(float))
      mrValsToPredictCurrSubj = [YvalidFilt[mriBiomksList[u]][s][closestMriIdx]
                                 for u in range(nrMriBiomks)]
      # print('mrValsToPredictCurrSubj', mrValsToPredictCurrSubj)
      assert len(mrValsToPredictCurrSubj) == nrMriBiomks
      nonMriPredValidMultivar = multivarModel(mrValsToPredictCurrSubj[0], mrValsToPredictCurrSubj[1],
        mrValsToPredictCurrSubj[2], mrValsToPredictCurrSubj[3], mrValsToPredictCurrSubj[4],
        mrValsToPredictCurrSubj[5])

      # print(np.array([nonMriPredValidLin]))
      # print(mrValsValidCurrSubj[closestMriIdx])
      # print(dasdsa)
      assert nonMriPredValidLin != []
      YvalidLinModelNonMri[b][s] = [nonMriPredValidLin]
      YvalidSplineModelNonMri[b][s] = [nonMriPredValidSpline]
      YvalidMultivarModelNonMri[b][s] = [nonMriPredValidMultivar]

      indOfXTrajClosestToCurrSubj = np.argmin(np.abs(XvalidShifNonMriFilt[b][s][0] - xsTrajX))
      nonMriPredValidDkt = predTrajNonMriXB[indOfXTrajClosestToCurrSubj, b]
      YvalidDktNonMri[b][s] = [np.array(nonMriPredValidDkt)]

      # if diagValidFilt[s] == PCA:
      squaredErrorsDpm[b] += [(nonMriValValidCurrSubj - nonMriPredValidDkt) ** 2]
      squaredErrorsLin[b] += [(nonMriValValidCurrSubj - nonMriPredValidLin) ** 2]
      squaredErrorsSpline[b] += [(nonMriValValidCurrSubj - nonMriPredValidSpline) ** 2]
      squaredErrorsMultivar[b] += [(nonMriValValidCurrSubj - nonMriPredValidMultivar) ** 2]

      nonMriValValidAll[b] += [nonMriValValidCurrSubj]
      nonMriPredValidDktAll[b] += [nonMriPredValidDkt]
      nonMriPredValidLinAll[b] += [nonMriPredValidLin]
      nonMriPredValidSplineAll[b] += [nonMriPredValidSpline]
      nonMriPredValidMultivarAll[b] += [nonMriPredValidMultivar]

      YvalidFiltMriClosestToNonMri[b][s] = np.array(YvalidFiltMriClosestToNonMri[b][s])
      YvalidDktNonMri[b][s] = np.array(YvalidDktNonMri[b][s])
      YvalidLinModelNonMri[b][s] = np.array(YvalidLinModelNonMri[b][s])
      YvalidSplineModelNonMri[b][s] = np.array(YvalidSplineModelNonMri[b][s])
      YvalidMultivarModelNonMri[b][s] = np.array(YvalidMultivarModelNonMri[b][s])

      print(nonMriPredValidLin)
      print(YvalidLinModelNonMri[b][s])
      assert YvalidLinModelNonMri[b][s].shape[0] > 0


    nonMriValValidAll[b] = np.array(nonMriValValidAll[b]).reshape(-1, 1).astype(float)
    nonMriPredValidDktAll[b] = np.array(nonMriPredValidDktAll[b]).reshape(-1, 1).astype(float)
    nonMriPredValidLinAll[b] = np.array(nonMriPredValidLinAll[b]).reshape(-1, 1).astype(float)
    nonMriPredValidSplineAll[b] = np.array(nonMriPredValidSplineAll[b]).reshape(-1, 1).astype(float)
    nonMriPredValidMultivarAll[b] = np.array(nonMriPredValidMultivarAll[b]).reshape(-1, 1).astype(float)


    # print('nonMriValValidAll', nonMriValValidAll[b].shape, nonMriValValidAll[b])
    # print('nonMriPredValidLinAll', nonMriPredValidLinAll[b].shape, nonMriPredValidLinAll[b])
    # print('nonMriPredValidDktAll', nonMriPredValidDktAll[b].shape, nonMriPredValidDktAll[b])

    corrDpm[b], pValDpm[b] = scipy.stats.spearmanr(nonMriValValidAll[b],
      nonMriPredValidDktAll[b])
    corrLin[b], pValLin[b] = scipy.stats.spearmanr(nonMriValValidAll[b],
      nonMriPredValidLinAll[b])
    corrSpline[b], pValSpline[b] = scipy.stats.spearmanr(nonMriValValidAll[b],
      nonMriPredValidSplineAll[b])
    corrMultivar[b], pValMultivar[b] = scipy.stats.spearmanr(nonMriValValidAll[b],
      nonMriPredValidMultivarAll[b])


  for b in range(nrNonMriBiomk):
    for s in range(nrSubjWithValid):
      print(YvalidFiltMriClosestToNonMri[b][s].shape)
      print(YvalidLinModelNonMri[b][s].shape)
      if YvalidFiltMriClosestToNonMri[b][s].shape[0] != YvalidLinModelNonMri[b][s].shape[0]:
        print('b s', b, s)
        print(YvalidFiltMriClosestToNonMri[b][s])
        print(YvalidLinModelNonMri[b][s])

        raise ValueError('array shapes do not match')

      if YvalidFiltMriClosestToNonMri[b][s].shape[0] != YvalidDktNonMri[b][s].shape[0]:
        print('b s', b, s)
        print(YvalidFiltMriClosestToNonMri[b][s])
        print(YvalidDktNonMri[b][s])
        raise ValueError('array shapes do not match')

      if YvalidFiltMriClosestToNonMri[b][s].shape[0] != YvalidFiltNonMri[b][s].shape[0]:
        print('b s', b, s)
        print(YvalidFiltMriClosestToNonMri[b][s])
        print(YvalidFiltNonMri[b][s])
        raise ValueError('array shapes do not match')

  for b in range(nrNonMriBiomk):
    squaredErrorsDpm[b] = np.array(squaredErrorsDpm[b])
    squaredErrorsLin[b] = np.array(squaredErrorsLin[b])
    squaredErrorsSpline[b] = np.array(squaredErrorsSpline[b])
    squaredErrorsMultivar[b] = np.array(squaredErrorsMultivar[b])

    # nonMriValValidAll[b] = nonMriValValidAll[b]
    # nonMriPredValidDktAll[b] = nonMriPredValidDktAll[b]
    # nonMriPredValidLinAll[b] = nonMriPredValidLinAll[b]

  nrBootStraps = 50
  mseDpmUB = np.zeros((nrNonMriBiomk, nrBootStraps), float)
  mseLinUB = np.zeros((nrNonMriBiomk, nrBootStraps), float)
  mseSplineUB = np.zeros((nrNonMriBiomk, nrBootStraps), float)
  mseMultivarUB = np.zeros((nrNonMriBiomk, nrBootStraps), float)
  corrDpmUB = np.zeros((nrNonMriBiomk, nrBootStraps), float)
  corrLinUB = np.zeros((nrNonMriBiomk, nrBootStraps), float)
  corrSplineUB = np.zeros((nrNonMriBiomk, nrBootStraps), float)
  corrMultivarUB = np.zeros((nrNonMriBiomk, nrBootStraps), float)

  for b in range(nrNonMriBiomk):
    nrSubjWithValidAndChosen = len(squaredErrorsLin[b])
    assert nonMriValValidAll[b].shape[0] == len(squaredErrorsLin[b])
    for b2 in range(nrBootStraps):
      idxBootCurr = np.array(np.random.choice(nrSubjWithValidAndChosen,
        nrSubjWithValidAndChosen), int)
      # print(len(squaredErrorsLin[b]))
      # print(idxBootCurr)
      mseDpmUB[b, b2] = np.mean(squaredErrorsLin[b][idxBootCurr])
      mseLinUB[b, b2] = np.mean(squaredErrorsDpm[b][idxBootCurr])
      mseSplineUB[b, b2] = np.mean(squaredErrorsSpline[b][idxBootCurr])
      mseMultivarUB[b, b2] = np.mean(squaredErrorsMultivar[b][idxBootCurr])

      idxBootCorrCurr = np.array(np.random.choice(nrSubjWithValidAndChosen,
        nrSubjWithValidAndChosen), int)


      print(idxBootCorrCurr)
      print(nonMriValValidAll[b])
      print(nonMriValValidAll[b][idxBootCorrCurr])
      print(nonMriPredValidDktAll[b][idxBootCorrCurr])
      print(corrDpmUB[b, b2])
      corrDpmUB[b, b2], _ = scipy.stats.spearmanr(nonMriValValidAll[b][idxBootCorrCurr],
        nonMriPredValidDktAll[b][idxBootCorrCurr])
      corrLinUB[b, b2], _ = scipy.stats.spearmanr(nonMriValValidAll[b][idxBootCorrCurr],
        nonMriPredValidLinAll[b][idxBootCorrCurr])
      corrSplineUB[b, b2], _ = scipy.stats.spearmanr(nonMriValValidAll[b][idxBootCorrCurr],
        nonMriPredValidSplineAll[b][idxBootCorrCurr])
      corrMultivarUB[b, b2], _ = scipy.stats.spearmanr(nonMriValValidAll[b][idxBootCorrCurr],
        nonMriPredValidMultivarAll[b][idxBootCorrCurr])



  metrics = {}
  metrics['dpm'] = {}
  metrics['dpm']['corrUB'] = corrDpmUB
  # metrics['dpm']['pValsU'] = pValDpm
  metrics['dpm']['mseUB'] = mseDpmUB
  metrics['lin'] = {}
  metrics['lin']['corrUB'] = corrLinUB
  # metrics['lin']['pValsU'] = pValLin
  metrics['lin']['mseUB'] = mseLinUB
  metrics['spline'] = {}
  metrics['spline']['corrUB'] = corrSplineUB
  metrics['spline']['mseUB'] = mseSplineUB
  metrics['multivar'] = {}
  metrics['multivar']['corrUB'] = corrMultivarUB
  metrics['multivar']['mseUB'] = mseMultivarUB
  labelsNonMri = [params['labels'][b] for b in nonMriBiomksList]
  metrics['labelsNonMri'] = labelsNonMri

  # plot against MRI vals instead of DPS time-shifts

  # also plot training data DTI[b] in MRI[b] space
  YDti = [params['Y'][b] for b in nonMriBiomksList]
  YMriClosestToDti = [[0 for s in range(len(YDti[b]))] for b in mriBiomksList]  # only the MRI where DTI exists
  # idxWithDti = [s for s in range(len(YDti)) ]
  # print('YDti', YDti)
  # print(adsa)

  for b in range(nrNonMriBiomk):

    for s in range(len(YDti[b])):

      YMriClosestToDti[b][s] = np.array([])

      if YDti[b][s].shape[0] > 0:

        xsMriCurrSubj = params['X'][mriBiomksList[b]][s]
        xsDtiCurrSubj = params['X'][nonMriBiomksList[b]][s]

        mriValsCorrespToDtiCurrSubj = []
        for t in range(xsDtiCurrSubj.shape[0]):
          mriIndClosestToCurrDtiScan = np.argmin(np.abs(xsDtiCurrSubj[t] - xsMriCurrSubj))

          mriValsCorrespToDtiCurrSubj += [params['Y'][mriBiomksList[b]][s][mriIndClosestToCurrDtiScan]]

        YMriClosestToDti[b][s] = np.array(mriValsCorrespToDtiCurrSubj)


      # print(YMriClosestToDti[b][s].shape[0])
      # print(YDti[b][s].shape[0])
      assert YMriClosestToDti[b][s].shape[0] == YDti[b][s].shape[0]

  # print(ads)

  labelsDti = [params['labels'][b] for b in nonMriBiomksList]
  metrics['labelsDti'] = labelsDti
  # change diagnosis numbers to get different plotting behaviour (specific labels, colors and markers)
  diagValidFiltLinModel = copy.deepcopy(diagValidFilt)
  diagValidFiltLinModel[diagValidFiltLinModel == CTL2] = CTL_OTHER_MODEL
  diagValidFiltLinModel[diagValidFiltLinModel == PCA] = PCA_OTHER_MODEL

  diagValidFiltDktModel = copy.deepcopy(diagValidFilt)
  diagValidFiltDktModel[diagValidFiltDktModel == CTL2] = CTL_DKT
  diagValidFiltDktModel[diagValidFiltDktModel == PCA] = PCA_DKT

  # plot just the trajectories by modality groups
  # for d in range(dpmObj.nrDis):
  #   fig = dpmObj.plotter.plotTrajInDisSpaceOverlap(dpmObj, d, params, replaceFig=True)
  #   fig.savefig('%s/trajDisSpaceOverlap_%s_%s.png' % (params['outFolder'],
  #     params['disLabels'][d], params['expName']))

  plotFigs = True
  if plotFigs:

    # for u in range(dpmObj.nrFuncUnits):
    #   trajStructUnitModel = dpmObj.unitModels[u].plotter.getTrajStructWithTrueParams(dpmObj.unitModels[u])
    #   fig = dpmObj.unitModels[u].plotter.plotTraj(dpmObj.unitModels[u], trajStructUnitModel,
    #     legendExtraPlot=True, rowsAuto=True)
    #   fig.savefig('%s/unit%d_allTraj.png' % (params['outFolder'], u))


    # for d in range(dpmObj.nrDis):
    #   # yNormMode = dpmObj.params['plotTrajParams']['yNormMode']
    #   yNormMode = 'unscaled'
    #   trajStructDisModel = dpmObj.disModels[d].plotter.getTrajStructWithTrueParams(dpmObj.disModels[d], yNormMode)
    #   fig = dpmObj.disModels[d].plotter.plotAllTrajZeroOne(dpmObj.disModels[d], trajStructDisModel)
    #   fig.savefig('%s/dis%d_%s_allTrajZeroOne.png' % (params['outFolder'], d, dpmObj.params['disLabels'][d]))


    # plot DTI over MRI space: traj, validation data, predictions of linear model, training data.
    fig = dpmObj.plotter.plotTrajInBiomkSpace(dpmObj=dpmObj,
      xsTrajXB=predTrajMriXB, predTrajXB=predTrajNonMriXB, trajSamplesBXS=trajSamplesNonMriBXS,
      XsubjData1BSX=YvalidFiltMriClosestToNonMri, YsubjData1BSX=YvalidFiltNonMri, diagData1S=diagValidFilt,
      XsubjData2BSX=YvalidFiltMriClosestToNonMri, YsubjData2BSX=YvalidLinModelNonMri, diagData2S=diagValidFiltLinModel,
      XsubjData3BSX=YMriClosestToDti, YsubjData3BSX=YDti, diagData3S=params['diag'],
      labels=labelsDti,
      ssdDKT=mseDpm, ssdNoDKT=mseLin, replaceFig=True)
    fig.savefig('%s/validTrajDtiOverMriPCA.png' % params['outFolder'])

    # plot DTI over MRI space: DKT predictions, predictions of linear model, validation data.
    fig = dpmObj.plotter.plotTrajInBiomkSpace(dpmObj=dpmObj,
      xsTrajXB=None, predTrajXB=None, trajSamplesBXS=None,
      XsubjData1BSX=YvalidFiltMriClosestToNonMri, YsubjData1BSX=YvalidFiltNonMri, diagData1S=diagValidFilt,
      XsubjData2BSX=YvalidFiltMriClosestToNonMri, YsubjData2BSX=YvalidLinModelNonMri, diagData2S=diagValidFiltLinModel,
      XsubjData3BSX=YvalidFiltMriClosestToNonMri, YsubjData3BSX=YvalidDktNonMri, diagData3S=diagValidFiltDktModel,
      labels=labelsDti,
      ssdDKT=None, ssdNoDKT=None, replaceFig=True)
    fig.savefig('%s/validPredDtiOverMriPCA.png' % params['outFolder'])

    # fig = dpmObj.plotterObj.plotTrajInDisSpace(xsTrajX, predTrajDtiXB, trajSamplesDtiBXS,
    #   XvalidShifDtiFilt, YvalidFiltDti, diagValidFilt,
    #   XvalidShifDtiFilt, YvalidLinModelNonMri, diagValidFiltLinModel,
    #   XsubjData3BSX=None, YsubjData3BSX=None, diagData3S=None,
    #   labelsDti, mseDpm, mseLin,
    #   replaceFig=False)
    # fig.savefig('%s/validDtiPCA.png' % params['outFolder'])

  return metrics


