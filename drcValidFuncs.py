from env import *
import pandas as pd
import numpy as np
from matplotlib import pyplot as pl

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
  fig = pl.figure(1)
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
  indxSubjToKeep = np.where(dpmObj.indxSubjForEachDisD[disNr])[0]

  nrBiomk = len(params['X'])
  print('nrBiomk', nrBiomk)
  Xfilt = [[] for b in range(nrBiomk)]
  Yfilt = [[] for b in range(nrBiomk)]
  for b in range(nrBiomk):
    Xfilt[b] = [params['X'][b][i] for i in indxSubjToKeep]
    Yfilt[b] = [params['Y'][b][i] for i in indxSubjToKeep]

  diagSubjCurrDis = diag[indxSubjToKeep]
  ridCurrDis = params['RID'][indxSubjToKeep]
  nrSubCurrDis = indxSubjToKeep.shape[0]

  XshiftedDisModelBS = [[] for b in range(nrBiomk)]
  ysPredBS = [[] for b in range(nrBiomk)]
  XshiftedDisModelUS, XdisModelUS, YdisModelUS = dpmObj.disModels[disNr].getData()
  xsOrigPred1S = XdisModelUS[0] # all biomarkers should contain all timepoints in the disease model


  for s in range(nrSubCurrDis):
    bTmp = 0 # some biomarker, doesn't matter which one

    ysCurrSubXB = dpmObj.predictBiomkSubjGivenXs(XshiftedDisModelUS[bTmp][s], disNr)

    for b in range(nrBiomk):
      ysPredBS[b] += [ysCurrSubXB[:,b]]

      if Xfilt[b][s].shape[0] > 0:
        # fix problem when a subject has the same xs twice (bad input dataset with same visit twice)
        while np.unique(Xfilt[b][s]).shape[0] < Xfilt[b][s].shape[0]:
          for x in Xfilt[b][s]:
            if np.sum(Xfilt[b][s] == x) > 1:
              idxToRemove = np.where(Xfilt[b][s] == x)[0][0]
              Yfilt[b][s] = np.concatenate((Yfilt[b][s][:idxToRemove], Yfilt[b][s][idxToRemove+1:]))
              Xfilt[b][s] = np.concatenate((Xfilt[b][s][:idxToRemove], Xfilt[b][s][idxToRemove + 1:]))

              break

        XshiftedDisModelBS[b] += [XshiftedDisModelUS[0][s]]
      else:
        XshiftedDisModelBS[b] += [[]]


  for b in range(nrBiomk):
    assert len(params['X'][b]) == len(params['Y'][b])
    assert len(XshiftedDisModelBS[b]) == len(Yfilt[b])


  # now get the validation set. This is already only for the DRC subjects
  Xvalid = params['Xvalid']
  Yvalid = params['Yvalid']
  RIDvalid = params['RIDvalid']
  diagValid = params['diagValid']

  labels = params['labels']
  print('labels', labels)
  dtiColsIdx = [i for i in range(len(labels)) if labels[i].startswith('DTI')]

  assert len(ysPredBS) == len(Yvalid)

  nrDtiCols = len(dtiColsIdx)
  mse = [0 for b in dtiColsIdx]

  subjWithValidIndx = np.where([ys.shape[0] > 0 for ys in Yvalid[dtiColsIdx[0]]])[0]
  nrSubjWithValid = subjWithValidIndx.shape[0]
  YvalidFilt = [0 for b in range(nrBiomk)]
  XvalidFilt = [0 for b in range(nrBiomk)]
  diagValidFilt = diagValid[subjWithValidIndx]
  for b in range(nrBiomk):
    XvalidFilt[b] = [Xvalid[b][s] for s in subjWithValidIndx]
    YvalidFilt[b] = [Yvalid[b][s] for s in subjWithValidIndx]

  RIDvalidFilt = RIDvalid[subjWithValidIndx]

  XvalidShifFilt = [[[] for s in range(nrSubjWithValid)] for b in range(nrBiomk)]

  for b in range(nrDtiCols):
    mseList = []
    for s in range(RIDvalidFilt.shape[0]):
      # for each validation subject
      idxCurrDis = np.where(RIDvalidFilt[s] == ridCurrDis)[0][0]
      xsOrigFromModel = xsOrigPred1S[idxCurrDis]

      assert np.where(xsOrigFromModel == XvalidFilt[dtiColsIdx[b]][s][0])[0].shape[0] == 1
      idxXsWithValid = np.where(xsOrigFromModel == XvalidFilt[dtiColsIdx[b]][s][0])[0][0]
      ysPredCurrSubj = ysPredBS[dtiColsIdx[b]][idxCurrDis][idxXsWithValid]

      assert YvalidFilt[dtiColsIdx[b]][s].shape[0] > 0

      mseList += [(ysPredCurrSubj - YvalidFilt[dtiColsIdx[b]][s][0]) ** 2]

      # also compose the shifted Xs for the validation subjects
      xsShiftedFromModel = XshiftedDisModelBS[0][idxCurrDis]
      XvalidShifFilt[dtiColsIdx[b]][s] = np.array([xsShiftedFromModel[idxXsWithValid]])

      assert XvalidShifFilt[dtiColsIdx[b]][s].shape[0] == YvalidFilt[dtiColsIdx[b]][s].shape[0]


    mse[b] = np.mean(mseList)

  # print('mse', mse)
  # print(ads)


  # part 2. plot the inferred dynamics for DRC data:
  # every biomarker against original DPS
  # also plot extra validation data on top
  xsTrajX = dpmObj.disModels[disNr].getXsMinMaxRange()
  predTrajXB = dpmObj.predictBiomkSubjGivenXs(xsTrajX, disNr)
  trajSamplesBXS = dpmObj.sampleBiomkTrajGivenXs(xsTrajX, disNr, nrSamples=100)

  print('XshiftedDisModelBS', XshiftedDisModelBS)
  print('XvalidShifFilt', XvalidShifFilt)
  print('predTrajXB', predTrajXB[:,0])
  print('xsTrajX', xsTrajX)
  print('ysPredBS', ysPredBS)
  # print(ads)


  # fig = dpmObj.plotterObj.plotTrajInDisSpace(xsTrajX, predTrajXB, trajSamplesBXS,
  #   XshiftedDisModelBS, Yfilt, diagSubjCurrDis,
  #   XvalidShifFilt, YvalidFilt, diagValidFilt, replaceFig=True)
  # fig.savefig('%s/validPCA.png' % params['outFolder'])


  # plot just the DTI biomarkers
  dtiColsArrayIndx = np.array(dtiColsIdx)
  print('dtiColsArrayIndx', dtiColsArrayIndx)
  predTrajDtiXB = predTrajXB[:,dtiColsArrayIndx]
  trajSamplesDtiBXS = trajSamplesBXS[dtiColsArrayIndx,:,:]
  # print(len(XshiftedDisModelBS))
  # print(ads)
  XvalidShifDtiFilt = [XvalidShifFilt[b] for b in dtiColsIdx]
  YvalidFiltDti = [YvalidFilt[b] for b in dtiColsIdx]

  labelsDti = [params['labels'][b] for b in dtiColsIdx]

  fig = dpmObj.plotterObj.plotTrajInDisSpace(xsTrajX, predTrajDtiXB, trajSamplesDtiBXS,
    XvalidShifDtiFilt, YvalidFiltDti, diagValidFilt,
    None, None, None, labelsDti, replaceFig=True)
  fig.savefig('%s/validDtiPCA.png' % params['outFolder'])

  # plot just the trajectories by modality groups
  # fig = dpmObj.plotterObj.plotTrajInDisSpaceOverlap(xsTrajX, predTrajXB,
  #   trajSamplesBXS, params, replaceFig=True)
  # fig.savefig('%s/trajDisSpaceOverlap_%s_%s.png' % (params['outFolder'],
  # params['disLabels'][disNr], params['expName']))