from env import *
import pandas as pd
import numpy as np
from drcValidFuncs import *
import copy
from matplotlib import pyplot as pl
import scipy
import auxFunc
import pickle

def addBiomks(biomkStruct, sourceDf, targetDf, collapseFunc):

  biomkLabels = np.sort(list(biomkStruct.keys()))
  nrBiomk = len(biomkStruct.keys())
  for b in range(nrBiomk):
    # print(biomkLabels[b], biomkStruct[biomkLabels[b]])
    # fullBiomkNames = ['%s_%s' % (x, spreadsheetTag)
    #   for x in biomkStruct[biomkLabels[b]]]
    sys.stdout.flush()
    # print(np.where(np.in1d(sourceDf.columns.tolist(), biomkStruct[biomkLabels[b]])))
    # print(sourceDf.columns.tolist()[1200:])

    dataFrameCurrSubset = sourceDf[biomkStruct[biomkLabels[b]]]
    cols = list(dataFrameCurrSubset.columns)
    # print(list(dataFrameCurrSubset.columns))
    # print(dict(zip(cols, [x[:6] for x in cols])))
    # dataFrameCurrSubset.rename(index=str, columns=dict(zip(cols, [x[:6] for x in cols])))
    dataFrameCurrSubset.columns = [x[:6] for x in cols]
    # print('dataFrameCurrSubset', dataFrameCurrSubset.loc[:5,:])
    # print('np.sum', np.sum(dataFrameCurrSubset,axis=1))
    targetDf.loc[:, biomkLabels[b]] = pd.Series(np.sum(dataFrameCurrSubset,axis=1),
      index=targetDf.index)
    # targetDf = targetDf.assign(asd=np.sum(dataFrameCurrSubset,axis=1))
    # print(targetDf.loc[:5, biomkLabels[b]])
    # print(ads)



  return targetDf

def regressCov(data, regressorVector, diag, diagsCTL = (CTL, CTL2), printFigs=False):
  oldData = copy.deepcopy(data)
  M = np.zeros((data.shape[1], 2))
  desiredMean = np.zeros(data.shape[1])
  regressorVectorArray = np.array(regressorVector)
  for i in range(data.shape[1]):
      currCol = data.columns[i]
      notNanIndices = np.logical_not(np.isnan(regressorVectorArray))
      notNanIndices = np.logical_and(notNanIndices,
        np.logical_not(np.isnan(data.loc[:,currCol])))
      regressorVectorNN = regressorVectorArray[notNanIndices]
      diagNN = diag[notNanIndices]
      dataNNcurrCol = data.loc[notNanIndices,currCol]
      indicesCtl = np.in1d(diagNN, diagsCTL)
      regressorCTL = regressorVectorNN[indicesCtl]

      # Solve the GLM: Y = [X 1] * M
      X = np.concatenate((regressorCTL.reshape(-1,1),
      np.ones((regressorCTL.shape[0],1))),axis=1)
      XXX = np.dot(np.linalg.pinv(np.dot(X.T,X)), X.T)

      M[i,:] = np.dot(XXX, dataNNcurrCol.loc[indicesCtl]) # params of linear fit
      assert(~any(np.isnan(M[i,:])));
      Xfull = np.concatenate((regressorVectorArray.reshape(-1,1),
      np.ones((regressorVector.shape[0],1))),axis=1)

      Yhat = np.dot(Xfull, M[i,:]) # estimated Ys
      desiredMean[i] = np.nanmean(dataNNcurrCol.loc[indicesCtl])
      data.loc[:, currCol] = data.loc[:,currCol] - (Yhat - desiredMean[i])

      if printFigs:
        h = pl.figure(1, figsize=(15,10))
        pl.scatter(regressorVector, oldData.loc[:,currCol], c='r',label='before', s=5)
        pl.scatter(regressorVector, data[currCol], c='b',label='after', s=5)
        pl.plot(regressorVectorNN, Yhat[notNanIndices], c='r')
        correctedPred = np.nanmean(dataNNcurrCol.loc[indicesCtl]) * \
                        np.ones(dataNNcurrCol.loc[indicesCtl].shape[0])
        pl.plot(regressorVectorNN[indicesCtl],correctedPred  , c='b')
        pl.title('%s' % data.columns[i])
        pl.legend()

        pl.show()

  params = dict(M=M, labels=data.columns, desiredMean=desiredMean)

  return data, params

def applyRegFromParams(data, regressorVector, diag, params, diagsCTL = (CTL, CTL2),
  printFigs=False, otherDataToPlot=None, otherRegVector=None):

  oldData = copy.deepcopy(data)
  M = params['M']
  desiredMean = params['desiredMean']
  diagNrs = [[CTL, CTL2], [PCA]]
  diagLabels = ['CTL', 'PCA']
  diagCols = ['g', 'r']

  regressorVectorArray = np.array(regressorVector)

  assert regressorVectorArray.shape[0] == diag.shape[0]
  for i in range(data.shape[1]):
      currCol = data.columns[i]

      assert(~any(np.isnan(M[i,:])));

      Xfull = np.concatenate((regressorVectorArray.reshape(-1,1),
      np.ones((regressorVectorArray.shape[0],1))),axis=1)

      Yhat = np.dot(Xfull, M[i,:]) # estimated Ys
      data.loc[:, currCol] = data.loc[:,currCol] - (Yhat - desiredMean[i])

      if printFigs and (i in range(16,20)):
        h = pl.figure(1, figsize=(15,10))
        pl.clf()
        for d in range(len(diagNrs)):
          currDiagInd = np.in1d(diag, diagNrs[d])
          print(regressorVectorArray[currDiagInd])
          print(oldData.loc[currDiagInd,currCol])
          pl.scatter(regressorVectorArray[currDiagInd], oldData.loc[currDiagInd,currCol],
            c=diagCols[d],marker='.',label='%s before' % diagLabels[d], s=12)
          pl.scatter(regressorVectorArray[currDiagInd], data.loc[currDiagInd, currCol],
            c=diagCols[d],marker='x',label='%s after' % diagLabels[d], s=12)

          # pl.scatter(regressorVector, oldData.loc[:, currCol], c='r', m='.', label='before', s=5)
          # pl.scatter(regressorVector, data[currCol], c='b', m='.', label='after', s=5)

        pl.plot(regressorVectorArray, Yhat, c='k', label='line')
        if otherDataToPlot is not None:
          pl.scatter(otherRegVector, otherDataToPlot.loc[:, currCol], c = 'k', marker = 'o',
                     label = 'ADNI data', s = 12)


        # correctedPred = np.nanmean(dataNNcurrCol.loc[indicesCtl]) * \
        #                 np.ones(dataNNcurrCol.loc[indicesCtl].shape[0])
        # pl.plot(regressorVectorNN[indicesCtl],correctedPred  , c='b')
        pl.title('%s' % data.columns[i])
        pl.legend()
        pl.gca().set_xlim([np.min(regressorVectorArray), np.max(regressorVectorArray)])

        pl.show()

  return data

def normaliseData(dataDfAll, validDf, allBiomkCols):

  # convert biomarkers to Z-scores
  # meanCtl = np.nanmean(dataDfAll[allBiomkCols][np.in1d(dataDfAll['diag'], [CTL, CTL2])],axis=0)
  # stdCtl = np.nanstd(dataDfAll[allBiomkCols][np.in1d(dataDfAll['diag'], [CTL, CTL2])], axis=0)
  # dataDfAll[allBiomkCols] = (np.array(dataDfAll[allBiomkCols]) - meanCtl[None, :]) / (stdCtl[None, :])

  # convert biomarkers to [0,1] interval
  minB = np.nanmin(dataDfAll[allBiomkCols], axis=0)
  maxB = np.nanmax(dataDfAll[allBiomkCols], axis=0)
  dataDfAll[allBiomkCols] = (np.array(dataDfAll[allBiomkCols]) - minB[None, :]) / (maxB - minB)[None, :]
  validDf[allBiomkCols] = (np.array(validDf[allBiomkCols]) - minB[None, :]) / (maxB - minB)[None, :]

  # bring the validation set in the same space as the training dataset.
  # match means and stds of controls in each datasets
  dtiCols = [c for c in validDf.columns if c.startswith('DTI')]
  validDfCtlInd = np.in1d(validDf.diag, [CTL, CTL2])
  meanValid = np.nanmean(validDf.loc[validDfCtlInd,dtiCols], axis = 0)
  stdValid = np.nanstd(validDf.loc[validDfCtlInd,dtiCols], axis = 0)

  trainDfCtlInd = np.in1d(dataDfAll.diag, [CTL, CTL2])
  meanTrain = np.nanmean(dataDfAll.loc[trainDfCtlInd,dtiCols], axis = 0)
  stdTrain = np.nanstd(dataDfAll.loc[trainDfCtlInd,dtiCols], axis = 0)
  validDf[dtiCols] = (validDf[dtiCols].as_matrix() - meanValid[None,:])/(stdValid[None,:])
  validDf[dtiCols] = validDf[dtiCols].as_matrix() * stdTrain[None, :] + meanTrain[None, :]

  print('meanValid', np.nanmean(validDf.loc[validDfCtlInd,dtiCols],axis=0), np.nanstd(validDf.loc[validDfCtlInd,dtiCols],axis=0))
  print('dataDfAll', np.nanmean(dataDfAll.loc[trainDfCtlInd,dtiCols],axis=0), np.nanstd(dataDfAll.loc[trainDfCtlInd,dtiCols],axis=0))
  # print(adsa)

  # print('meanValid', np.nanmean(validDf[allBiomkCols],axis=0))
  # print(adsa)

  # # also normalise the validation set to be in the same space as ADNI.
  # # Note that the previous dataset normalisation doesn't work, because in the training
  # # set there were no DTI biomarkers in dataset 2.
  # for c in range(len(dtiCols)):
  #
  #   stdADNI =np.nanstd(dataDfAll.loc[:, dtiCols[c]])
  #   stdDRC = np.nanstd(validDf.loc[:, dtiCols[c]])
  #   stdRatio = stdDRC / stdADNI
  #   validDf.loc[:, dtiCols[c]] = validDf.loc[:, dtiCols[c]] / stdRatio
  #
  #   meanADNI = np.nanmean(dataDfAll.loc[:, dtiCols[c]])
  #   meanDRC = np.nanmean(validDf.loc[:, dtiCols[c]])
  #   meanDiff = (meanDRC - meanADNI)
  #   validDf.loc[:, dtiCols[c]] = validDf.loc[:, dtiCols[c]] - meanDiff
  #
  #   meanDRC = np.nanmean(validDf.loc[:, dtiCols[c]])
  #   stdDRC = np.nanstd(validDf.loc[:, dtiCols[c]])
  #
  #   # print('ADNI mean', meanADNI)&
  #   # print('ADNI std', stdADNI)
  #   # print('DRC mean', meanDRC)
  #   # print('DRC std', stdDRC)
  #   #
  #   # print(asda)

  return dataDfAll, validDf

def prepareData(finalDataFile, tinyData, addExtraBiomk):

  tadpoleFile = 'TADPOLE_D1_D2.csv'
  # dataDfTadpole = loadTadpole(tadpoleFile)
  # dataDfTadpole.to_csv('tadpoleCleanDf.csv', sep=',', quotechar='"')
  dataDfTadpole = pd.read_csv('tadpoleCleanDf.csv')

  # print(dsa)

  drcFile = 'drcVolsFSX.csv'

  # dataDfDrc = loadDRC(drcFile,columnsFormat=dataDfTadpole.columns)
  # dataDfDrc.to_csv('drcCleanDf.csv')
  dataDfDrc = pd.read_csv('drcCleanDf.csv')

  dataDfAll = pd.concat([dataDfTadpole, dataDfDrc], ignore_index=True)
  dataDfAll = dataDfAll[[x for x in dataDfAll.columns if x != 'Unnamed: 0']]

  # add extra number to RID to ensure no collisions occur with RIDs of other datasets
  dataDfAll['RID'] = dataDfAll['RID']*10 + dataDfAll['dataset']

  dataDfAll.to_csv('tadpoleDrcAll.csv')

  # exact same format as dataDfAll. make deep copy of the DRC data only

  validDf = dataDfAll[dataDfAll.dataset == 2]
  # validDf.drop(validDf.index[idxToDrop], inplace = True)
  validDf = validDf.copy(deep=True)
  validDf.reset_index(drop = True, inplace = True)
  # validDf.sort_index(inplace=True)
  # print('validDf', validDf)
  #validDf = addDRCValidDataMock(validDf) # generate random numbers for now
  validDf = addDRCValidData(validDf) # change to this real dataset one when ready

  outFilePrefix = 'befReg'
  # visValidDf(validDf, outFilePrefix)

  validDf.to_csv('validDf.csv')
  # print(asdas)

  print('validDf', validDf)

  testValidDfConsist(validDf, dataDfAll)

  print(dataDfTadpole.columns.tolist())
  print(dataDfDrc.columns.tolist())
  assert all([x == y for x,y in zip(dataDfTadpole.columns.tolist(), dataDfDrc.columns.tolist())])

  # regress out covariates: age, gender, ICV and dataset
  colsList = dataDfAll.columns.tolist()
  mriCols = [x for x in colsList if x.startswith('Volume')]
  allBiomkCols = dataDfAll.loc[:, 'CDRSB' : ].columns.tolist()



  # print('meanValid', np.nanmean(validDf.loc[validDfCtlInd,dtiCols],axis=0), np.nanstd(validDf.loc[validDfCtlInd,dtiCols],axis=0))
  # print('dataDfAll', np.nanmean(dataDfAll.loc[trainDfCtlInd,dtiCols],axis=0), np.nanstd(dataDfAll.loc[trainDfCtlInd,dtiCols],axis=0))
  # print('trainCTLS', )
  # print(adsa)

  # print(dataDfAll[mriCols])
  # print(dataDfAll['ICV'])
  # print(dataDfAll['diag'])
  # print(adsa)

  # perform correction for both the data and the validation set
  dataDfAll[mriCols], regParamsICV = regressCov(dataDfAll[mriCols],
    dataDfAll['ICV'], dataDfAll['diag'])

  validDf[mriCols] = applyRegFromParams(validDf[mriCols],
    validDf['ICV'], validDf['diag'], regParamsICV, printFigs=False, otherDataToPlot = dataDfAll[mriCols], otherRegVector
    = dataDfAll['ICV'])

  testValidDfConsist(validDf, dataDfAll)

  dataDfAll[allBiomkCols], regParamsAge = regressCov(dataDfAll[allBiomkCols],
    dataDfAll['age'], dataDfAll['diag'])

  validDf[allBiomkCols] = applyRegFromParams(validDf[allBiomkCols],
    validDf['age'], validDf['diag'], regParamsAge,printFigs=False, otherDataToPlot = dataDfAll[allBiomkCols], otherRegVector = dataDfAll['age'])

  dataDfAll[allBiomkCols], regParamsGender = regressCov(dataDfAll[allBiomkCols],
    dataDfAll['gender-0f1m'], dataDfAll['diag'], printFigs=False)

  validDf[allBiomkCols] = applyRegFromParams(validDf[allBiomkCols],
    validDf['gender-0f1m'], validDf['diag'], regParamsGender,printFigs=False, otherDataToPlot = dataDfAll[allBiomkCols], otherRegVector = dataDfAll['gender-0f1m'])

  dataDfAll[allBiomkCols], regParamsDataset = regressCov(dataDfAll[allBiomkCols],
    dataDfAll['dataset'], dataDfAll['diag'], printFigs=False)

  validDf[allBiomkCols] = applyRegFromParams(validDf[allBiomkCols],
    validDf['dataset'], validDf['diag'], regParamsDataset,printFigs=False, otherDataToPlot = dataDfAll[allBiomkCols], otherRegVector = dataDfAll['dataset'])

  # change directionality of decreasing markers: volume, DTI-FA and FDG
  # This is because the model assumes all biomarkers are increasing
  dataDfAll[mriCols] *= -1
  validDf[mriCols] *= -1

  dtiFaCols = [x for x in colsList if x.startswith('DTI FA')]
  # print(dataDfAll[dtiFaCols])
  dataDfAll[dtiFaCols] *= -1
  validDf[dtiFaCols] *= -1

  fdgCols = [x for x in colsList if x.startswith('FDG')]
  dataDfAll[fdgCols] *= -1
  validDf[fdgCols] *= -1

  dataDfAll[['MMSE', 'RAVLT_immediate']] *= -1
  validDf[['MMSE', 'RAVLT_immediate']] *= -1

  print(dataDfAll.shape)
  if tinyData:
    # or try to balance the modalities, currently MRI seems to dominate the fitting.

    hasNonMriImgInd = ~np.isnan(dataDfAll['FDG Temporal']) | (~np.isnan(dataDfAll['DTI FA Temporal'])) \
      | (~np.isnan(dataDfAll['AV45 Temporal'])) | (~np.isnan(dataDfAll['AV1451 Temporal']))
    # print('hasNonMriImgInd', np.sum(hasNonMriImgInd))
    drcDatasetInd = dataDfAll.dataset == 2
    # print('drcDatasetInd', np.sum(drcDatasetInd))
    idxToDrop = np.logical_not(hasNonMriImgInd | drcDatasetInd)
    # print('idxToDrop', np.sum(idxToDrop))
    dataDfAll.drop(dataDfAll.index[idxToDrop], inplace=True)
    dataDfAll.reset_index(drop=True, inplace=True)

    unqRID = np.unique(dataDfAll.RID)
    adniUnqRID = np.unique(dataDfAll.RID[dataDfAll.dataset == 1])
    pcaUnqRID = np.unique(dataDfAll.RID[dataDfAll.dataset == 2])
    print('unqRID', unqRID.shape)
    print('adniUnqRID', adniUnqRID.shape)
    print('pcaUnqRID', pcaUnqRID.shape)
    # print(adas)
    ridToKeep = np.random.choice(adniUnqRID, 230, replace=False)
    ridToKeep = np.concatenate((ridToKeep, pcaUnqRID), axis=0)
    idxToDrop = np.logical_not(np.in1d(dataDfAll.RID, ridToKeep))
    dataDfAll.drop(dataDfAll.index[idxToDrop], inplace=True)
    dataDfAll.reset_index(drop=True, inplace=True)


  # update 6 Aug 2018: moved normalisation after making the data tiny.
  dataDfAll, validDf = normaliseData(dataDfAll, validDf, allBiomkCols)

  # fill in the missing diagnoses
  # print(np.sum(np.isnan(dataDfAll.diag)))
  unqRID = np.unique(dataDfAll.RID)
  # also drop subjects with two or more entries per visit
  idxToKeepDiffAge = np.zeros(dataDfAll.RID.shape[0], bool)
  for s in unqRID:
    idxCurrSubj = np.where(dataDfAll.RID == s)[0]

    monthCurrSubj = dataDfAll.Month_bl[idxCurrSubj]
    diagCurrSubj = dataDfAll.diag[idxCurrSubj]

    idxCurrSubjDiagExists = ~np.isnan(diagCurrSubj)

    if np.sum(idxCurrSubjDiagExists) > 0:
      for v in range(monthCurrSubj.shape[0]):
        if np.isnan(dataDfAll.diag[idxCurrSubj[v]]):
          timeDiffs = monthCurrSubj[idxCurrSubjDiagExists] - monthCurrSubj[idxCurrSubjDiagExists]
          dataDfAll.loc[idxCurrSubj[v], 'diag'] = diagCurrSubj[idxCurrSubjDiagExists][np.argmin(timeDiffs)]
    else:
      dataDfAll.loc[idxCurrSubj, 'diag'] = MCI # only one subj has absolutely no diag. assign MCI

    ageCurrSubj = dataDfAll.age[idxCurrSubj]
    ageUnq, ageInd = np.unique(ageCurrSubj, return_index=True)
    maskCurr = np.in1d(np.array(range(ageCurrSubj.shape[0])), ageInd)
    idxToKeepDiffAge[idxCurrSubj] = maskCurr
    # print('ageInd', ageInd)
    # print('maskCurr', maskCurr)

  # print('idxToKeepDiffAge', idxToKeepDiffAge)
  # print(np.sum(np.logical_not(idxToKeepDiffAge)))

  # print(dataDfAll[np.logical_not(idxToKeepDiffAge)])

  # dataDfAll = dataDfAll[idxToKeepDiffAge]
  dataDfAll.drop(dataDfAll.index[np.logical_not(idxToKeepDiffAge)], inplace=True)
  dataDfAll.reset_index(drop=True, inplace=True)

  # print(np.sum(np.isnan(dataDfAll.diag)))
  # print(dataDfAll)
  # print(ads)

  if tinyData:
    dataDfAll.to_csv('tadpoleDrcRegDataTiny.csv')
  else:
    dataDfAll.to_csv('tadpoleDrcRegData.csv')

  validDf.to_csv('validDfReg.csv')

  # print(dataDfAll.shape)
  # print(ads)

  cogTests = dataDfAll.loc[:,'CDRSB' : 'FAQ' ].columns.tolist()

  for c in cogTests:
    for m in mriCols:
      nnInd = ~(np.isnan(dataDfAll.loc[:, c]) | np.isnan(dataDfAll.loc[:, m]))
      (r, p) = scipy.stats.pearsonr(dataDfAll.loc[nnInd, c], dataDfAll.loc[nnInd, m])

      print('%s - %s: r %f   pval %e' % (c, m, r, p))


  selectedBiomk = dataDfAll.loc[:, 'Volume Cingulate' : ].columns.tolist()
  if addExtraBiomk:
    selectedBiomk += ['ADAS13', 'CDRSB', 'RAVLT_immediate']

  # print(dataDfAll.dtypes)
  for c in selectedBiomk:
    dataDfAll[c] = dataDfAll[c].astype(np.float128) # increase precision of floats to 128
    validDf[c] = validDf[c].astype(np.float128)




  # print(dataDfAll.dtypes)
  # print(adsa)

  testValidDfConsist(validDf, dataDfAll)

  X, Y, RID, list_biomarkers, diag, visitIndices = \
    auxFunc.convert_table_marco(dataDfAll, list_biomarkers=selectedBiomk)




  # now drop all the mri values, which were used for testing consistency
  # and only keep the DTI. Don't remove the MRI cols though, needs to be in
  # same format as dataDfAll
  # UPDATE May 2018: No, don't drop MRI values. I need them for prediction of DTI vals
  # from simpler linear model
  # print('validDf', validDf.loc[:, mriCols])
  # validDf.loc[:,mriCols] = np.nan
  # print('validDf', validDf.loc[:,mriCols])

  outFilePrefix = 'afterReg'
  visValidDf(validDf, outFilePrefix)
  # print(ads)

  Xvalid, Yvalid, RIDvalid, _, diagValid, visitIndicesValid = \
    auxFunc.convert_table_marco(validDf, list_biomarkers = selectedBiomk)

  print('validDf.RID', validDf.RID)
  print('RIDvalid', len(RIDvalid))
  # print(ads)

  ds = dict(X=X, Y=Y, RID=RID, list_biomarkers=list_biomarkers,visitIndices=visitIndices,
    dataDfAll=dataDfAll, regParamsICV=regParamsICV,
    regParamsAge=regParamsAge, regParamsGender=regParamsGender,
    regParamsDataset=regParamsDataset, diag=diag, Xvalid=Xvalid, Yvalid=Yvalid,
    RIDvalid=RIDvalid, diagValid=diagValid, visitIndicesValid=visitIndicesValid)
  pickle.dump(ds, open(finalDataFile, 'wb'), protocol=pickle.HIGHEST_PROTOCOL)

  # print('RID', RID)
  # print('X', len(X), len(X[0]))
  # print('Y', len(Y), len(Y[0]))
  # print(adsa)

def testValidDfConsist(validDf, dataDfAll):
  sID = np.unique(dataDfAll.scanID)[3]
  assert not np.isnan(sID)
  idxValid = np.where(validDf.scanID == sID)[0][0]
  idxData = np.where(dataDfAll.scanID == sID)[0][0]
  print('validDf.loc[3, Volume Parietal]', validDf.loc[idxValid, 'Volume Parietal'])
  print('dataDfAll.loc[3, Volume Parietal]', dataDfAll.loc[idxData, 'Volume Parietal'])
  print('validDf.loc[3, DTI FA Parietal]', validDf.loc[idxValid, 'DTI FA Parietal'])
  print('dataDfAll.loc[3, DTI FA Parietal]', dataDfAll.loc[idxData, 'DTI FA Parietal'])
  print('idxValid', idxValid)
  print('idxData', idxData)
  assert validDf.at[idxValid, 'Volume Parietal'] == \
    dataDfAll.loc[idxData, 'Volume Parietal']
  assert validDf.at[idxValid, 'DTI FA Parietal'] != \
    dataDfAll.at[idxData, 'DTI FA Parietal']