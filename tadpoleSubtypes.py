import sys
import numpy
import numpy as np
import colorsys
from socket import gethostname
import time
import argparse
import os
import colorsys
import copy
import pandas as pd
import pickle
import random
import auxFunc
import scipy
import scipy.io as sio
import scipy.stats
from scipy.interpolate import UnivariateSpline

# sys.path.append(os.path.abspath("../diffEqModel/"))


parser = argparse.ArgumentParser(description='Launches voxel-wise/point-wise DPM on ADNI'
                                             'using cortical thickness maps derived from MRI')

parser.add_argument('--agg', dest='agg', type=int, default=0,
  help='agg=1 => plot figures without using Xwindows, for use on cluster where the plots cannot be displayed '
       ' agg=0 => plot with Xwindows (for use on personal machine)')

parser.add_argument('--runIndex', dest='runIndex', type=int,
  default=1, help='index of run instance/process .. for cross-validation')

parser.add_argument('--nrProc', dest='nrProc', type=int,
  default=1, help='# of processes')

parser.add_argument('--modelToRun', dest='modelToRun', type=int,
  help='index of model to run')

parser.add_argument('--cluster', action="store_true",
  help='need to include this flag if runnin on cluster')

parser.add_argument('--nrRows', dest='nrRows', type=int,
  help='nr of subfigure rows to plot at every iteration')

parser.add_argument('--nrCols', dest='nrCols', type=int,
  help='nr of subfigure columns to plot at every iteration')

parser.add_argument('--penalty', dest='penalty', type=float,
  help='penalty value for non-monotonic trajectories. between 0 (no effect) and 10 (strong effect). ')

parser.add_argument('--regData', action="store_true", default=False,
  help=' add this flag to regenerate the data')

parser.add_argument('--runPartStd', dest='runPartStd', default='RR',
  help=' choose whether to (R) run or (L) load from the checkpoints: '
  'either LL, RR, LR or RL. ')

parser.add_argument('--tinyData', action="store_true", default=False,
  help=' only run on a tiny subset of the data: around 200/1980 subjects')


args = parser.parse_args()

if args.agg:
  # print(matplotlib.__version__)
  import matplotlib
  matplotlib.use('Agg')
  # print(asds)

import genSynthData
import ParHierModel
import Plotter
from auxFunc import *
import evaluationFramework
from matplotlib import pyplot as pl
# from env import *

from drcValidFuncs import *
from tadpoleDataLoader import *
from drcDataLoader import *
import tadpoleDrcPrepData


# TADPOLE SuStain Subgroups
# CTL = 1
CTL1 = 1
CTL2 = 2
CTL3 = 3

SBG1 = 4
SBG2 = 5
SBG3 = 6

CTL1_LIN = 7
CTL2_LIN = 8
CTL3_LIN = 9

SBG1_LIN = 10
SBG2_LIN = 11
SBG3_LIN = 12

CTL1_DKT = 13
CTL2_DKT = 14
CTL3_DKT = 15

SBG1_DKT = 16
SBG2_DKT = 17
SBG3_DKT = 18


hostName = gethostname()
if hostName == 'razvan-Inspiron-5547':
  height = 350
else: #if hostName == 'razvan-Precision-T1700':
  height = 450

# if hostName == 'razvan-Inspiron-5547':
#   freesurfPath = '/usr/local/freesurfer-6.0.0'
# elif hostName == 'razvan-Precision-T1700':
#   freesurfPath = '/usr/local/freesurfer-6.0.0'
# elif args.cluster:
#   freesurfPath = '/home/rmarines/src/freesurfer-6.0.0'
# elif hostName == 'planell-VirtualBox':
#   freesurfPath = ""
# else:
#   freesurfPath = ""
#   # raise ValueError('You need to add Freesurfer paths specific to this PC')



#                      DKT     OTHER_MODEL        VALID           TRAINING
#                     circle    triangle      diagonal cross       square
#
# ADNI CTL  green
#      MCI  orange
#      AD   red
# DRC  CTL  yellow
#      PCA  magenta
#      AD   blue

pd.set_option('display.max_rows', 100)
pd.set_option('display.max_columns', 50)
pd.set_option('display.width', 5000)

def initTrajParams():


  plotTrajParams = {}
  plotTrajParams['SubfigTrajWinSize'] = (1600, 900)
  plotTrajParams['nrRows'] = args.nrRows
  plotTrajParams['nrCols'] = args.nrCols
  plotTrajParams['diagColors'] = {CTL1: 'g', CTL2: 'pink', CTL3: 'skyblue', SBG1: 'chocolate', SBG2: 'r',
    SBG3: 'y', CTL1_LIN: 'aqua', CTL2_LIN: 'gold', CTL3_LIN: 'yellow',
    SBG1_LIN: 'm', SBG2_LIN: 'darkslategray', SBG3_LIN: 'darkblue',
    CTL1_DKT: 'orange', CTL2_DKT: 'violet', CTL3_DKT: 'brown',
    SBG1_DKT: 'darkgoldenrod', SBG2_DKT: 'deeppink', SBG3_DKT: 'teal'}
  plotTrajParams['diagScatterMarkers'] = {CTL1: 's', CTL2: 's', CTL3: 's', SBG1: 's', SBG2: 's', SBG3: 's',
    CTL1_LIN: '^', CTL2_LIN: '^', CTL3_LIN: '^',
    SBG1_LIN: '^', SBG2_LIN: '^', SBG3_LIN: '^',
    CTL1_DKT: 'o', CTL2_DKT: 'o', CTL3_DKT: 'o', SBG1_DKT: 'o', SBG2_DKT: 'o',
    SBG3_DKT: 'o'}
  plotTrajParams['legendCols'] = 4
  # ['Hippocampal', 'Cortical', 'Subcortical']
  plotTrajParams['diagLabels'] = {CTL1: 'CTL1', CTL2: 'CTL2', CTL3: 'CTL3',
    SBG1: 'SBG1', SBG2: 'SBG2', SBG3: 'SBG3',
    CTL1_LIN: 'CTL1 - Lin', CTL2_LIN: 'CTL2 - Lin',
    CTL3_LIN: 'CTL3 - Lin',
    SBG1_LIN: 'SBG1 - Lin', SBG2_LIN: 'SBG2 - Lin',
    SBG3_LIN: 'SBG3 - Lin',
    CTL1_DKT: 'CTL1 - DKT', CTL2_DKT: 'CTL2 - DKT', CTL3_DKT: 'CTL3 - DKT',
    SBG1_DKT: 'SBG1 - DKT', SBG2_DKT: 'SBG2 - DKT', SBG3_DKT: 'SBG3 - DKT'}

  # plotTrajParams['freesurfPath'] = freesurfPath
  # plotTrajParams['blenderPath'] = blenderPath
  plotTrajParams['isSynth'] = False
  plotTrajParams['padTightLayout'] = 0.0

  if args.agg:
    plotTrajParams['agg'] = True
  else:
    plotTrajParams['agg'] = False

  return plotTrajParams

def visDataHist(dataDfAll):

  unqDiags = np.unique(dataDfAll.diag)
  biomks = dataDfAll.loc[:, 'CDRSB':].columns.tolist()
  for b in range(len(biomks)):

    fig = pl.figure(5)
    fig.clf()
    for d in unqDiags:
      pl.hist(dataDfAll.loc[dataDfAll.diag == d, biomks[b]].dropna(), bins=15,
        color=plotTrajParams['diagColors'][d], label=plotTrajParams['diagLabels'][d], alpha=0.5)

    pl.legend(loc='west')
    pl.title(biomks[b])

    fig.show()
    os.system('mkdir -p resfiles/tad-drc')
    fig.savefig('resfiles/tad-drc/%d_%s.png' % (b, biomks[b]))


def normaliseData(dataDfAll, allBiomkCols):
  # convert biomarkers to [0,1] interval
  nrBiomk = len(allBiomkCols)
  minB = np.zeros(nrBiomk)
  maxB = np.zeros(nrBiomk)
  for b in range(nrBiomk):

    x = dataDfAll[allBiomkCols[b]].values
    x = x[~np.isnan(x)]
    sortedVals = np.sort(x, axis=0)
    nrSortedVals = sortedVals.shape[0]
    if sortedVals.shape[0] < 5:
      minB[b] = np.nanmin(sortedVals)
      maxB[b] = np.nanmax(sortedVals)
    else:
      minB[b] = sortedVals[max(1,int(0.03 * nrSortedVals))]
      maxB[b] = sortedVals[min(nrSortedVals-1,int(0.97 * nrSortedVals))]
      # print(adsa)

  dataDfAll[allBiomkCols] = (np.array(dataDfAll[allBiomkCols]) - minB[None, :]) / (maxB - minB)[None, :]

  return dataDfAll

def prepareData(finalDataFile, tinyData, addExtraBiomk):

  tadpoleFile = 'TADPOLE_D1_D2.csv'
  # dataDfTadpole = loadTadpole(tadpoleFile)
  # dataDfTadpole.to_csv('tadpoleCleanDf.csv', sep=',', quotechar='"')
  dataDfTadpole = pd.read_csv('data_processed/tadpoleCleanDf.csv')

  print('dataDfTadpole', dataDfTadpole)
  print('RID', dataDfTadpole.RID)

  subtypesDf = pd.read_csv('data/Subtypes_SuStaIn_TADPOLE.csv')
  print('subtypesDf', subtypesDf)
  # print(ads)

  unqSubjRid = np.unique(dataDfTadpole.RID)
  nrUnqSubj = unqSubjRid.shape[0]
  dataDfTadpole.subtype = np.nan * np.ones(dataDfTadpole.shape[0])
  dataDfTadpole.subtypeCtl123 = np.nan * np.ones(dataDfTadpole.shape[0])
  for s in range(nrUnqSubj):
    markCurrSub = dataDfTadpole.RID == unqSubjRid[s]
    dataDfTadpole.subtype[markCurrSub] = subtypesDf.Subtype[subtypesDf.RID == unqSubjRid[s]]
    # print(subtypesDf.loc[
    #    subtypesDf.RID == unqSubjRid[s], 'ProbSubtype1' : 'ProbSubtype3'])
    dataDfTadpole.subtypeCtl123[markCurrSub] = 1 + np.argmax(subtypesDf.loc[
       subtypesDf.RID == unqSubjRid[s], 'ProbSubtype1' : 'ProbSubtype3'].values)


  dataDfTadpole.dataset = dataDfTadpole.subtypeCtl123
  # +1 in order to map subgroup 0 (i.e. CTL) to 1 as in env.py

  # first set the diagnoses of patinets, leave controls for later
  dataDfTadpole.diag = dataDfTadpole.subtype + 3

  # set the diag of controls
  dataDfTadpole.diag[dataDfTadpole.subtype == 0] = dataDfTadpole.dataset[
    dataDfTadpole.subtype == 0]

  print()
  print('dataDfTadpole.diag', dataDfTadpole.diag)
  assert np.max(dataDfTadpole.diag) == 6
  assert np.min(dataDfTadpole.diag) == 1
  # print(adsa)

  # map controls to the three datasets based on sustain probability.
  # there will be 3 datasets:
  # 1. CTL-hippocampal + patients-hippocampal
  # 2. CTL-cortical + patients-cortical
  # 3. CTL-subcortical + patients-subcortical

  # dataDfTadpole.dataset[dataDfTadpole.diag == CTL] = dataDfTadpole.subtype[dataDfTadpole.diag == CTL]

  # regress out covariates: age, gender, ICV and dataset
  colsList = dataDfTadpole.columns.tolist()
  mriCols = [x for x in colsList if x.startswith('Volume')]
  allBiomkCols = dataDfTadpole.loc[:, 'CDRSB' : ].columns.tolist()


  # perform bias correction
  dataDfTadpole[mriCols], regParamsICV = tadpoleDrcPrepData.regressCov(dataDfTadpole[mriCols],
    dataDfTadpole['ICV'], dataDfTadpole['diag'])

  dataDfTadpole[allBiomkCols], regParamsAge = tadpoleDrcPrepData.regressCov(dataDfTadpole[allBiomkCols],
    dataDfTadpole['age'], dataDfTadpole['diag'])

  dataDfTadpole[allBiomkCols], regParamsGender = tadpoleDrcPrepData.regressCov(dataDfTadpole[allBiomkCols],
    dataDfTadpole['gender-0f1m'], dataDfTadpole['diag'], printFigs=False)

  # dataDfTadpole[allBiomkCols], regParamsDataset = tadpoleDrcPrepData.regressCov(dataDfTadpole[allBiomkCols],
  #   dataDfTadpole['dataset'], dataDfTadpole['diag'], printFigs=False)

  # change directionality of decreasing markers: volume, DTI-FA and FDG
  # This is because the model assumes all biomarkers are increasing
  dataDfTadpole[mriCols] *= -1

  dtiFaCols = [x for x in colsList if x.startswith('DTI FA')]
  # print(dataDfTadpole[dtiFaCols])
  dataDfTadpole[dtiFaCols] *= -1

  fdgCols = [x for x in colsList if x.startswith('FDG')]
  dataDfTadpole[fdgCols] *= -1

  dataDfTadpole[['MMSE', 'RAVLT_immediate']] *= -1


  print('nan(diag)', np.sum(np.isnan(dataDfTadpole.diag)))
  dataDfTadpole.drop(dataDfTadpole.index[
    np.isnan(dataDfTadpole.diag)], inplace=True)
  dataDfTadpole.reset_index(drop=True, inplace=True)
  print('nan(diag)', np.sum(np.isnan(dataDfTadpole.diag)))
  # print(ads)


  unqRID = np.unique(dataDfTadpole.RID)
  # drop subjects with two or more entries per visit
  idxToKeepDiffAge = np.zeros(dataDfTadpole.RID.shape[0], bool)
  for s in unqRID:
    idxCurrSubj = np.where(dataDfTadpole.RID == s)[0]

    monthCurrSubj = dataDfTadpole.Month_bl[idxCurrSubj]
    diagCurrSubj = dataDfTadpole.diag[idxCurrSubj]

    idxCurrSubjDiagExists = ~np.isnan(diagCurrSubj)

    ageCurrSubj = dataDfTadpole.age[idxCurrSubj]
    ageUnq, ageInd = np.unique(ageCurrSubj, return_index=True)
    maskCurr = np.in1d(np.array(range(ageCurrSubj.shape[0])), ageInd)
    idxToKeepDiffAge[idxCurrSubj] = maskCurr
    # print('ageInd', ageInd)
    # print('maskCurr', maskCurr)


  # dataDfTadpole = dataDfTadpole[idxToKeepDiffAge]
  dataDfTadpole.drop(dataDfTadpole.index[np.logical_not(idxToKeepDiffAge)], inplace=True)
  dataDfTadpole.reset_index(drop=True, inplace=True)




  cogTests = dataDfTadpole.loc[:,'CDRSB' : 'FAQ' ].columns.tolist()

  for c in cogTests:
    for m in mriCols:
      nnInd = ~(np.isnan(dataDfTadpole.loc[:, c]) | np.isnan(dataDfTadpole.loc[:, m]))
      (r, p) = scipy.stats.pearsonr(dataDfTadpole.loc[nnInd, c], dataDfTadpole.loc[nnInd, m])

      print('%s - %s: r %f   pval %e' % (c, m, r, p))


  selectedBiomk = dataDfTadpole.loc[:, 'Volume Cingulate' : ].columns.tolist()
  if addExtraBiomk:
    selectedBiomk += ['ADAS13', 'CDRSB', 'RAVLT_immediate', 'MMSE', 'FAQ']

  for c in selectedBiomk:
    dataDfTadpole[c] = dataDfTadpole[c].astype(np.float128) # increase precision of floats to 128


  #### make a tiny dataset for testing. When model is ready, run on full data ###
  print(dataDfTadpole.shape)
  nrModPerSubj = (~np.isnan(dataDfTadpole['FDG Temporal'])).astype(int) + (~np.isnan(dataDfTadpole['DTI FA Temporal'])).astype(int) \
                    + (~np.isnan(dataDfTadpole['AV45 Temporal'])).astype(int) + (~np.isnan(dataDfTadpole['AV1451 Temporal'])).astype(int)

  hasMriPlusOneOtherImgInd = np.logical_and((~np.isnan(dataDfTadpole['Volume Temporal'])), nrModPerSubj >= 1)

  print('nrModPerSubj', nrModPerSubj)
  # print(das)

  print('dataDfTadpole s2', dataDfTadpole.shape)

  idxToDrop = np.logical_not(hasMriPlusOneOtherImgInd)
  # print('idxToDrop', np.sum(idxToDrop))
  dataDfTadpole.drop(dataDfTadpole.index[idxToDrop], inplace=True)
  dataDfTadpole.reset_index(drop=True, inplace=True)

  # print(dataDfTadpole.shape)
  # print(adsa)

  unqRID = np.unique(dataDfTadpole.RID)
  g1RID = np.unique(dataDfTadpole.RID[dataDfTadpole.dataset == 1])
  g2RID = np.unique(dataDfTadpole.RID[dataDfTadpole.dataset == 2])
  g3RID = np.unique(dataDfTadpole.RID[dataDfTadpole.dataset == 3])

  print('g1RID', g1RID.shape)
  print('g2RID', g2RID.shape)
  print('g3RID', g3RID.shape)
  # print(adsa)

  minNr = np.min([g1RID.shape[0], g2RID.shape[0], g3RID.shape[0]])
  if tinyData:
    nrSubjToKeep = 60
  else:
    nrSubjToKeep = minNr


  assert nrSubjToKeep <= minNr

  np.random.seed(1)
  ridToKeepG1 = np.random.choice(g1RID, nrSubjToKeep, replace=False)
  ridToKeepG2 = np.random.choice(g2RID, nrSubjToKeep, replace=False)
  ridToKeepG3 = np.random.choice(g3RID, nrSubjToKeep, replace=False)
  # ridToKeepG4 = np.random.choice(g4RID, nrSubjToKeep, replace=False)

  print('dataDfTadpole s3', dataDfTadpole.shape)

  ridToKeep = np.concatenate((ridToKeepG1, ridToKeepG2, ridToKeepG3), axis=0)
  idxToDrop = np.logical_not(np.in1d(dataDfTadpole.RID, ridToKeep))
  dataDfTadpole.drop(dataDfTadpole.index[idxToDrop], inplace=True)
  dataDfTadpole.reset_index(drop=True, inplace=True)

  print('dataDfTadpole s4', dataDfTadpole.shape)

  # remove AV1451 due to very sparse data
  selectedBiomk = [x for x in selectedBiomk if not x.startswith('AV1451')]
  allBiomkCols = [x for x in allBiomkCols if not x.startswith('AV1451')]
  dataDfTadpole = dataDfTadpole.loc[:, 'RID': 'AV45 Temporal']

  print(dataDfTadpole)

  # print(np.logical_and(dataDfTadpole.Month_bl == 0, dataDfTadpole.dataset=1))

  dfBl = dataDfTadpole.loc[dataDfTadpole.Month_bl == 0, :]
  print('Group & Number & Visits & Age & Gender (\\%%F)\\\\')
  grNames = ['Controls (Hippocampal)', 'Controls (cortical)', 'Controls (subcortical)',
             'AD (Hippocampal)', 'AD (cortical)', 'AD (subcortical)']
  for i in [1,4,2,5,3,6]:
    # print('dfBl\n', dfBl)
    dfCurr = dfBl.loc[dfBl.diag == i, :]
    # print('dfCurr\n',dfCurr)
    # print(np.nanmean(dfCurr.age))
    # print(np.nanmean(dfCurr.loc[:,'gender-0f1m']))
    dfGrByRID = dataDfTadpole.loc[dataDfTadpole.diag == i,:].groupby(['RID'], as_index=False)
    avgNrVisits = np.mean(dfGrByRID.count().age)
    stdNrVisits = np.std(dfGrByRID.count().age)


    print('%s & %d & %.1f $\pm$ %.1f & %.1f $\pm$ %.1f & %d(\\%%)\\\\' % (grNames[i-1], dfCurr.shape[0],
      avgNrVisits, stdNrVisits, np.nanmean(dfCurr.age), np.nanstd(dfCurr.age),
      100 * (1 - np.nanmean(dfCurr.loc[:,'gender-0f1m']))))




  # update 6 Aug 2018: moved normalisation after making the data tiny.
  dataDfTadpole = normaliseData(dataDfTadpole, allBiomkCols)



  # print('dataDfTadpole',dataDfTadpole.shape)
  # print(dass)

  # now generate three training datasets:
  # X1/Y1: Subtype0+1 full data, Subtypes 2&3 MRI only
  # X2/Y2: Subtype0+2 full data, Subtypes 1&3 MRI only
  # X3/Y3: Subtype0+3 full data, Subtypes 1&2 MRI only

  dataDfTrainAll = [0, 0, 0]
  XtrainAll = [0 ,0 ,0]
  YtrainAll = [0, 0, 0]
  RIDtrainAll = [0, 0, 0]
  diagTrainAll = [0, 0, 0]
  visitIndicesTrainAll = [0, 0, 0]
  nrSubgr = 3



  dataDfTadpole.to_csv(finalDataFile.split('.')[0] + '.csv',index=False)

  # remove the non-MRI imaging data for the subjects in the other diseases
  for s in range(nrSubgr):


    dataDfTrainAll[s] = dataDfTadpole.copy()
    maskSubjMisData = np.logical_not(dataDfTrainAll[s].dataset == s+1)
    print('maskSubjMisData',maskSubjMisData)
    print(np.sum(maskSubjMisData))

    nonMriColumns = ['DTI FA Cingulate', 'DTI FA Frontal',
         'DTI FA Hippocampus', 'DTI FA Occipital', 'DTI FA Parietal',
         'DTI FA Temporal', 'FDG Cingulate', 'FDG Frontal', 'FDG Hippocampus',
         'FDG Occipital', 'FDG Parietal', 'FDG Temporal', 'AV45 Cingulate',
         'AV45 Frontal', 'AV45 Hippocampus', 'AV45 Occipital', 'AV45 Parietal',
         'AV45 Temporal'
         # ,'AV1451 Cingulate', 'AV1451 Frontal', 'AV1451 Hippocampus', 'AV1451 Occipital',
         # 'AV1451 Parietal', 'AV1451 Temporal'
                    ]
    dataDfTrainAll[s].loc[maskSubjMisData, nonMriColumns] = np.nan

    dataDfTrainAll[s].to_csv(finalDataFile.split('.')[0] + ('Gr%d.csv' % (s+1)),index=False)

    print('nr with FDG', np.sum(np.logical_not(np.isnan(dataDfTrainAll[s].loc[:,'FDG Temporal']))))
    print('nr with FDG all data', np.sum(np.logical_not(np.isnan(dataDfTadpole.loc[:, 'FDG Temporal']))))

    XtrainAll[s], YtrainAll[s], RIDtrainAll[s], _, diagTrainAll[s], visitIndicesTrainAll[s] = \
      auxFunc.convert_table_marco(dataDfTrainAll[s], list_biomarkers=selectedBiomk)

    assert len(XtrainAll[s]) > 0
    assert len(YtrainAll[s]) > 0

  # import pdb
  # pdb.set_trace()


  Xvalid, Yvalid, RIDvalid, list_biomarkers, diagValid, visitIndicesValid = \
    auxFunc.convert_table_marco(dataDfTadpole, list_biomarkers=selectedBiomk)

  assert len(Xvalid) > 0
  assert len(Yvalid) > 0

  ds = dict(XtrainAll=XtrainAll, YtrainAll=YtrainAll, RIDtrainAll=RIDtrainAll,
    list_biomarkers=list_biomarkers,visitIndicesTrainAll=visitIndicesTrainAll,
    diagTrainAll=diagTrainAll, dataDfTrainAll=dataDfTrainAll,
    regParamsICV=regParamsICV, regParamsAge=regParamsAge,
    regParamsGender=regParamsGender,
    Xvalid=Xvalid, Yvalid=Yvalid, RIDvalid=RIDvalid,
    diagValid=diagValid, visitIndicesValid=visitIndicesValid, dataDfTadpole=dataDfTadpole)
  pickle.dump(ds, open(finalDataFile, 'wb'), protocol=pickle.HIGHEST_PROTOCOL)

  # print(das)

def main():

  # global plotTrajParams
  plotTrajParams = initTrajParams()

  addExtraBiomk = True

  np.random.seed(1)
  random.seed(1)
  pd.set_option('display.max_columns', 50)
  tinyData = args.tinyData

  finalDataFile = 'tadSubtypes.npz'
  expName = 'tadSubtypes'


  if args.tinyData:
    finalDataFile = finalDataFile.split('.')[0] + 'Tiny.npz'
    expName = expName.split('.')[0] + 'Tiny'

  if addExtraBiomk:
    finalDataFile = finalDataFile.split('.')[0] + 'Cog.npz'
    expName = expName.split('.')[0] + 'Cog'

  regenerateData = (not os.path.isfile(finalDataFile)) or args.regData
  if regenerateData:
    prepareData(finalDataFile, tinyData, addExtraBiomk)

  ds = pickle.load(open(finalDataFile, 'rb'))

  X = ds['XtrainAll'][0]
  Y = ds['YtrainAll'][0]
  RID = np.array(ds['RIDtrainAll'][0], int)
  labels = ds['list_biomarkers']
  diag = ds['diagTrainAll']

  outFolder = 'resfiles/'

  params = {}

  if ds['list_biomarkers'][-1].startswith('AV1451'):
    nrBiomkInFuncUnits = 5
  else:
    nrBiomkInFuncUnits = 4

  nrDis = 3 # nr of diseases
  params['nrDis'] = nrDis

  # change the order of the functional units so that the hippocampus and occipital are fitted first
  unitPermutation = [5,3,2,1,4,0]

  nrFuncUnits = 6  # add the 3 extra cog markers to a unique functional unit
  mapBiomkToFuncUnits = np.array((unitPermutation * nrBiomkInFuncUnits))
  nrExtraBiomk = 0

  if addExtraBiomk:
    nrExtraBiomk = 5
    nrFuncUnits += nrExtraBiomk  # add the 3 extra cog markers to a unique functional unit
    mapBiomkToFuncUnits = np.array((unitPermutation * nrBiomkInFuncUnits)
                                   + list(range(nrFuncUnits-nrExtraBiomk, nrFuncUnits)))

  unitNames = [l.split(' ')[-1] for l in labels]
  unitNames = [unitNames[i] for i in unitPermutation]
  if addExtraBiomk:
    extraBiomkNames = ['ADAS13', 'CDRSB', 'RAVLT', 'MMSE', 'FAQ']
    unitNames += extraBiomkNames
    assert len(extraBiomkNames) == nrExtraBiomk

  nrBiomk = mapBiomkToFuncUnits.shape[0]
  biomkInFuncUnit = [0 for u in range(nrFuncUnits + 1)]
  for u in range(nrFuncUnits):
    biomkInFuncUnit[u] = np.where(mapBiomkToFuncUnits == u)[0]

  # if addExtraBiomk:
  #   # add extra entry with other biomks to be added in the disease models
  #   biomkInFuncUnit[nrFuncUnits] = np.array([nrBiomk-3, nrBiomk-2, nrBiomk-1])
  # else:
  biomkInFuncUnit[nrFuncUnits] = np.array([])  # need to leave this as empty list

  plotTrajParams['biomkInFuncUnit'] = biomkInFuncUnit
  plotTrajParams['labels'] = labels
  plotTrajParams['nrRowsFuncUnit'] = 3
  plotTrajParams['nrColsFuncUnit'] = 4
  plotTrajParams['colorsTrajBiomkB'] = [colorsys.hsv_to_rgb(hue, 1, 1) for hue in
    np.linspace(0, 1, num=nrBiomk, endpoint=False)]
  plotTrajParams['colorsTrajUnitsU'] = [colorsys.hsv_to_rgb(hue, 1, 1) for hue in
    np.linspace(0, 1, num=nrFuncUnits, endpoint=False)]
  plotTrajParams['nrBiomk'] = nrBiomk
  params['nrBiomk'] = nrBiomk

  plotTrajParams['yNormMode'] = 'zScoreTraj'
  # plotTrajParams['yNormMode'] = 'zScoreEarlyStageTraj'
  # plotTrajParams['yNormMode'] = 'unscaled'

  # if False, plot estimated traj. in separate plot from true traj.
  plotTrajParams['allTrajOverlap'] = False

  params['nrFuncUnitsImgOnly'] = nrFuncUnits - nrExtraBiomk
  params['unitNames'] = unitNames
  params['runIndex'] = args.runIndex
  params['nrProc'] = args.nrProc
  params['cluster'] = args.cluster
  params['plotTrajParams'] = plotTrajParams
  params['penaltyUnits'] = args.penalty
  params['penaltyDis'] = args.penalty
  params['nrFuncUnits'] = nrFuncUnits
  params['biomkInFuncUnit'] = biomkInFuncUnit
  params['mapBiomkToFuncUnits'] = mapBiomkToFuncUnits
  params['labels'] = labels
  params['nrExtraBiomk'] = nrExtraBiomk

  # the validation set will contain everything, while the training set will
  # have missing biomarkers. X,Y will be created in runAllExpTadpoleSubtypes
  params['XtrainAll'] = ds['XtrainAll']
  params['YtrainAll'] = ds['YtrainAll']
  params['RIDtrainAll'] = ds['RIDtrainAll']
  params['diagTrainAll'] = ds['diagTrainAll']
  params['visitIndicesTrainAll'] = ds['visitIndicesTrainAll']
  params['diag'] = params['diagTrainAll'][0]

  params['Xvalid'] = ds['Xvalid']
  params['Yvalid'] = ds['Yvalid']
  params['RIDvalid'] = ds['RIDvalid']
  params['diagValid'] = ds['diagValid']
  params['visitIndicesValid'] = ds['visitIndicesValid']
  params['plotTrajParams']['diag'] = params['diagValid']

  params['dataDfTadpole'] = ds['dataDfTadpole'] # WARNING. this contains both train+test data
  params['dataDfTrainAll'] = ds['dataDfTrainAll'] # only training data, one df for each disease

  # by default we have no priors
  params['priors'] = None

  ############# set priors for specific models ################

  # params['priors'] = dict(prior_length_scale_mean_ratio=0.33, # mean_length_scale = (self.maxX-self.minX)/3
  #     prior_length_scale_std=1e-4, prior_sigma_mean=2,prior_sigma_std = 1e-3,
  #     prior_eps_mean = 1, prior_eps_std = 1e-2)
  # params['priors'] = dict(prior_length_scale_mean_ratio=0.9,  # mean_length_scale = (self.maxX-self.minX)/3
  #                             prior_length_scale_std=1e-4, prior_sigma_mean=3, prior_sigma_std=1e-3,
  #                             prior_eps_mean=0.1, prior_eps_std=1e-6)

  params['priorsUnitModelsMarcoModel'] = [dict(prior_length_scale_mean_ratio=0.05,  # mean_length_scale = (self.maxX-self.minX)/3
                              prior_length_scale_std=1e-6, prior_sigma_mean=0.5, prior_sigma_std=1e-3,
                              prior_eps_mean=0.1, prior_eps_std=1e-6) for u in range(nrFuncUnits)]

  transitionTimePriorMean = 1 # in DPS 0-1 space, prior mean
  transitionTimePriorMin = 0.9
  transitionTimePriorMax = 1.1

  bPriorShape, bPriorRate = getGammShapeRateFromTranTime(
    transitionTimePriorMean, transitionTimePriorMin, transitionTimePriorMax)

  # standard sigmoid priors
  params['priorsDisModels'] = [dict(meanA=1, stdA=1e-20, meanD=0, stdD=1e-20,
    shapeB=2, rateB=2, timeShiftStd=20000) for d in range(nrDis)]
  params['priorsUnitModels'] = [dict(meanA=1, stdA=1e-5, meanD=0, stdD=1e-5,
    shapeB=bPriorShape, rateB=bPriorRate, timeShiftStd=20000) for u in range(nrFuncUnits-nrExtraBiomk)]

  if nrExtraBiomk > 0:
    params['priorsUnitModelsLinear'] = [dict(meanA=1, stdA=0.1, meanB=0, stdB=0.1, timeShiftStd=20000) for u in
                                       range(nrExtraBiomk)]
    params['priorsUnitModels'] += params['priorsUnitModelsLinear']


  bPriorShapeNoDKT, bPriorRateNoDKT = getGammShapeRateFromTranTime(
    transitionTimePriorMean=50, transitionTimePriorMin=40, transitionTimePriorMax=60)
  params['priorsNoDKTSigmoid'] = dict(meanA=1, stdA=1e-3, meanD=0, stdD=1e-3,
    shapeB=bPriorShapeNoDKT, rateB=bPriorRateNoDKT, timeShiftStd=20000)

  ######################

  nrBiomkDisModel = nrFuncUnits
  params['nrBiomkDisModel'] = nrBiomkDisModel

  if addExtraBiomk:
    params['plotTrajParams']['unitNames'] = unitNames + labels[-3:]
  else:
    params['plotTrajParams']['unitNames'] = unitNames


  # map which subtypes belong to which disease.
  # disease 1: subtype 0+1 - Hippocampal
  # disease 2: subtype 0+2 - cortical
  # disease 3: subtype 0+3 - subcortical
  # note subtype 0 are controls
  params['diagsSetInDis'] = [np.array([CTL1,SBG1]), np.array([CTL2,SBG2]), np.array([CTL3,SBG3])]
  params['disLabels'] = ['Hippocampal', 'Cortical', 'Subcortical']
  params['expNamesSubgr'] = [x[:2] for x in params['disLabels']]
  # if addExtraBiomk:
  #   params['otherBiomkPerDisease'] = [[nrBiomk-3,nrBiomk-2, nrBiomk-1],
  #     [nrBiomk-3,nrBiomk-2, nrBiomk-1], [nrBiomk-3,nrBiomk-2, nrBiomk-1]] # can also add 3 extra cognitive tests
  # else:
  #   params['otherBiomkPerDisease'] = [[], [], []]

  params['binMaskSubjForEachDisD'] = [np.in1d(params['diag'],
                                      params['diagsSetInDis'][disNr]) for disNr in range(nrDis)]

  eps = 0.001
  nrXPoints = 50
  params['trueParams'] = {}
  subShiftsS = np.zeros(RID.shape[0])
  trueDysfuncXsX = np.linspace(0, 1, nrXPoints)
  trueTrajFromDysXB = eps * np.ones((nrXPoints, nrBiomk))

  trueLineSpacedDPSsX = np.linspace(-10, 10, nrXPoints)
  trueTrajPredXB = eps * np.ones((nrXPoints,nrBiomk))
  trueDysTrajFromDpsXU = eps * np.ones((nrXPoints,nrBiomkDisModel))

  scalingBiomk2B = np.zeros((2, nrBiomk))
  scalingBiomk2B[1,:] = 1

  trueParamsFuncUnits = [0 for _ in range(nrFuncUnits)]
  for f in range(nrFuncUnits):
    trueParamsFuncUnits[f] = dict(xsX=trueDysfuncXsX, ysXB=trueTrajFromDysXB[:, biomkInFuncUnit[f]],
                                  subShiftsS=subShiftsS,
                                  scalingBiomk2B=scalingBiomk2B[:, biomkInFuncUnit[f]])

  # disease specific
  trueParamsDis = [0 for _ in range(nrDis)]
  for d in range(nrDis):
    trueParamsDis[d] = dict(xsX=trueLineSpacedDPSsX, ysXU=trueDysTrajFromDpsXU, ysXB=trueTrajPredXB,
    subShiftsS=np.zeros(np.sum(np.in1d(params['diag'],params['diagsSetInDis'][d]))),
    scalingBiomk2B=scalingBiomk2B)


  # for DKT DPMs
  params['trueParamsFuncUnits'] = trueParamsFuncUnits
  params['trueParamsDis'] = trueParamsDis

  # simpler non-DKT DPMs
  params['trueParams'] = dict(xsX=trueLineSpacedDPSsX, ysXU = trueTrajPredXB, ysXB = trueTrajPredXB,
    subShiftsS=subShiftsS, scalingBiomk2B=scalingBiomk2B)
  params['plotTrajParams']['trueParams'] = params['trueParams']

  assert params['diag'].shape[0] == len(params['XtrainAll'][0][0])


  # params['runPartStd'] = ['L', 'L']
  params['runPartStd'] = args.runPartStd
  params['runPartMain'] = ['R', 'I', 'I'] # [mainPart, plot, stage]
  params['masterProcess'] = args.runIndex == 0


  modelNames, res = evaluationFramework.runModels(params, expName,
   args.modelToRun, runAllExpTadpoleSubtypes)
  pickle.dump(dict(modelNames=modelNames, res=res, plotTrajParams=plotTrajParams, params=params),
              open('tempRes.npz', 'wb'), protocol=pickle.HIGHEST_PROTOCOL)

  ds = pickle.load(open('tempRes.npz', 'rb'))
  modelNames = ds['modelNames']
  res = ds['res']
  plotTrajParams = ds['plotTrajParams']
  params = ds['params']

  if params['masterProcess']:
    printRes(modelNames, res, plotTrajParams, params)


def printRes(modelNames, res, plotTrajParams, params):
  #nrModels = len(modelNames)

  nrDis = params['nrDis']
  modelNames += ['Lin', 'Spline', 'Multivar']
  officialNames = {'JMD': 'DKT', 'Sig': 'Latent stage', 'Lin': 'Linear',
                   'Spline': 'Spline', 'Multivar': 'Multivariate'}

  for d in [0]: # range(nrDis):

    disNrsValid = [d2 for d2 in range(nrDis) if d2 not in [d]]
    biomkNames = res[0]['metrics'][d][disNrsValid[0]]['labelsNonMri']
    nrBiomk = len(biomkNames)
    resDf = pd.DataFrame(index=range(12 * len(disNrsValid)), columns=['Model'] + biomkNames)

    c = 0

    # dpmObjStd[s].plotter.plotAllBiomkDisSpace(dpmObjStd[s], params, disNr=0)
    for disNrValid in disNrsValid:
      print('%d-%d training on dis %s   validation on disease %s' % (d, disNrValid,
        params['disLabels'][d], params['disLabels'][disNrValid]))



      dktIndex = 0
      sigIndex = 1
      linIndex = 2
      splineIndex = 3
      multivarIndex = 4

      # print('##### biomk prediction ######')
      nrModels = len(officialNames)
      mseMUB = list(range(nrModels))
      mseMeanMU = list(range(nrModels))
      mseStdMU = list(range(nrModels))

      corrMUB = list(range(nrModels))
      corrMeanMU = list(range(nrModels))
      corrStdMU = list(range(nrModels))
      pvalsMU = list(range(nrModels))

      for m in range(len(res)):
        # print(res[m]['metrics'])
        # print(res[m]['metrics'][d])
        mseMUB[m] = res[m]['metrics'][d][disNrValid]['dpm']['mseUB']
        mseMeanMU[m] = np.nanmean(mseMUB[m], axis=1)
        mseStdMU[m] = np.nanstd(mseMUB[m], axis=1)

        corrMUB[m] = res[m]['metrics'][d][disNrValid]['dpm']['corrUB']
        corrMeanMU[m] = np.nanmean(corrMUB[m], axis=1)
        corrStdMU[m] = np.nanstd(corrMUB[m], axis=1)

      mseMUB[linIndex] = res[0]['metrics'][d][disNrValid]['lin']['mseUB']
      mseMeanMU[linIndex] = np.nanmean(mseMUB[linIndex], axis=1)
      mseStdMU[linIndex] = np.nanstd(mseMUB[linIndex], axis=1)

      mseMUB[splineIndex] = res[0]['metrics'][d][disNrValid]['spline']['mseUB']
      mseMeanMU[splineIndex] = np.nanmean(mseMUB[splineIndex], axis=1)
      mseStdMU[splineIndex] = np.nanstd(mseMUB[splineIndex], axis=1)

      mseMUB[multivarIndex] = res[0]['metrics'][d][disNrValid]['multivar']['mseUB']
      mseMeanMU[multivarIndex] = np.nanmean(mseMUB[multivarIndex], axis=1)
      mseStdMU[multivarIndex] = np.nanstd(mseMUB[multivarIndex], axis=1)


      corrMUB[linIndex] = res[0]['metrics'][d][disNrValid]['lin']['corrUB']
      corrMeanMU[linIndex] = np.nanmean(corrMUB[linIndex], axis=1)
      corrStdMU[linIndex] = np.nanstd(corrMUB[linIndex], axis=1)

      corrMUB[splineIndex] = res[0]['metrics'][d][disNrValid]['spline']['corrUB']
      corrMeanMU[splineIndex] = np.nanmean(corrMUB[splineIndex], axis=1)
      corrStdMU[splineIndex] = np.nanstd(corrMUB[splineIndex], axis=1)

      corrMUB[multivarIndex] = res[0]['metrics'][d][disNrValid]['multivar']['corrUB']
      corrMeanMU[multivarIndex] = np.nanmean(corrMUB[multivarIndex], axis=1)
      corrStdMU[multivarIndex] = np.nanstd(corrMUB[multivarIndex], axis=1)

      # Perform Bonferroni correction
      sigLevel = 0.05/(6*2*nrModels)

      # print('mseMUB[dktIndex][u, :]', np.nanmean(mseMUB[dktIndex][1, :]), mseMUB[dktIndex][1, :])
      # print('mseMUB[linIndex][u, :]', np.nanmean(mseMUB[linIndex][1, :]), mseMUB[linIndex][1, :])
      # pl.figure(1)
      # pl.hist(mseMUB[dktIndex][1, :], color='g', label='dkt')
      # pl.hist(mseMUB[linIndex][1, :], color='r', label='lin')
      # pl.show()

      print('##### mean squared error and rank correlation ######')
      # print(r'''\textbf{Model} & ''' + ' & '.join(['\\textbf{%s}' % b for b in biomkNames]) + '\\\\')
      # print('& \multicolumn{6}{c}{\\textbf{Prediction Error (MSE)}}\\\\')
      resDf.iloc[c, 0] = 'Prediction Error (MSE)'
      c += 1
      modelIndxs = [dktIndex, sigIndex, multivarIndex, splineIndex, linIndex]

      for m in modelIndxs:
        # print('%s' % officialNames[modelNames[m]], end='')
        print('modelNames[m]', modelNames[m])
        print('officialNames[modelNames[m]]',officialNames[modelNames[m]])
        resDf.iloc[c,0] = officialNames[modelNames[m]]
        # c += 1

        for u in range(mseMeanMU[m].shape[0]):
          # print('resDf.shape', resDf.shape)
          # print('mseMeanMU[m].shape[0]', mseMeanMU[m].shape[0])
          # print('resDf', resDf)
          sigLabel = getSigLabel(mseMUB[m][u, :], mseMUB[dktIndex][u, :], sigLevel)
          # print(' & %.2f$\pm$%.2f%s' % (mseMeanMU[m][u], mseStdMU[m][u], sigLabel), end='')
          resDf.iloc[c, u+1] = '%.2f +/- %.2f%s' % (mseMeanMU[m][u], mseStdMU[m][u], sigLabel)

        c += 1

        # print('\\\\')

      # print('& \multicolumn{6}{c}{\\textbf{Rank Correlation (Spearman rho)}}\\\\')
      resDf.iloc[c, 0] = 'Rank Correlation (Spearman rho)'
      c += 1

      for m in modelIndxs:
        # print('%s ' % officialNames[modelNames[m]], end='')
        resDf.iloc[c, 0] = officialNames[modelNames[m]]
        # c += 1

        for u in range(mseMeanMU[m].shape[0]):
          sigLabel = getSigLabel(corrMUB[m][u,:], corrMUB[dktIndex][u,:], sigLevel)
          # print(' & %.2f%s ' % (corrMeanMU[m][u], sigLabel), end='')
          resDf.iloc[c, u+1] = '%.2f +/- %.2f%s' % (corrMeanMU[m][u], corrStdMU[m][u], sigLabel)

        c += 1


        # print('\\\\' )

    print(resDf)
    resDf.to_html('valid_train%d.html' % d)
    resDf.loc[:, 'Model' : 'DTI FA Temporal'].to_latex('valid_train%d.tex' % d, index=False)



def getSigLabel(xs, xsMyModel, sigLevel):
  tstatCorrDkt, pValCorrDkt = scipy.stats.ttest_rel(xs, xsMyModel)

  if pValCorrDkt < sigLevel:
    sigLabel = '*'
  else:
    sigLabel = ''

  return sigLabel

def runAllExpTadpoleSubtypes(params, expName, dpmBuilder, compareTrueParamsFunc = None):
  """ runs all experiments"""

  res = {}

  params['patientID'] = AD
  params['excludeID'] = -1
  params['excludeXvalidID'] = -1
  params['excludeStaging'] = [-1]

  params['outFolder'] = 'resfiles/%s' % expName

  nrDis = params['nrDis']
  dpmObjStd = [0 for d in range(nrDis)]
  res['std'] = [0 for d in range(nrDis)]
  res['metrics'] = [{} for d in range(nrDis)]

  # res['metrics'][1][2] = 3
  # print(res['metrics'])
  # print(das)

  for d in [0]: # range(nrDis):
    params['expName'] = params['expNamesSubgr'][d]
    params['X'] = params['XtrainAll'][d]
    params['Y'] = params['YtrainAll'][d]
    params['RID'] = params['RIDtrainAll'][d]
    params['diag'] = params['diagTrainAll'][d]
    params['visitIndices'] = params['visitIndicesTrainAll'][d]

    # print('X', params['X'])
    # print('Y', params['Y'])
    # print(adsas)

    dpmObjStd[d], res['std'][d] = evaluationFramework.runStdDPM(params,
      expName, dpmBuilder, params['runPartMain'])
    disNrsValid = [d2 for d2 in range(nrDis) if d2 not in [d]]

    # dpmObjStd[s].plotter.plotAllBiomkDisSpace(dpmObjStd[s], params, disNr=0)
    for disNrValid in disNrsValid:
      # perform the validation against DRC data
      res['metrics'][d][disNrValid] = validateSubtypes(dpmObjStd[d], params, d, disNrValid)


  return res


def validateSubtypes(dpmObj, params, disNrTrain, disNrValid):



  # the dpmObj contains all subjects with all diseases
  # select only subjects that are used for validation, i.e. the 'rare diseases' with limited data
  diag = params['diag']
  indxSubjToKeep = dpmObj.getIndxSubjToKeep(disNrValid)

  import DPMModelGeneric
  Xfilt, Yfilt = DPMModelGeneric.DPMModelGeneric.filterXYsubjInd(params['X'], params['Y'], indxSubjToKeep)

  diagSubjCurrDis = diag[indxSubjToKeep]

  #### construct sub-shifts for each biomarker
  XshiftedDisModelBS, ysPredBS, xsOrigPred1S = dpmObj.getDataDisOverBiomk(disNrValid)


  for b in range(dpmObj.nrBiomk):
    assert len(params['X'][b]) == len(params['Y'][b])
    assert len(XshiftedDisModelBS[b]) == len(Yfilt[b])

  # now get the validation set. This is already only for the DRC subjects
  Xvalid = params['Xvalid']
  Yvalid = params['Yvalid']
  RIDvalid = params['RIDvalid']
  diagValid = params['diagValid']

  labels = params['labels']
  nonMriBiomksList = [i for i in range(6,len(labels)-params['nrExtraBiomk'])]
  mriBiomksList = [i for i in range(len(labels)) if labels[i].startswith('Volume')]

  print('labels', labels)
  print('nonMriBiomksList', nonMriBiomksList)
  # print(adsa)

  assert len(ysPredBS) == len(Yvalid)

  nrNonMriCols = len(nonMriBiomksList)
  mse = [0 for b in nonMriBiomksList]

  # subjects who have non-MRI validation data
  subjWithValidMask = np.zeros(len(Yvalid[0]), bool)
  for s in range(len(Yvalid[0])):
    subjWithValidMask[s] = np.any([Yvalid[b][s].shape[0] > 0 for b in nonMriBiomksList])
  # print('dpmObj.disIdxForEachSubjS.shape', dpmObj.disIdxForEachSubjS.shape, dpmObj.disIdxForEachSubjS)
  # print('subjWithValidMask', subjWithValidMask.shape, subjWithValidMask)

  subjWithValidIndx = np.where(np.logical_and(subjWithValidMask,
                                              dpmObj.disIdxForEachSubjS == disNrValid))[0]
  nrSubjWithValid = subjWithValidIndx.shape[0]
  XvalidFilt, YvalidFilt = DPMModelGeneric.DPMModelGeneric.filterXYsubjInd(Xvalid, Yvalid, subjWithValidIndx)
  diagValidFilt = diagValid[subjWithValidIndx]
  RIDvalidFilt = RIDvalid[subjWithValidIndx]
  ridCurrDis = params['RID'][indxSubjToKeep]

  # print('subjWithValidIndx', subjWithValidIndx)
  # print('XvalidFilt', XvalidFilt)
  # print(adsass)

  XvalidShifFilt = [[np.array([]) for s in range(nrSubjWithValid)] for b in range(dpmObj.nrBiomk)]

  ###### construct the shifts of the subjects in validation set #############
  for b in range(nrNonMriCols):
    mseList = []
    for s in range(RIDvalidFilt.shape[0]):
      # for each validation subject
      # print('disNrValid', disNrValid)
      # print('RIDvalid', RIDvalid)
      # print('params[RID]', params['RID'])
      # print(RIDvalidFilt)
      # print(RIDvalidFilt[s])
      # print(ridCurrDis)
      idxCurrDis = np.where(RIDvalidFilt[s] == ridCurrDis)[0][0]
      xsOrigFromModel = xsOrigPred1S[idxCurrDis]

      if XvalidFilt[nonMriBiomksList[b]][s].shape[0] > 0:
        # print('xsOrigFromModel', xsOrigFromModel)
        # print('XvalidFilt[nonMriBiomksList[b]][s]', XvalidFilt[nonMriBiomksList[b]][s])
        # print('XvalidFilt[nonMriBiomksList[b]][s][0]', XvalidFilt[nonMriBiomksList[b]][s][0])
        assert np.where(xsOrigFromModel == XvalidFilt[nonMriBiomksList[b]][s][0])[0].shape[0] == 1
        idxXsWithValid = np.where(np.in1d(xsOrigFromModel, XvalidFilt[nonMriBiomksList[b]][s]))[0]
        ysPredCurrSubj = ysPredBS[nonMriBiomksList[b]][idxCurrDis][idxXsWithValid]

        assert YvalidFilt[nonMriBiomksList[b]][s].shape[0] > 0

        mseList += list((ysPredCurrSubj - YvalidFilt[nonMriBiomksList[b]][s][0]) ** 2)

        # also compose the shifted Xs for the validation subjects
        xsShiftedFromModel = XshiftedDisModelBS[0][idxCurrDis]
        XvalidShifFilt[nonMriBiomksList[b]][s] = np.array(xsShiftedFromModel[idxXsWithValid])

        # print('xsOrigFromModel', xsOrigFromModel)
        # print('XvalidFilt[nonMriBiomksList[b]][s]', XvalidFilt[nonMriBiomksList[b]][s])
        # print('idxXsWithValid', np.where(np.in1d(xsOrigFromModel, XvalidFilt[nonMriBiomksList[b]][s]))[0])
        # print('XshiftedDisModelBS[0][idxCurrDis]', XshiftedDisModelBS[0][idxCurrDis])
        # print('xsShiftedFromModel[idxXsWithValid]', xsShiftedFromModel[idxXsWithValid])
        # print(XvalidShifFilt[nonMriBiomksList[b]][s])
        # print(YvalidFilt[nonMriBiomksList[b]][s])
        assert XvalidShifFilt[nonMriBiomksList[b]][s].shape[0] == YvalidFilt[nonMriBiomksList[b]][s].shape[0]

      else:
        XvalidShifFilt[nonMriBiomksList[b]][s] = np.array([])

    # print('mseList', mseList)
    if len(mseList) > 0:
      mse[b] = np.mean(mseList)
    else:
      mse[b] = np.nan


  # print('mse', mse)
  # print('mseList', mseList)
  # print(adsas)

  # part 2. plot the inferred dynamics for validation data:
  # every biomarker against original DPS
  # also plot extra validation data on top
  xsTrajX = dpmObj.getXsMinMaxRange(disNrValid)
  predTrajXB = dpmObj.predictBiomkSubjGivenXs(xsTrajX, disNrValid)
  trajSamplesBXS = dpmObj.sampleBiomkTrajGivenXs(xsTrajX, disNrValid, nrSamples=100)

  ### part 3. build a simpler linear predictor from MR to Non-MRI for every ROI independently.
  # Train it on ADNI MR+Non-MRI data and use it to predict DRC-NonMri from DRC-MR.

  dataDfTadpole = params['dataDfTrainAll'][disNrTrain]
  colsList = dataDfTadpole.columns.tolist()
  # print('colsList', colsList)
  mriBiomksColsInd = [i for i in range(len(colsList)) if colsList[i].startswith('Volume')]
  nonMriBiomksColsInd = [i for i in range(len(colsList)) if (colsList[i].split(' ')[0] in ['DTI', 'FDG', 'AV45', 'AV1451'])]

  nrMriBiomks = len(mriBiomksColsInd)
  # print(dataDfTadpole.loc[:, 'DTI FA Cingulate' : colsList[-1]])
  # nonMriBiomksDf = nonMriBiomksList



  dataDfAllMat = dataDfTadpole.values
  # print('dataDfTadpole', dataDfTadpole)
  # print('dataDfAllMat', dataDfAllMat)
  # print(dassa)

  nrNonMriBiomk = len(nonMriBiomksList)

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

  # select just the non-MRI biomarkers
  nonMriColsArrayIndx = np.array(nonMriBiomksList)
  mriColsArrayIndx = np.array(mriBiomksList)
  # print('nonMriColsArrayIndx', nonMriColsArrayIndx)
  predTrajNonMriXB = predTrajXB[:,nonMriColsArrayIndx]
  predTrajMriXB = predTrajXB[:, mriColsArrayIndx]
  predTrajMriRepXB = np.tile(predTrajMriXB, int(nrNonMriBiomk/nrMriBiomks))
  assert predTrajMriXB.shape[0] == predTrajMriRepXB.shape[0]

  trajSamplesNonMriBXS = trajSamplesBXS[nonMriColsArrayIndx,:,:]
  XvalidShifNonMriFilt = [XvalidShifFilt[b] for b in nonMriBiomksList]
  YvalidFiltNonMri = [YvalidFilt[b] for b in nonMriBiomksList]
  # list of all MRI values corresponding to non-MRI. repeats for all non-MRI modalities (e.g. FDG, DTI, ...)
  YvalidFiltMriClosestToNonMri = [[[] for s in range(nrSubjWithValid) ] for b in nonMriBiomksList] # only the MRI where Non-MRI exists

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

  # print('dataDfAllMat', dataDfAllMat)
  # print(dassa)

  # for each biomarkers in the validation set (non-MRI) find the corresponding MRI biomarkers
  for b in range(nrNonMriBiomk):
    f = b % nrMriBiomks

    mriDataCurrCol = dataDfAllMat[:, mriBiomksColsInd[f]]
    nonMriDataCurrCol = dataDfAllMat[:, nonMriBiomksColsInd[b]]

    nnMask = ~np.isnan(mriDataCurrCol) & ~np.isnan(nonMriDataCurrCol)
    linModel = sklearn.linear_model.LinearRegression(fit_intercept=True)
    linModel.fit(mriDataCurrCol[nnMask].reshape(-1,1),
      nonMriDataCurrCol[nnMask].reshape(-1,1))
    # print('print(np.sum(nnMask))', np.sum(nnMask))

    from scipy.interpolate import Rbf, UnivariateSpline
    sortedXInd = np.argsort(mriDataCurrCol[nnMask])
    splineModel = UnivariateSpline(mriDataCurrCol[nnMask][sortedXInd],
                                               nonMriDataCurrCol[nnMask][sortedXInd], k=3)

    # Now do multivariate prediction. Based on all MRI values at each lobe, predict one DTI biomk.
    nnMaskMulti = (np.sum(np.isnan(dataDfAllMat[:, mriBiomksColsInd]),axis=1) == 0) & ~np.isnan(nonMriDataCurrCol)
    # print('print(np.sum(nnMaskMulti))', np.sum(nnMaskMulti))
    assert np.isfinite(dataDfAllMat[nnMaskMulti, :][:, mriBiomksColsInd]).all()
    assert np.isfinite(nonMriDataCurrCol[nnMaskMulti]).all()
    nnDataMri = dataDfAllMat[nnMaskMulti, :][:, mriBiomksColsInd]
    # print(nnDataMri[:,0].shape[0])
    # print(nonMriDataCurrCol[nnMaskMulti].shape[0])
    assert nnDataMri[:,0].shape[0] == nnDataMri[:, 1].shape[0]
    assert nnDataMri[:,0].shape[0] == nonMriDataCurrCol[nnMaskMulti].shape[0]

    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import RBF, WhiteKernel

    multivarModelRbf = Rbf(nnDataMri[:,0], nnDataMri[:,1], nnDataMri[:,2], nnDataMri[:,3], nnDataMri[:,4],
               nnDataMri[:,5], nonMriDataCurrCol[nnMaskMulti], function='cubic')

    kernel = 1.0 * RBF(length_scale=1, length_scale_bounds=(0.1, 10)) \
             + WhiteKernel(noise_level=0.3, noise_level_bounds=(0.1, 0.1))
    # print('nnDataMri[:,0:5].shape', nnDataMri[:,0:6].shape)
    # print(nonMriDataCurrCol[nnMaskMulti].shape)
    multivarModel = GaussianProcessRegressor(kernel=kernel,
                                  alpha=0.0).fit(nnDataMri[:,0:6], nonMriDataCurrCol[nnMaskMulti])

    YvalidDktNonMri[b] = [[] for s in range(nrSubjWithValid)]  # Non-MRI predictions of DKT model for subj in validation set
    YvalidLinModelNonMri[b] = [[] for s in range(nrSubjWithValid) ] # Non-MRI predictions of linear model for subj in validation set
    YvalidSplineModelNonMri[b] = [[] for s in range(nrSubjWithValid)]  # Non-MRI predictions of linear model for subj in validation set
    YvalidMultivarModelNonMri[b] = [[] for s in range(nrSubjWithValid)]  # Non-MRI predictions of linear model for subj in validation set

    for s in range(nrSubjWithValid):
      if YvalidFilt[nonMriBiomksList[b]][s].shape[0] > 0:
        for v in range(YvalidFilt[nonMriBiomksList[b]][s].shape[0]):
          mrValsValidCurrSubj = np.array(YvalidFilt[mriBiomksList[f]][s]).reshape(-1,1)
          nonMriValValidCurrSubj = YvalidFilt[nonMriBiomksList[b]][s][v]

          xMriCurr = np.array(XvalidFilt[mriBiomksList[f]][s])
          xNonMriCurr = XvalidFilt[nonMriBiomksList[b]][s][v]

          closestMriIdx = np.argmin(np.abs(xMriCurr - xNonMriCurr))

          YvalidFiltMriClosestToNonMri[b][s] += [mrValsValidCurrSubj[closestMriIdx]]
          nonMriPredValidLin = linModel.predict(mrValsValidCurrSubj[closestMriIdx].reshape(-1,1))
          nonMriPredValidLin = nonMriPredValidLin[0][0]
          nonMriPredValidSpline = splineModel(mrValsValidCurrSubj[closestMriIdx])
          mrValsToPredictCurrSubj = [YvalidFilt[mriBiomksList[u]][s][closestMriIdx]
                                     for u in range(nrMriBiomks)]
          # print('mrValsToPredictCurrSubj', mrValsToPredictCurrSubj)
          assert len(mrValsToPredictCurrSubj) == nrMriBiomks
          nonMriPredValidMultivarRbf = multivarModelRbf(mrValsToPredictCurrSubj[0],mrValsToPredictCurrSubj[1],
            mrValsToPredictCurrSubj[2], mrValsToPredictCurrSubj[3], mrValsToPredictCurrSubj[4],
            mrValsToPredictCurrSubj[5])

          # print(np.array(mrValsToPredictCurrSubj).reshape(1, -1).shape)
          nonMriPredValidMultivar = multivarModel.predict(np.array(mrValsToPredictCurrSubj).reshape(1, -1))

          # print('nonMriPredValidMultivarRbf', nonMriPredValidMultivarRbf)
          # print('nonMriPredValidMultivar', nonMriPredValidMultivar)
          # print(dadsa)


          YvalidLinModelNonMri[b][s] += [nonMriPredValidLin]
          YvalidSplineModelNonMri[b][s] += [nonMriPredValidSpline]
          YvalidMultivarModelNonMri[b][s] += [nonMriPredValidMultivar]

          indOfXTrajClosestToCurrSubj = np.argmin(np.abs(XvalidShifNonMriFilt[b][s][v] - xsTrajX))
          nonMriPredValidDkt = predTrajNonMriXB[indOfXTrajClosestToCurrSubj, b]
          YvalidDktNonMri[b][s] += [nonMriPredValidDkt]

          if diagValidFilt[s] > CTL3: # don't include CTLs for validation
            squaredErrorsDpm[b]  += [(nonMriValValidCurrSubj - nonMriPredValidDkt) ** 2]
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

      else:
        YvalidFiltMriClosestToNonMri[b][s] = np.array([])
        YvalidDktNonMri[b][s] = np.array([])
        YvalidLinModelNonMri[b][s] = np.array([])
        YvalidSplineModelNonMri[b][s] = np.array([])
        YvalidMultivarModelNonMri[b][s] = np.array([])

    nonMriValValidAll[b] = np.array(nonMriValValidAll[b]).reshape(-1, 1).astype(float)
    nonMriPredValidDktAll[b] = np.array(nonMriPredValidDktAll[b]).reshape(-1, 1).astype(float)
    nonMriPredValidLinAll[b] = np.array(nonMriPredValidLinAll[b]).reshape(-1, 1).astype(float)
    nonMriPredValidSplineAll[b] = np.array(nonMriPredValidSplineAll[b]).reshape(-1, 1).astype(float)
    nonMriPredValidMultivarAll[b] = np.array(nonMriPredValidMultivarAll[b]).reshape(-1, 1).astype(float)


    corrDpm[b], pValDpm[b] = scipy.stats.spearmanr(nonMriValValidAll[b],
      nonMriPredValidDktAll[b])
    corrLin[b], pValLin[b] = scipy.stats.spearmanr(nonMriValValidAll[b],
      nonMriPredValidLinAll[b])
    corrSpline[b], pValSpline[b] = scipy.stats.spearmanr(nonMriValValidAll[b],
      nonMriPredValidSplineAll[b])
    corrMultivar[b], pValMultivar[b] = scipy.stats.spearmanr(nonMriValValidAll[b],
      nonMriPredValidMultivarAll[b])

    # print('nonMriValValidAll[b]',nonMriValValidAll[b])
    # print('nonMriPredValidSplineAll[b]',nonMriPredValidSplineAll[b])
    # print(adsa)


  for b in range(nrNonMriBiomk):
    for s in range(nrSubjWithValid):
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
    for b2 in range(nrBootStraps):
      idxBootCurr = np.array(np.random.choice(nrSubjWithValidAndChosen,nrSubjWithValidAndChosen), int)
      # print(len(squaredErrorsLin[b]))
      # print('idxBootCurr', idxBootCurr)
      mseDpmUB[b, b2] = np.mean(squaredErrorsDpm[b][idxBootCurr])
      mseLinUB[b, b2] = np.mean(squaredErrorsLin[b][idxBootCurr])
      mseSplineUB[b, b2] = np.mean(squaredErrorsSpline[b][idxBootCurr])
      mseMultivarUB[b, b2] = np.mean(squaredErrorsMultivar[b][idxBootCurr])

      idxBootCorrCurr = np.array(np.random.choice(nonMriValValidAll[b].shape[0], nonMriValValidAll[b].shape[0]), int)

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
  metrics['dpm']['mseUB'] = mseDpmUB
  metrics['lin'] = {}
  metrics['lin']['corrUB'] = corrLinUB
  metrics['lin']['mseUB'] = mseLinUB
  metrics['spline'] = {}
  metrics['spline']['corrUB'] = corrSplineUB
  metrics['spline']['mseUB'] = mseSplineUB
  metrics['multivar'] = {}
  metrics['multivar']['corrUB'] = corrMultivarUB
  metrics['multivar']['mseUB'] = mseMultivarUB

  # plot against MRI vals instead of DPS time-shifts

  # also plot training data non-MRI[b] in MRI[b] space
  YNonMri = [params['Y'][b] for b in nonMriBiomksList]
  YMriClosestToNonMri = [[0 for s in range(len(YNonMri[b]))] for b in range(len(nonMriBiomksList))]  # only the MRI where Non-MRI exists
  # idxWithNonMri = [s for s in range(len(YNonMri)) ]
  # print('YNonMri', YNonMri)
  # print(adsa)

  for b in range(nrNonMriBiomk):
    f = b % nrMriBiomks # for each biomarker in the validation set find the MRI biomarker
    for s in range(len(YNonMri[b])):

      YMriClosestToNonMri[b][s] = np.array([])

      if YNonMri[b][s].shape[0] > 0:

        xsMriCurrSubj = params['X'][mriBiomksList[f]][s]
        xsnonMriCurrSubj = params['X'][nonMriBiomksList[b]][s]

        mriValsCorrespToNonMriCurrSubj = []
        for t in range(xsnonMriCurrSubj.shape[0]):
          mriIndClosestToCurrNonMriScan = np.argmin(np.abs(xsnonMriCurrSubj[t] - xsMriCurrSubj))

          mriValsCorrespToNonMriCurrSubj += [params['Y'][mriBiomksList[f]][s][mriIndClosestToCurrNonMriScan]]

          YMriClosestToNonMri[b][s] = np.array(mriValsCorrespToNonMriCurrSubj)


      # print(YMriClosestToNonMri[b][s].shape[0])
      # print(YNonMri[b][s].shape[0])
      assert YMriClosestToNonMri[b][s].shape[0] == YNonMri[b][s].shape[0]

  # print(ads)

  labelsNonMri = [params['labels'][b] for b in nonMriBiomksList]
  metrics['labelsNonMri'] = labelsNonMri
  # change diagnosis numbers to get different plotting behaviour (specific labels, colors and markers)
  diagValidFiltLinModel = copy.deepcopy(diagValidFilt)
  diagValidFiltLinModel[diagValidFiltLinModel == CTL1] = CTL1_LIN
  diagValidFiltLinModel[diagValidFiltLinModel == CTL2] = CTL2_LIN
  diagValidFiltLinModel[diagValidFiltLinModel == CTL3] = CTL3_LIN
  diagValidFiltLinModel[diagValidFiltLinModel == SBG1] = SBG1_LIN
  diagValidFiltLinModel[diagValidFiltLinModel == SBG2] = SBG2_LIN
  diagValidFiltLinModel[diagValidFiltLinModel == SBG3] = SBG3_LIN

  diagValidFiltDktModel = copy.deepcopy(diagValidFilt)
  diagValidFiltDktModel[diagValidFiltDktModel == CTL1] = CTL1_DKT
  diagValidFiltDktModel[diagValidFiltDktModel == CTL2] = CTL2_DKT
  diagValidFiltDktModel[diagValidFiltDktModel == CTL3] = CTL3_DKT
  diagValidFiltDktModel[diagValidFiltDktModel == SBG1] = SBG1_DKT
  diagValidFiltDktModel[diagValidFiltDktModel == SBG2] = SBG2_DKT
  diagValidFiltDktModel[diagValidFiltDktModel == SBG3] = SBG3_DKT


  # plot just the trajectories by modality groups
  # for d in range(dpmObj.nrDis):
  #   fig = dpmObj.plotter.plotTrajInDisSpaceOverlap(dpmObj, d, params, replaceFig=True)
  #   fig.savefig('%s/trajDisSpaceOverlap_%s_%s.png' % (params['outFolder'],
  #     params['disLabels'][d], params['expName']))

  plotFigs = False
  if plotFigs:

    # for u in range(dpmObj.nrFuncUnits):
    #   trajStructUnitModel = dpmObj.unitModels[u].plotter.getTrajStructWithTrueParams(dpmObj.unitModels[u])
    #   fig = dpmObj.unitModels[u].plotter.plotTraj(dpmObj.unitModels[u], trajStructUnitModel,
    #     legendExtraPlot=True, rowsAuto=True)
    #   fig.savefig('%s/unit%d_allTraj.png' % (params['outFolder'], u))


    # for d in range(dpmObj.nrDis):
    #   yNormMode = dpmObj.params['plotTrajParams']['yNormMode'] # for TADPOLE subtypes
    #   # yNormMode = 'unscaled' # for non-TADPOLE subtypes
    #   trajStructDisModel = dpmObj.disModels[d].plotter.getTrajStructWithTrueParams(dpmObj.disModels[d], yNormMode)
    #   fig = dpmObj.disModels[d].plotter.plotAllTrajZeroOne(dpmObj.disModels[d], trajStructDisModel)
    #   fig.savefig('%s/dis%d_%s_allTrajZeroOne.png' % (params['outFolder'], d, dpmObj.params['disLabels'][d]))


    mseDpm = np.nanmean(mseDpmUB, axis=1)
    mseLin = np.nanmean(mseLinUB, axis=1)
    print('mseDpmUB', mseDpmUB)
    print('mseLinUB', mseLinUB)
    print('mseDpm', mseDpm)
    print('mseLin', mseLin)
    # print(adsas)

    # print('YvalidFiltMriClosestToNonMri', YvalidFiltMriClosestToNonMri)
    # print('YvalidFiltNonMri', YvalidFiltNonMri)
    # print('YvalidLinModelNonMri', YvalidLinModelNonMri)
    # print(adsas)
    # plot Non-MRI over MRI space: traj, validation data, predictions of linear model, training data.
    # print(params['plotTrajParams']['diagLabels'])
    # print(das)
    # dpmObj.plotter.plotTrajParams = params['plotTrajParams']
    # dpmObj.unitModels[0].plotter

    # XshiftedDisModelBS, _, _ = dpmObj.getDataDisOverBiomk(disNrValid)
    # diagTrainCurrDis = params['diag'][dpmObj.disIdxForEachSubjS == disNrValid]
    # indxSubjToKeep = dpmObj.getIndxSubjToKeep(disNrValid)
    # _, YvalidDis = DPMModelGeneric.DPMModelGeneric.filterXYsubjInd(params['X'], params['Y'], indxSubjToKeep)

    import JointModel
    if isinstance(dpmObj, JointModel.JointModel):
      # add empty lists for predictions of linear model for MRI biomarkers, as we only predict non-MRI
      emptyLists = [[np.array([]) for s in range(len(XvalidShifFilt[0]))] for b in range(nrMriBiomks)]
      emptyListsCogBiomk = [[np.array([]) for s in range(len(XvalidShifFilt[0]))] for b in range(params['nrExtraBiomk'])]
      YvalidLinModel = emptyLists + YvalidLinModelNonMri + emptyListsCogBiomk
      # print('YvalidLinModel', YvalidLinModel)
      print(len(XvalidShifFilt))
      print(len(XvalidShifFilt))
      fig = plotAllBiomkDisSpaceAndValidData(dpmObj, params, disNrValid, XvalidShifFilt, YvalidFilt,
       diagValidFilt, XvalidShifFilt, YvalidLinModel, diagValidFiltLinModel)
      fig.savefig('%s/validDisSpaceDisNr%d.png' % (params['outFolder'], disNrValid))

    # print('YMriClosestToNonMri', YMriClosestToNonMri)
    # print('XvalidShifNonMriFilt', XvalidShifNonMriFilt)
    # print(dasa)

    # gpPlotter = Plotter.PlotterDis(dpmObj.plotter.plotTrajParams)
    # fig = gpPlotter.plotTrajInDisSpace(xsTrajX=xsTrajX, predTrajXB=predTrajNonMriXB, trajSamplesBXS=trajSamplesNonMriBXS,
    #   XsubjData1BSX=XvalidShifNonMriFilt, YsubjData1BSX=YvalidFiltNonMri, diagData1S=diagValidFilt,
    #   XsubjData2BSX=XvalidShifNonMriFilt, YsubjData2BSX=YvalidLinModelNonMri, diagData2S=diagValidFiltLinModel,
    #   XsubjData3BSX=XshiftedDisModelBS, YsubjData3BSX=YvalidDis, diagData3S=diagTrainCurrDis,
    #   labels=labelsNonMri, ssdDKT=mseDpm, ssdNoDKT=mseLin, replaceFig=False)
    # fig.savefig('%s/validNonMriPCANr%d.png' % (params['outFolder'], disNrValid))


    fig = dpmObj.plotter.plotTrajInBiomkSpace(dpmObj=dpmObj,
      xsTrajXB=predTrajMriRepXB, predTrajXB=predTrajNonMriXB, trajSamplesBXS=trajSamplesNonMriBXS,
      XsubjData1BSX=YvalidFiltMriClosestToNonMri, YsubjData1BSX=YvalidFiltNonMri, diagData1S=diagValidFilt,
      XsubjData2BSX=YvalidFiltMriClosestToNonMri, YsubjData2BSX=YvalidLinModelNonMri, diagData2S=diagValidFiltLinModel,
      XsubjData3BSX=YMriClosestToNonMri, YsubjData3BSX=YNonMri, diagData3S=params['diag'],
      labels=labelsNonMri, ssdDKT=mseDpm, ssdNoDKT=mseLin, replaceFig=True)
    fig.savefig('%s/validTrajNonMriOverMriDisNr%d.png' % (params['outFolder'], disNrValid))

    # plot Non-MRI over MRI space: DKT predictions, predictions of linear model, validation data.
    fig = dpmObj.plotter.plotTrajInBiomkSpace(dpmObj=dpmObj,
      xsTrajXB=None, predTrajXB=None, trajSamplesBXS=None,
      XsubjData1BSX=YvalidFiltMriClosestToNonMri, YsubjData1BSX=YvalidFiltNonMri, diagData1S=diagValidFilt,
      XsubjData2BSX=YvalidFiltMriClosestToNonMri, YsubjData2BSX=YvalidLinModelNonMri, diagData2S=diagValidFiltLinModel,
      XsubjData3BSX=YvalidFiltMriClosestToNonMri, YsubjData3BSX=YvalidDktNonMri, diagData3S=diagValidFiltDktModel,
      labels=labelsNonMri, ssdDKT=None, ssdNoDKT=None, replaceFig=True)
    fig.savefig('%s/validPredNonMriOverMriNr%d.png' % (params['outFolder'], disNrValid))

  return metrics


def plotAllBiomkDisSpaceAndValidData(dpmObj, params, disNr, XsubjData1BSX, YsubjData1BSX,
     diagData1S, XsubjData2BSX, YsubjData2BSX, diagData2S):

  diag = params['diag']
  indxSubjToKeep = np.where(dpmObj.binMaskSubjForEachDisD[disNr])[0]

  nrBiomk = len(params['X'])
  print('nrBiomk', nrBiomk)
  Xfilt = [0 for b in range(nrBiomk)]
  Yfilt = [0 for b in range(nrBiomk)]
  for b in range(nrBiomk):
    Xfilt[b] = [params['X'][b][i] for i in indxSubjToKeep]
    Yfilt[b] = [params['Y'][b][i] for i in indxSubjToKeep]

  diagSubjCurrDis = diag[indxSubjToKeep]
  ridCurrDis = params['RID'][indxSubjToKeep]
  nrSubCurrDis = indxSubjToKeep.shape[0]

  XshiftedDisModelBS = [[] for b in range(nrBiomk)]
  XshiftedDisModelUS, XdisModelUS, YdisModelUS, _ = dpmObj.disModels[disNr].getData()

  for s in range(nrSubCurrDis):
    # bTmp = 0  # some biomarker, doesn't matter which one
    # ysCurrSubXB = dpmObj.predictBiomkSubjGivenXs(XshiftedDisModelUS[bTmp][s], disNr)

    for b in range(nrBiomk):

      if Xfilt[b][s].shape[0] > 0:
        # fix problem when a subject has the same xs twice (bad input dataset with same visit twice)
        while np.unique(Xfilt[b][s]).shape[0] < Xfilt[b][s].shape[0]:
          for x in Xfilt[b][s]:
            if np.sum(Xfilt[b][s] == x) > 1:
              idxToRemove = np.where(Xfilt[b][s] == x)[0][0]
              Yfilt[b][s] = np.concatenate((Yfilt[b][s][:idxToRemove], Yfilt[b][s][idxToRemove + 1:]))
              Xfilt[b][s] = np.concatenate((Xfilt[b][s][:idxToRemove], Xfilt[b][s][idxToRemove + 1:]))

              break

        XshiftedDisModelBS[b] += [XshiftedDisModelUS[0][s]]
      else:
        XshiftedDisModelBS[b] += [np.array([])]

      if Yfilt[b][s].shape[0] < XshiftedDisModelBS[b][s].shape[0]:
        allXsOrig = XdisModelUS[0][s]
        xsCurrSubj = Xfilt[b][s]
        maskMissing = np.in1d(allXsOrig, xsCurrSubj)
        XshiftedDisModelBS[b][s] = XshiftedDisModelBS[b][s][maskMissing]
        # print('xsCurrSubj', xsCurrSubj)
        # print('allXsOrig', allXsOrig)
        # print('maskMissing', maskMissing)
        # print('XshiftedDisModelBS[b][s]', XshiftedDisModelBS[b][s])
        # print(das)

  for b in range(nrBiomk):
    assert len(params['X'][b]) == len(params['Y'][b])
    assert len(XshiftedDisModelBS[b]) == len(Yfilt[b])

  print(len(XshiftedDisModelBS))
  print(len(XsubjData1BSX))
  assert len(XshiftedDisModelBS) == len(XsubjData1BSX)

  # part 2. plot the inferred dynamics for DRC data:
  # every biomarker against original DPS
  # also plot extra validation data on top
  xsTrajX = dpmObj.disModels[disNr].getXsMinMaxRange()
  predTrajXB = dpmObj.predictBiomkSubjGivenXs(xsTrajX, disNr)
  trajSamplesBXS = dpmObj.sampleBiomkTrajGivenXs(xsTrajX, disNr, nrSamples=100)

  print('predTrajXB', predTrajXB)
  # print('XshiftedDisModelBS', XshiftedDisModelBS[0][0])
  # print('Yfilt', Yfilt[0][0])
  # print(adsa)

  gpPlotter = Plotter.PlotterDis(params['plotTrajParams'])

  print(len(XsubjData1BSX))
  print(len(XsubjData2BSX))
  print(len(YsubjData2BSX))
  # print(das)
  fig = gpPlotter.plotTrajInDisSpace(xsTrajX, predTrajXB, trajSamplesBXS,
     XsubjData1BSX=XsubjData1BSX, YsubjData1BSX=YsubjData1BSX, diagData1S=diagData1S,
     XsubjData2BSX=XsubjData2BSX, YsubjData2BSX=YsubjData2BSX, diagData2S=diagData2S,
     XsubjData3BSX=XshiftedDisModelBS, YsubjData3BSX=Yfilt, diagData3S=diagSubjCurrDis,
     labels=params['plotTrajParams']['labels'], ssdDKT=None, ssdNoDKT=None,
     replaceFig=True)

  pl.pause(1)

  return fig

if __name__ == '__main__':
  main()


