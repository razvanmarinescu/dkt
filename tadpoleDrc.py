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
import GPModel
import ParHierModel
import Plotter
from auxFunc import *
import evaluationFramework
from matplotlib import pyplot as pl
from env import *

from drcValidFuncs import *
from tadpoleDataLoader import *
from drcDataLoader import *
from tadpoleDrcPrepData import *



#                      DKT     OTHER_MODEL        VALID           TRAINING
#                     circle    triangle      diagonal cross       square
#
# ADNI CTL  green
#      MCI  orange
#      AD   red
# DRC  CTL  yellow
#      PCA  magenta
#      AD   blue

plotTrajParams = {}
plotTrajParams['SubfigTrajWinSize'] = (1600,900)
plotTrajParams['nrRows'] = args.nrRows
plotTrajParams['nrCols'] = args.nrCols
plotTrajParams['diagColors'] = {CTL:'g', MCI:'#FFA500', AD:'r',
  CTL2:'y', PCA:'m', AD2:'b', CTL_OTHER_MODEL:'k', PCA_OTHER_MODEL:'b',
  CTL_DKT:'g', PCA_DKT:'r'}
plotTrajParams['diagScatterMarkers'] = {CTL:'s', MCI:'s', AD:'s',
  CTL2:'s', PCA:'s', AD2:'s', CTL_OTHER_MODEL:'^', PCA_OTHER_MODEL:'^',
  CTL_DKT: 'o', PCA_DKT: 'o'}
plotTrajParams['legendCols'] = 4
plotTrajParams['diagLabels'] = {CTL:'CTL ADNI', MCI:'MCI ADNI', AD:'tAD ADNI',
  CTL2:'CTL LOCAL', PCA:'PCA LOCAL', AD2:'tAD LOCAL', CTL_OTHER_MODEL:'CTL LOCAL - No DKT',
  PCA_OTHER_MODEL:'PCA LOCAL - No DKT', CTL_DKT:'CTL - DTK', PCA_DKT:'PCA - DTK'}

# plotTrajParams['freesurfPath'] = freesurfPath
# plotTrajParams['blenderPath'] = blenderPath
plotTrajParams['isSynth'] = False
plotTrajParams['padTightLayout'] = 0.4

if args.agg:
  plotTrajParams['agg'] = True
else:
  plotTrajParams['agg'] = False

hostName = gethostname()
if hostName == 'razvan-Inspiron-5547':
  height = 350
else: #if hostName == 'razvan-Precision-T1700':
  height = 450


pd.set_option('display.max_rows', 100)
pd.set_option('display.max_columns', 50)
pd.set_option('display.width', 5000)


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


def main():

  # don't turn this on unless I add cognitive markers in the DRC dataset.
  addExtraBiomk = False

  np.random.seed(1)
  random.seed(1)
  pd.set_option('display.max_columns', 50)
  tinyData = args.tinyData

  finalDataFile = 'tadDrc.npz'
  expName = 'tadDrc'

  if args.tinyData:
    finalDataFile = finalDataFile.split('.')[0] + 'Tiny.npz'
    expName = expName.split('.')[0] + 'Tiny'

  if addExtraBiomk:
    finalDataFile = finalDataFile.split('.')[0] + 'Cog.npz'
    expName = expName.split('.')[0] + 'Cog'

  regenerateData = (not os.path.isfile(finalDataFile)) or args.regData
  if regenerateData:
    prepareData(finalDataFile, tinyData, addExtraBiomk)
    # print(dada)



  ds = pickle.load(open(finalDataFile, 'rb'))
  dataDfAll = ds['dataDfAll']
  regParamsICV = ds['regParamsICV']
  regParamsAge = ds['regParamsAge']
  regParamsGender = ds['regParamsGender']
  regParamsDataset = ds['regParamsDataset']
  X = ds['X']
  Y = ds['Y']
  RID = np.array(ds['RID'], int)
  labels = ds['list_biomarkers']
  diag = ds['diag']

  outFolder = 'resfiles/'

  params = {}

  av45InListBiomk = np.array([True for x in ds['list_biomarkers'] if x.startswith('AV1451')]).any()
  if av45InListBiomk:
    nrBiomkInFuncUnits = 5
  else:
    nrBiomkInFuncUnits = 4

  # print('dataDfAll', dataDfAll)



  nrDis = 2 # nr of diseases
  params['nrDis'] = nrDis

  # change the order of the functional units so that the hippocampus and occipital are fitted first
  unitPermutation = [5,3,2,1,4,0]

  nrFuncUnits = 6
  mapBiomkToFuncUnits = np.array((unitPermutation * nrBiomkInFuncUnits))
  nrExtraBiomk = 0

  if addExtraBiomk:
    nrExtraBiomk = 5
    nrFuncUnits += nrExtraBiomk  # add the 3 extra cog markers to a unique functional unit

    mapBiomkToFuncUnits = np.array((unitPermutation * nrBiomkInFuncUnits) + list(range(nrFuncUnits-nrExtraBiomk, nrFuncUnits)))

  # print(mapBiomkToFuncUnits)
  # print(dasdas)

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
  #   extraBiomkNames = ['ADAS13', 'CDRSB', 'RAVLT', 'MMSE', 'FAQ']
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

  # plotTrajParams['yNormMode'] = 'zScoreTraj'
  # plotTrajParams['yNormMode'] = 'zScoreEarlyStageTraj'
  plotTrajParams['yNormMode'] = 'unscaled'

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

  params['X'] = X
  params['Y'] = Y
  params['RID'] = RID
  # print('RID', RID)
  # print(ads)
  params['diag'] = diag
  params['plotTrajParams']['diag'] = params['diag']
  params['Xvalid'] = ds['Xvalid']
  params['Yvalid'] = ds['Yvalid']
  params['RIDvalid'] = ds['RIDvalid']
  params['diagValid'] = ds['diagValid']
  params['dataDfAll'] = dataDfAll
  params['visitIndices'] = ds['visitIndices']
  params['visitIndicesValid'] = ds['visitIndicesValid']

  # params['nrGlobIterUnit'] = 10 # these parameters are specific for the Joint Model of Disease (JMD)
  # params['iterParamsUnit'] = 60
  # params['nrGlobIterDis'] = 10
  # params['iterParamsDis'] = 60

  # by default we have no priors
  params['priors'] = None

  # print([params['X'][b2][subjIndCurrDis[s]] for b2 in range(params['nrBiomk'])])
  # print([params['Y'][b2][subjIndCurrDis[s]] for b2 in range(params['nrBiomk'])])

  for s in range(len(X[0])):
    entriesCurrSubj = [X[b][s].shape[0] > 0 for b in range(30)]
    nrEntriesPerSubj = np.sum(entriesCurrSubj)
    if nrEntriesPerSubj == 0:
      print(s, entriesCurrSubj)
      print(dadsa)

  print(labels)
  # print(dasda)

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

  transitionTimePriorMeanAD = 0.1 # using months instead of years
  transitionTimePriorMinAD = 0.09
  transitionTimePriorMaxAD = 0.11

  bPriorShapeDisAD, bPriorRateDisAD = getGammShapeRateFromTranTime(
    transitionTimePriorMeanAD, transitionTimePriorMinAD, transitionTimePriorMaxAD)

  _, bPriorStdAD = getMeanStdBFromTranTime(
    transitionTimePriorMeanAD, transitionTimePriorMinAD, transitionTimePriorMaxAD)

  transitionTimePriorMeanPCA = 500
  transitionTimePriorMinPCA = 400
  transitionTimePriorMaxPCA = 600

  bPriorShapeDisPCA, bPriorRateDisPCA = getGammShapeRateFromTranTime(
    transitionTimePriorMeanPCA, transitionTimePriorMinPCA, transitionTimePriorMaxPCA)

  _, bPriorStdPCA = getMeanStdBFromTranTime(
    transitionTimePriorMeanPCA, transitionTimePriorMinPCA, transitionTimePriorMaxPCA)

  params['priorsDisModels'] = [0, 0]
  # priors for tAD
  params['priorsDisModels'][0] = dict(meanA=1, stdA=1e-20, meanD=0, stdD=1e-20,
    shapeB=bPriorShapeDisAD, rateB=bPriorRateDisAD, stdPerturbB=bPriorStdAD, timeShiftStd=20000)
  # priors for PCA
  params['priorsDisModels'][1] = dict(meanA=1, stdA=1e-20, meanD=0, stdD=1e-20,
    shapeB=bPriorShapeDisPCA, rateB=bPriorRateDisPCA, stdPerturbB=bPriorStdPCA, timeShiftStd=20000)

  # params['priorsUnitModels'] = [dict(meanA=1, stdA=1e-20, meanD=0, stdD=1e-20,
  #  shapeB=2, rateB=2, timeShiftStd=20000) for d in range(nrDis)]
  params['priorsUnitModels'] = [dict(meanA=1, stdA=1e-5, meanD=0, stdD=1e-5,
    shapeB=bPriorShape, rateB=bPriorRate, timeShiftStd=20000) for u in range(nrFuncUnits-nrExtraBiomk)]

  if nrExtraBiomk > 0:
    params['priorsUnitModelsLinear'] = [dict(meanA=1, stdA=0.1, meanB=0, stdB=0.1, timeShiftStd=20000)
        for u in range(nrExtraBiomk)]
    params['priorsUnitModels'] += params['priorsUnitModelsLinear']


  bPriorShapeNoDKT, bPriorRateNoDKT = getGammShapeRateFromTranTime(
    transitionTimePriorMean=50, transitionTimePriorMin=40, transitionTimePriorMax=60)
  params['priorsNoDKTSigmoid'] = dict(meanA=1, stdA=1e-5, meanD=0, stdD=1e-5,
    shapeB=bPriorShapeNoDKT, rateB=bPriorRateNoDKT, timeShiftStd=20000)

  ######################

  nrBiomkDisModel = nrFuncUnits
  params['nrBiomkDisModel'] = nrBiomkDisModel

  if addExtraBiomk:
    params['plotTrajParams']['unitNames'] = unitNames + labels[-3:]
  else:
    params['plotTrajParams']['unitNames'] = unitNames

  # map which diagnoses belong to which disease
  # first disease has CTL+AD, second disease has CTL2+PCA
  params['diagsSetInDis'] = [np.array([CTL, MCI, AD, AD2]), np.array([CTL2, PCA])]
  params['disLabels'] = ['tAD', 'PCA']
  # if addExtraBiomk:
  #   params['otherBiomkPerDisease'] = [[nrBiomk-3,nrBiomk-2, nrBiomk-1], []] # can also add 3 extra cognitive tests
  # else:
  #   params['otherBiomkPerDisease'] = [[], []]

  params['binMaskSubjForEachDisD'] = [np.in1d(params['diag'],
                                      params['diagsSetInDis'][disNr]) for disNr in range(nrDis)]

  eps = 0.001
  nrXPoints = 50
  params['trueParams'] = {}
  subShiftsS = np.zeros(RID.shape[0])
  # params['trueParams']['trueSubjDysfuncScoresSU'] = np.zeros((RID.shape[0],nrFuncUnits))
  trueDysfuncXsX = np.linspace(0,1, nrXPoints)
  # params['trueParams']['trueTrajXB'] = eps * np.ones((nrXPoints, nrBiomk))
  trueTrajFromDysXB = eps * np.ones((nrXPoints, nrBiomk))

  trueLineSpacedDPSsX = np.linspace(-10,10, nrXPoints)
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

  print('diag', params['diag'].shape[0])
  # print(adsa)
  print('X[0]',len(params['X'][0]))
  assert params['diag'].shape[0] == len(params['X'][0])
  # assert params['diag'].shape[0] == len(params['trueParams']['subShiftsTrueMarcoFormatS'])
  # assert params['diag'].shape[0] == len(params['trueParams']['trueSubjDysfuncScoresSU'])

  # if args.penalty is not None:
  #   if np.abs(args.penalty - int(args.penalty) < 0.00001):
  #     expName = '%sPen%d' % (expName, args.penalty)
  #   else:
  #     expName = '%sPen%.1f' % (expName, args.penalty)

  # params['runPartStd'] = ['L', 'L']
  params['runPartStd'] = args.runPartStd
  params['runPartMain'] = ['R', 'I', 'I'] # [mainPart, plot, stage]
  params['masterProcess'] = args.runIndex == 0

  expNameDisOne = '%s' % expName
  modelNames, res = evaluationFramework.runModels(params, expName,
   args.modelToRun, runAllExpTadpoleDrc)


  if params['masterProcess']:
    printRes(modelNames, res, plotTrajParams, params)


def printRes(modelNames, res, plotTrajParams, params):
  #nrModels = len(modelNames)

  nrDis = params['nrDis']
  modelNames += ['Lin', 'Spline', 'Multivar']
  officialNames = {'JMD': 'DKT', 'Sig': 'Latent stage', 'Lin': 'Linear',
                   'Spline': 'Spline', 'Multivar': 'Multivariate'}

  d = 0

  disNrValid = 1
  disNrsValid = [disNrValid]

  biomkNames = res[0]['metrics']['labelsNonMri']


  resDf = pd.DataFrame(index=range(12 * len(disNrsValid)), columns=['Model'] + biomkNames)

  c = 0

  # dpmObjStd[s].plotter.plotAllBiomkDisSpace(dpmObjStd[s], params, disNr=0)
  # for disNrValid in disNrsValid:
  print('%d-%d training on dis %s   validation on disease %s' % (d, disNrValid,
    params['disLabels'][0], params['disLabels'][1]))

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
    mseMUB[m] = res[m]['metrics']['dpm']['mseUB']
    mseMeanMU[m] = np.nanmean(mseMUB[m], axis=1)
    mseStdMU[m] = np.nanstd(mseMUB[m], axis=1)

    corrMUB[m] = res[m]['metrics']['dpm']['corrUB']
    corrMeanMU[m] = np.nanmean(corrMUB[m], axis=1)
    corrStdMU[m] = np.nanstd(corrMUB[m], axis=1)

  mseMUB[linIndex] = res[0]['metrics']['lin']['mseUB']
  mseMeanMU[linIndex] = np.nanmean(mseMUB[linIndex], axis=1)
  mseStdMU[linIndex] = np.nanstd(mseMUB[linIndex], axis=1)

  mseMUB[splineIndex] = res[0]['metrics']['spline']['mseUB']
  mseMeanMU[splineIndex] = np.nanmean(mseMUB[splineIndex], axis=1)
  mseStdMU[splineIndex] = np.nanstd(mseMUB[splineIndex], axis=1)

  mseMUB[multivarIndex] = res[0]['metrics']['multivar']['mseUB']
  mseMeanMU[multivarIndex] = np.nanmean(mseMUB[multivarIndex], axis=1)
  mseStdMU[multivarIndex] = np.nanstd(mseMUB[multivarIndex], axis=1)


  corrMUB[linIndex] = res[0]['metrics']['lin']['corrUB']
  corrMeanMU[linIndex] = np.nanmean(corrMUB[linIndex], axis=1)
  corrStdMU[linIndex] = np.nanstd(corrMUB[linIndex], axis=1)

  corrMUB[splineIndex] = res[0]['metrics']['spline']['corrUB']
  corrMeanMU[splineIndex] = np.nanmean(corrMUB[splineIndex], axis=1)
  corrStdMU[splineIndex] = np.nanstd(corrMUB[splineIndex], axis=1)

  corrMUB[multivarIndex] = res[0]['metrics']['multivar']['corrUB']
  corrMeanMU[multivarIndex] = np.nanmean(corrMUB[multivarIndex], axis=1)
  corrStdMU[multivarIndex] = np.nanstd(corrMUB[multivarIndex], axis=1)

  # Perform Bonferroni correction
  sigLevel = 0.05/(6*2*nrModels)

  print('##### mean squared error and rank correlation ######')
  resDf.iloc[c, 0] = 'Prediction Error (MSE)'
  c += 1
  modelIndxs = [dktIndex, sigIndex, multivarIndex, splineIndex, linIndex]

  for m in modelIndxs:
    resDf.iloc[c,0] = officialNames[modelNames[m]]

    for u in range(mseMeanMU[m].shape[0]):
      sigLabel = getSigLabel(mseMUB[m][u, :], mseMUB[dktIndex][u, :], sigLevel)
      resDf.iloc[c, u+1] = '%.2f +/- %.2f%s' % (mseMeanMU[m][u], mseStdMU[m][u], sigLabel)

    c += 1

  resDf.iloc[c, 0] = 'Rank Correlation (Spearman rho)'
  c += 1

  for m in modelIndxs:
    resDf.iloc[c, 0] = officialNames[modelNames[m]]
    # c += 1

    for u in range(mseMeanMU[m].shape[0]):
      sigLabel = getSigLabel(corrMUB[m][u,:], corrMUB[dktIndex][u,:], sigLevel)
      resDf.iloc[c, u+1] = '%.2f +/- %.2f%s' % (corrMeanMU[m][u], corrStdMU[m][u], sigLabel)

    c += 1


  print(resDf)
  resDf.to_html('drcRes.html')
  resDf.loc[:, 'Model' : 'DTI FA Temporal'].to_latex('drcRes.tex', index=False)


def getSigLabel(xs, xsMyModel, sigLevel):
  tstatCorrDkt, pValCorrDkt = scipy.stats.ttest_rel(xs, xsMyModel)

  if pValCorrDkt < sigLevel:
    sigLabel = '*'
  else:
    sigLabel = ''

  return sigLabel

def runAllExpTadpoleDrc(params, expName, dpmBuilder, compareTrueParamsFunc = None):
  """ runs all experiments"""

  res = {}

  params['patientID'] = AD
  params['excludeID'] = -1
  params['excludeXvalidID'] = -1
  params['excludeStaging'] = [-1]

  params['outFolder'] = 'resfiles/%s' % expName
  params['expName'] = expName

  dpmObjStd, res['std'] = evaluationFramework.runStdDPM(params,
    expName, dpmBuilder, params['runPartMain'])

  # dpmObjStd.plotter.plotAllBiomkDisSpace(dpmObjStd, params, disNr=0)

  # perform the validation against DRC data
  res['metrics'] = validateDRCBiomk(dpmObjStd, params)


  return res



if __name__ == '__main__':
  main()


