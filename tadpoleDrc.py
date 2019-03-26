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

hostName = gethostname()
if hostName == 'razvan-Inspiron-5547':
  freesurfPath = '/usr/local/freesurfer-5.3.0'
  homeDir = '/home/razvan'
  blenderPath = 'blender'
elif hostName == 'razvan-Precision-T1700':
  freesurfPath = '/usr/local/freesurfer-5.3.0'
  homeDir = '/home/razvan'
  blenderPath = 'blender'
elif args.cluster:
  freesurfPath = '/share/apps/freesurfer-5.3.0'
  homeDir = '/home/rmarines'
  blenderPath = '/share/apps/blender-2.75/blender'
elif hostName == 'planell-VirtualBox':
  homeDir = '/home/planell'
  freesurfPath = ""
  blenderPath = ""
else:
  raise ValueError('Wrong hostname. If running on new machine, add '
                   'application paths in python code above')


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

plotTrajParams['freesurfPath'] = freesurfPath
plotTrajParams['blenderPath'] = blenderPath
plotTrajParams['isSynth'] = False
plotTrajParams['padTightLayout'] = 0.0

if args.agg:
  plotTrajParams['agg'] = True
else:
  plotTrajParams['agg'] = False

hostName = gethostname()
if hostName == 'razvan-Inspiron-5547':
  height = 350
else: #if hostName == 'razvan-Precision-T1700':
  height = 450

if hostName == 'razvan-Inspiron-5547':
  homeDir = '/home/razvan'
  freesurfPath = '/usr/local/freesurfer-6.0.0'
elif hostName == 'razvan-Precision-T1700':
  homeDir = '/home/razvan'
  freesurfPath = '/usr/local/freesurfer-6.0.0'
elif args.cluster:
  homeDir = '/home/rmarines'
  freesurfPath = '/home/rmarines/src/freesurfer-6.0.0'
elif hostName == 'planell-VirtualBox':
  homeDir = '/home/planell'
  freesurfPath = ""
  blenderPath = ""
else:
  raise ValueError('wrong hostname or cluster flag')



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

  addExtraBiomk = False

  np.random.seed(1)
  random.seed(1)
  pd.set_option('display.max_columns', 50)
  tinyData = args.tinyData
  regenerateData = args.regData
  if tinyData:
    finalDataFile = 'tadpoleDrcTiny.npz'
    expName = 'tad-drcTiny'
  else:
    finalDataFile = 'tadpoleDrcFinalDataWithRegParams.npz'
    expName = 'tad-drc'

  if regenerateData:
    prepareData(finalDataFile, tinyData, addExtraBiomk)

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

  # visDataHist(dataDfAll)
  nrUnqDiags = np.unique(dataDfAll.diag)
  print(dataDfAll)
  for d in nrUnqDiags:
    idxCurrDiag = ds['diag'] == d
    print('nr subj %s %d' % (plotTrajParams['diagLabels'][d], np.sum(idxCurrDiag)))
    # avgScans = []
    # print('avg scans %s %d' % plotTrajParams['diagLabels'][d])

  meanVols = np.array([np.mean(Y[0][s]) for s in range(RID.shape[0])])
  meanVols[diag != CTL2] = np.inf
  idxOfDRCSubjWithLowVol = np.argmin(meanVols)
  print('idxOfDRCSubjWithLowVol', idxOfDRCSubjWithLowVol)
  print(diag[idxOfDRCSubjWithLowVol])

  outFolder = 'resfiles/'

  params = {}

  nrFuncUnits = 6
  nrBiomkInFuncUnits = 5
  nrDis = 2 # nr of diseases

  # nrBiomk = nrBiomkInFuncUnits * nrFuncUnits
  # print(labels)
  # print(adss)
  # mapBiomkToFuncUnits = np.array(list(range(nrFuncUnits)) * nrBiomkInFuncUnits)
  # should give smth like [0,1,2,3,0,1,2,3,0,1,2,3]



  # change the order of the functional units so that the hippocampus and occipital are fitted first
  unitPermutation = [5,3,2,1,4,0]
  if addExtraBiomk:
    mapBiomkToFuncUnits = np.array((unitPermutation * nrBiomkInFuncUnits) + [-1,-1,-1])
  else:
    mapBiomkToFuncUnits = np.array((unitPermutation * nrBiomkInFuncUnits))

  unitNames = [l.split(' ')[-1] for l in labels]
  unitNames = [unitNames[i] for i in unitPermutation]
  nrBiomk = mapBiomkToFuncUnits.shape[0]
  biomkInFuncUnit = [0 for u in range(nrFuncUnits + 1)]
  for u in range(nrFuncUnits):
    biomkInFuncUnit[u] = np.where(mapBiomkToFuncUnits == u)[0]

  if addExtraBiomk:
    # add extra entry with other biomks to be added in the disease models
    biomkInFuncUnit[nrFuncUnits] = np.array([nrBiomk-3, nrBiomk-2, nrBiomk-1])
  else:
    biomkInFuncUnit[nrFuncUnits] = np.array([])  # need to leave this as empty list

  nrExtraBiomkInDisModel = biomkInFuncUnit[-1].shape[0]

  plotTrajParams['biomkInFuncUnit'] = biomkInFuncUnit
  plotTrajParams['labels'] = labels
  plotTrajParams['nrRowsFuncUnit'] = 3
  plotTrajParams['nrColsFuncUnit'] = 4
  plotTrajParams['colorsTrajBiomkB'] = [colorsys.hsv_to_rgb(hue, 1, 1) for hue in
    np.linspace(0, 1, num=nrBiomk, endpoint=False)]
  plotTrajParams['colorsTrajUnitsU'] = [colorsys.hsv_to_rgb(hue, 1, 1) for hue in
    np.linspace(0, 1, num=nrFuncUnits + nrExtraBiomkInDisModel, endpoint=False)]
  plotTrajParams['nrBiomk'] = 3

  # plotTrajParams['yNormMode'] = 'zScoreTraj'
  # plotTrajParams['yNormMode'] = 'zScoreEarlyStageTraj'
  plotTrajParams['yNormMode'] = 'unscaled'

  # if False, plot estimated traj. in separate plot from true traj.
  plotTrajParams['allTrajOverlap'] = False

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

  params['nrGlobIterUnit'] = 10 # these parameters are specific for the Joint Model of Disease (JMD)
  params['iterParamsUnit'] = 60
  params['nrGlobIterDis'] = 10
  params['iterParamsDis'] = 60

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

  # print('bPriorShape', bPriorShape)
  # print('bPriorRate', bPriorRate)
  # print(adsas)

  params['priorsDisModels'] = [dict(meanA=1, stdA=1e-20, meanD=0, stdD=1e-20,
    shapeB=2, rateB=2, timeShiftStd=20000) for d in range(nrDis)]
  # params['priorsUnitModels'] = [dict(meanA=1, stdA=1e-20, meanD=0, stdD=1e-20,
  #  shapeB=2, rateB=2, timeShiftStd=20000) for d in range(nrDis)]
  params['priorsUnitModels'] = [dict(meanA=1, stdA=1e-5, meanD=0, stdD=1e-5,
    shapeB=bPriorShape, rateB=bPriorRate, timeShiftStd=20000) for u in range(nrFuncUnits)]

  bPriorShapeNoDKT, bPriorRateNoDKT = getGammShapeRateFromTranTime(
    transitionTimePriorMean=50, transitionTimePriorMin=40, transitionTimePriorMax=60)
  params['priorsNoDKTSigmoid'] = dict(meanA=1, stdA=1e-5, meanD=0, stdD=1e-5,
    shapeB=bPriorShapeNoDKT, rateB=bPriorRateNoDKT, timeShiftStd=20000)

  ######################

  nrBiomkDisModel = nrFuncUnits + nrExtraBiomkInDisModel
  params['nrBiomkDisModel'] = nrBiomkDisModel

  if addExtraBiomk:
    params['plotTrajParams']['unitNames'] = unitNames + labels[-3:]
  else:
    params['plotTrajParams']['unitNames'] = unitNames

  # map which diagnoses belong to which disease
  # first disease has CTL+AD, second disease has CTL2+PCA
  params['diagsSetInDis'] = [np.array([CTL, MCI, AD, AD2]), np.array([CTL2, PCA])]
  params['disLabels'] = ['tAD', 'PCA']
  if addExtraBiomk:
    params['otherBiomkPerDisease'] = [[nrBiomk-3,nrBiomk-2, nrBiomk-1], []] # can also add 3 extra cognitive tests
  else:
    params['otherBiomkPerDisease'] = [[], []]

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

  from tadpoleSubtypes import printRes

  if params['masterProcess']:
    printRes(modelNames, res, plotTrajParams, params)


# def printRes(modelNames, res, plotTrajParams):
#   #nrModels = len(modelNames)
#
#   print(modelNames)
#
#   modelNames += ['Lin']
#
#   dktModelName = 'JMD'
#   sigModelName = 'Sig'
#   linModelName = 'Lin'
#
#   officialNames = {'JMD' :  'DKT', 'Sig' : 'Latent stage model', 'Lin' : 'Linear Model'}
#
#   dktIndex = 0
#   sigIndex = 1
#   linIndex = 2
#
#   # print('##### biomk prediction ######')
#   nrModels = len(officialNames)
#   mseMUB = list(range(nrModels))
#   mseMeanMU = list(range(nrModels))
#   mseStdMU = list(range(nrModels))
#
#   corrMUB = list(range(nrModels))
#   corrMeanMU = list(range(nrModels))
#   corrStdMU = list(range(nrModels))
#   pvalsMU = list(range(nrModels))
#
#   for m in range(nrModels-1):
#     mseMUB[m] = res[m]['metrics']['dpm']['mseUB']
#     mseMeanMU[m] = np.nanmean(mseMUB[m], axis=1)
#     mseStdMU[m] = np.nanstd(mseMUB[m], axis=1)
#
#     corrMUB[m] = res[m]['metrics']['dpm']['corrUB']
#     # pvalsMU[m] = res[m]['metrics']['dpm']['pValsUB']
#
#     corrMeanMU[m] = np.nanmean(corrMUB[m], axis=1)
#     corrStdMU[m] = np.nanstd(corrMUB[m], axis=1)
#
#   mseMUB[linIndex] = res[0]['metrics']['lin']['mseUB']
#   mseMeanMU[linIndex] = np.nanmean(mseMUB[linIndex], axis=1)
#   mseStdMU[linIndex] = np.nanstd(mseMUB[linIndex], axis=1)
#
#   corrMUB[linIndex] = res[0]['metrics']['lin']['corrUB']
#   corrMeanMU[linIndex] = np.nanmean(corrMUB[linIndex], axis=1)
#   corrStdMU[linIndex] = np.nanstd(corrMUB[linIndex], axis=1)
#   # pvalsMU[linIndex] = res[0]['metrics']['lin']['pValsU']
#
#   biomkNames = [x.split(' ')[-1] for x in res[0]['metrics']['labelsDti']]
#
#   # Perform Bonferroni correction
#   sigLevel = 0.05/(6*2*2)
#
#   # print('mseMUB[dktIndex][u, :]', np.nanmean(mseMUB[dktIndex][1, :]), mseMUB[dktIndex][1, :])
#   # print('mseMUB[linIndex][u, :]', np.nanmean(mseMUB[linIndex][1, :]), mseMUB[linIndex][1, :])
#   # pl.figure(1)
#   # pl.hist(mseMUB[dktIndex][1, :], color='g', label='dkt')
#   # pl.hist(mseMUB[linIndex][1, :], color='r', label='lin')
#   # pl.show()
#
#   print('##### mean squared error and rank correlation ######')
#   print(r'''\textbf{Model} & ''' + ' & '.join(['\\textbf{%s}' % b for b in biomkNames]) + '\\\\')
#   print('& \multicolumn{6}{c}{\\textbf{Prediction Error (MSE)}}\\\\')
#   for m in [dktIndex, sigIndex, linIndex]:
#     print('%s' % officialNames[modelNames[m]], end='')
#
#     for u in range(mseMeanMU[m].shape[0]):
#       sigLabel = getSigLabel(mseMUB[m][u, :], mseMUB[dktIndex][u, :], sigLevel)
#       print(' & %.2f$\pm$%.2f%s' % (mseMeanMU[m][u], mseStdMU[m][u], sigLabel), end='')
#
#     print('\\\\')
#
#   print('& \multicolumn{6}{c}{\\textbf{Rank Correlation (Spearman rho)}}\\\\')
#   for m in [dktIndex, sigIndex, linIndex]:
#     print('%s ' % officialNames[modelNames[m]], end='')
#
#
#     for u in range(mseMeanMU[m].shape[0]):
#       sigLabel = getSigLabel(corrMUB[m][u,:], corrMUB[dktIndex][u,:], sigLevel)
#       print(' & %.2f%s ' % (corrMeanMU[m][u], sigLabel), end='')
#
#     print('\\\\' )


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


