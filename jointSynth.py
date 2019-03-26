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

import MarcoModel
import SigmoidModel

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

# parser.add_argument('--disModelObj', dest='disModelObj',
#   help=' either SigmoidModel or ')

parser.add_argument('--expName', dest="expName",
  help='synth1 or synth2')

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

hostName = gethostname()
if hostName == 'razvan-Inspiron-5547':
  homeDir = '/home/razvan'
  blenderPath = 'blender'
elif hostName == 'razvan-Precision-T1700':
  homeDir = '/home/razvan'
  blenderPath = 'blender'
elif args.cluster:
  homeDir = '/home/rmarines'
  blenderPath = '/share/apps/blender-2.75/blender'
else:
  raise ValueError('Wrong hostname. If running on new machine, add '
                   'application paths in python code above')


plotTrajParams = {}
plotTrajParams['SubfigTrajWinSize'] = (1600,900)
plotTrajParams['nrRows'] = args.nrRows
plotTrajParams['nrCols'] = args.nrCols
plotTrajParams['diagColors'] = {CTL:'g', MCI:'y', AD:'r',
  CTL2:'g', PCA:'y', AD2:'r'}
plotTrajParams['diagScatterMarkers'] = {CTL:'o', MCI:'o', AD:'o',
  CTL2:'x', PCA:'x', AD2:'x'}
plotTrajParams['legendCols'] = 4
plotTrajParams['diagLabels'] = {CTL:'CTL', AD:'AD', PCA:'PCA', CTL2:'CTL2'}
# plotTrajParams['ylimitsRandPoints'] = (-3,2)
plotTrajParams['blenderPath'] = blenderPath
plotTrajParams['isSynth'] = True
plotTrajParams['padTightLayout'] = 1



if args.agg:
  plotTrajParams['agg'] = True
else:
  plotTrajParams['agg'] = False

hostName = gethostname()
if hostName == 'razvan-Inspiron-5547':
  height = 350
else: #if hostName == 'razvan-Precision-T1700':
  height = 450


def main():

  nrSubjLong = 100
  nrTimepts = 4

  lowerAgeLim = 60
  upperAgeLim = 80

  shiftsLowerLim = -13
  shiftsUpperLim = 10

  outFolder = 'resfiles/synth/'

  expName = args.expName
  fileName = '%s.npz' % expName

  regenerateData = args.regData

  params = {}

  nrFuncUnits = 2
  nrBiomkInFuncUnits = 3
  nrDis = 2



  nrBiomk = nrBiomkInFuncUnits * nrFuncUnits
  mapBiomkToFuncUnits = np.array(list(range(nrFuncUnits)) * nrBiomkInFuncUnits)
  # should give smth like [0,1,2,3,0,1,2,3,0,1,2,3]
  print('mapBiomkToFuncUnits', mapBiomkToFuncUnits)

  biomkInFuncUnit = [0 for u in range(nrFuncUnits+1)]
  for u in range(nrFuncUnits):
    biomkInFuncUnit[u] = np.where(mapBiomkToFuncUnits == u)[0]

  biomkInFuncUnit[nrFuncUnits] = np.array([]) # need to leave this as empty list

  plotTrajParams['biomkInFuncUnit'] = biomkInFuncUnit
  plotTrajParams['labels'] = ['biomarker %d' % n for n in range(nrBiomk)]
  plotTrajParams['nrRowsFuncUnit'] = 3
  plotTrajParams['nrColsFuncUnit'] = 4
  plotTrajParams['colorsTrajBiomkB'] = [colorsys.hsv_to_rgb(hue, 1, 1) for hue in
    np.linspace(0, 1, num=nrBiomk, endpoint=False)]
  plotTrajParams['colorsTrajUnitsU'] = [colorsys.hsv_to_rgb(hue, 1, 1) for hue in
    np.linspace(0, 1, num=nrFuncUnits, endpoint=False)]

  # plotTrajParams['yNormMode'] = 'zScoreTraj'
  # plotTrajParams['yNormMode'] = 'zScoreEarlyStageTraj'
  plotTrajParams['yNormMode'] = 'unscaled'

  # if False, plot estimated traj. in separate plot from true traj.
  plotTrajParams['allTrajOverlap'] = False

  params['unitNames'] = ['Unit%d' % f for f in range(nrFuncUnits)]

  params['runIndex'] = args.runIndex
  params['nrProc'] = args.nrProc
  params['cluster'] = args.cluster
  params['plotTrajParams'] = plotTrajParams
  params['penalty'] = args.penalty
  params['penaltyUnits'] = 20
  params['penaltyDis'] = 1
  params['nrFuncUnits'] = nrFuncUnits
  params['biomkInFuncUnit'] = biomkInFuncUnit
  params['nrBiomkDisModel'] = nrFuncUnits


  params['nrGlobIterUnit'] = 10 # these parameters are specific for the Joint Model of Disease (JMD)
  params['iterParamsUnit'] = 50
  params['nrGlobIterDis'] = 10
  params['iterParamsDis'] = 50

  # # params['unitModelObjList'] = MarcoModel.GP_progression_model
  # params['unitModelObjList'] = SigmoidModel.SigmoidModel
  # params['disModelObj'] = SigmoidModel.SigmoidModel

  # by default we have no priors
  params['priors'] = None

  ####### set priors for specific models #########

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
  transitionTimePriorMin = 0.1
  transitionTimePriorMax = 10

  bPriorShape, bPriorRate = getGammShapeRateFromTranTime(
    transitionTimePriorMean, transitionTimePriorMin, transitionTimePriorMax)

  params['priorsDisModels'] = [dict(meanA=1, stdA=1e-5, meanD=0, stdD=1e-5,
    shapeB=bPriorShape, rateB=bPriorRate, timeShiftStd=15)
    for d in range(nrDis)]
  params['priorsUnitModels'] = [None for d in range(nrDis)]


  ##### disease agnostic parameters ###########
  # params of individual biomarkers
  thetas = np.zeros((nrBiomk, 4), float)
  thetas[:, 0] = 1
  thetas[:, 3] = 0
  for f in range(nrFuncUnits):
    thetas[mapBiomkToFuncUnits == f, 2] = np.linspace(0.2, 0.9, num=nrBiomkInFuncUnits, endpoint=True)

  # set first funtional unit to have traj with lower slopes
  thetas[mapBiomkToFuncUnits == 0, 1] = 5
  thetas[mapBiomkToFuncUnits == 1, 1] = 10
  # thetas[mapBiomkToFuncUnits == 2, 1] = 7


  if args.expName == 'synth1':
    sigmaB = 0.05 * np.ones(nrBiomk)
  elif args.expName == 'synth2':
    sigmaB = 0.01 * np.ones(nrBiomk)
  else:
    raise ValueError('expName should be synth1 or synth2')


  # scale every biomarker with mean and std.
  scalingBiomk2B = np.zeros((2, nrBiomk))
  # scalingBiomk2B[:, 0] = [200, 100] # mean +/- std
  # scalingBiomk2B[:, 0] = [200, 100]  # mean +/- std
  #
  # scalingBiomk2B[:, 1] = [-20, 3]  # mean +/- std
  # scalingBiomk2B[:, 1] = [-20, 3]  # mean +/- std
  #
  # scalingBiomk2B[:, 2:4] = scalingBiomk2B[:, 0:2]
  # scalingBiomk2B[:, 4:6] = scalingBiomk2B[:, 0:2]

  scalingBiomk2B[1,:] = 1

  ##### disease 1 - disease specific parameters ###########

  # params of the dysfunctional trajectories
  dysfuncParamsDisOne = np.zeros((nrFuncUnits, 4), float)
  dysfuncParamsDisOne[:, 0] = 1  # ak
  dysfuncParamsDisOne[:, 1] = [0.3, 0.2] # bk
  dysfuncParamsDisOne[:, 2] = [-4, 6]  # ck
  dysfuncParamsDisOne[:, 3] = 0  # dk

  synthModelDisOne = ParHierModel.ParHierModel(dysfuncParamsDisOne, thetas,
    mapBiomkToFuncUnits, sigmoidFunc, sigmaB)

  paramsDisOne = copy.deepcopy(params)

  paramsDisOne = genSynthData.generateDataJMD(nrSubjLong, nrBiomk, nrTimepts,
  shiftsLowerLim, shiftsUpperLim, synthModelDisOne, outFolder, fileName,
    regenerateData, paramsDisOne, scalingBiomk2B, ctlDiagNr=CTL, patDiagNr=AD)

  # paramsDisOne['plotTrajParams']['trueParams'] = paramsDisOne['trueParams']


  replaceFigMode = True

  if regenerateData:
    synthPlotter = Plotter.PlotterJDM(paramsDisOne['plotTrajParams'])
    fig = synthPlotter.plotTrajDataMarcoFormat(paramsDisOne['X'], paramsDisOne['Y'],
      paramsDisOne['diag'], synthModelDisOne, paramsDisOne['trueParamsDis'], replaceFigMode=replaceFigMode)
    fig.savefig('%s/%sDis1GenData.png' % (outFolder, expName))

  ##### disease 2 - disease specific parameters ###########

  # params of the dysfunctional trajectories
  dysfuncParamsDisTwo = copy.deepcopy(dysfuncParamsDisOne)
  dysfuncParamsDisTwo[:, 1] = [0.3, 0.2] # bk
  dysfuncParamsDisTwo[:, 2] = [6, -4]

  synthModelDisTwo = ParHierModel.ParHierModel(dysfuncParamsDisTwo, thetas, mapBiomkToFuncUnits, sigmoidFunc, sigmaB)

  paramsDisTwo = copy.deepcopy(paramsDisOne)
  nrSubjLongDisTwo = 50
  nrTimeptsDisTwo = 4

  paramsDisTwo = genSynthData.generateDataJMD(nrSubjLongDisTwo, nrBiomk,
    nrTimeptsDisTwo, shiftsLowerLim, shiftsUpperLim, synthModelDisTwo,
    outFolder, fileName, regenerateData, paramsDisTwo, scalingBiomk2B,
    ctlDiagNr=CTL2, patDiagNr=PCA)

  # for disease two, only keep the second biomarker in each functional unit
  indBiomkInDiseaseTwo = np.array(range(nrFuncUnits,(2*nrFuncUnits)))
  print('indBiomkInDiseaseTwo', indBiomkInDiseaseTwo)
  paramsDisTwo['Xtrue'] = paramsDisTwo['X']
  paramsDisTwo['Ytrue'] = paramsDisTwo['Y']

  # for disease two, change the format of the X and Y arrays, add the missing biomarkers with empty lists
  XemptyListsAllBiomk = [0 for _ in range(nrBiomk)]
  YemptyListsAllBiomk = [0 for _ in range(nrBiomk)]
  visitIndicesDisTwoMissing = [0 for _ in range(nrBiomk)]
  for b in range(nrBiomk):
    XemptyListsAllBiomk[b] = [0 for _ in range(nrSubjLongDisTwo)]
    YemptyListsAllBiomk[b] = [0 for _ in range(nrSubjLongDisTwo)]
    visitIndicesDisTwoMissing[b] = [0 for _ in range(nrSubjLongDisTwo)]

    for s in range(nrSubjLongDisTwo):
      if b in indBiomkInDiseaseTwo:
        XemptyListsAllBiomk[b][s] = paramsDisTwo['Xtrue'][b][s]
        YemptyListsAllBiomk[b][s] = paramsDisTwo['Ytrue'][b][s]
        visitIndicesDisTwoMissing[b][s] = paramsDisTwo['visitIndices'][b][s]
      else:
        XemptyListsAllBiomk[b][s] = np.array([])
        YemptyListsAllBiomk[b][s] = np.array([])
        visitIndicesDisTwoMissing[b][s] = np.array([])

  paramsDisTwo['XemptyListsAllBiomk'] = XemptyListsAllBiomk
  paramsDisTwo['YemptyListsAllBiomk'] = YemptyListsAllBiomk
  paramsDisTwo['visitIndicesMissing'] = visitIndicesDisTwoMissing

  if regenerateData:
    synthPlotter = Plotter.PlotterJDM(paramsDisTwo['plotTrajParams'])
    fig = synthPlotter.plotTrajDataMarcoFormat(paramsDisTwo['Xtrue'],
      paramsDisTwo['Ytrue'], paramsDisTwo['diag'],
      synthModelDisTwo, paramsDisTwo['trueParamsDis'], replaceFigMode=replaceFigMode)
    fig.savefig('%s/%sDis2GenDataFull.png' % (outFolder, expName))

    synthPlotter = Plotter.PlotterJDM(paramsDisTwo['plotTrajParams'])
    fig = synthPlotter.plotTrajDataMarcoFormat(paramsDisTwo['XemptyListsAllBiomk'],
      paramsDisTwo['YemptyListsAllBiomk'], paramsDisTwo['diag'],
      synthModelDisTwo, paramsDisTwo['trueParamsDis'], replaceFigMode=replaceFigMode)
    fig.savefig('%s/%sDis2GenDataMissing.png' % (outFolder, expName))


  ############### now merge the two datasets ############

  # add the biomarkers from the second dataset, same format as dataset 1
  # but with missing entries
  params = paramsDisOne
  for b in range(nrBiomk):
    params['X'][b] += paramsDisTwo['XemptyListsAllBiomk'][b]
    params['Y'][b] += paramsDisTwo['YemptyListsAllBiomk'][b]
    params['visitIndices'][b] += paramsDisTwo['visitIndicesMissing'][b]

  # print('visitIndicesDisTwoMissing', visitIndicesDisTwoMissing)
  # print(adssa)

  params['RID'] = np.concatenate((params['RID'],
  nrSubjLong + paramsDisTwo['RID']),axis=0) # RIDs must be different

  # this is the full vector of diagnoses for all diseases
  params['diag'] = np.concatenate((paramsDisOne['diag'], paramsDisTwo['diag']),axis=0)
  params['plotTrajParams']['diag'] = params['diag']

  params['trueParamsDis'] = [params['trueParamsDis'], paramsDisTwo['trueParamsDis']]

  for f in range(nrFuncUnits):
    params['trueParamsFuncUnits'][f]['subShiftsS'] = np.concatenate(
      (params['trueParamsFuncUnits'][f]['subShiftsS'],
      paramsDisTwo['trueParamsFuncUnits'][f]['subShiftsS']),axis=0)

  # map which diagnoses belong to which disease
  # first disease has CTL+AD, second disease has CTL2+PCA
  params['diagsSetInDis'] = [np.array([CTL, AD]), np.array([CTL2, PCA])]
  params['disLabels'] = ['Dis0', 'Dis1']
  params['otherBiomkPerDisease'] = [[], []]

  params['binMaskSubjForEachDisD'] = [np.in1d(params['diag'],
                                      params['diagsSetInDis'][disNr]) for disNr in range(nrDis)]

  assert params['diag'].shape[0] == len(params['X'][0])
  assert np.sum(params['binMaskSubjForEachDisD'][0]) == len(params['trueParamsDis'][0]['subShiftsS'])
  assert params['diag'].shape[0] == len(params['trueParamsFuncUnits'][0]['subShiftsS'])

  # if np.abs(args.penalty - int(args.penalty) < 0.00001):
  #   expName = '%sPen%d' % (expName, args.penalty)
  # else:
  #   expName = '%sPen%.1f' % (expName, args.penalty)

  params['runPartStd'] = args.runPartStd
  params['runPartMain'] = ['R', 'I', 'I'] # [mainPart, plot, stage]
  params['masterProcess'] = args.runIndex == 0

  expNameDisOne = '%s' % expName
  modelNames, res = evaluationFramework.runModels(params, expName,
    args.modelToRun, runAllExpSynth)


def runAllExpSynth(params, expName, dpmBuilder, compareTrueParamsFunc = None):
  """ runs all experiments"""

  res = {}

  params['patientID'] = AD
  params['excludeID'] = -1
  params['excludeXvalidID'] = -1
  params['excludeStaging'] = [-1]

  params['outFolder'] = 'resfiles/synth/%s' % expName

  dpmObjStd, res['std'] = evaluationFramework.runStdDPM(params,
    expName, dpmBuilder, params['runPartMain'])

  return res


def transferProgression(dpmObjStdDisOne, paramsDisTwo,
  expNameDisTwo, dpmBuilderDisTwo, runPart):

  dataIndices = np.logical_not(np.in1d(paramsDisTwo['diag'], paramsDisTwo['excludeXvalidID']))
  print(np.sum(np.logical_not(dataIndices)))
  print('excludeID', params['excludeXvalidID'])
  print(params['diag'].shape)
  dpmObj = dpmBuilder.generate(dataIndices, expNameDisTwo, paramsDisTwo)
  res = None
  if runPart[0] == 'R':
    res = dpmObj.runStd(params['runPartStd'])





if __name__ == '__main__':
  main()


