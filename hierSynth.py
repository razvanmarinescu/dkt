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

# sys.path.append(os.path.abspath("../diffEqModel/"))


parser = argparse.ArgumentParser(description='Launches two diseases hierarchical model ...'
'i.e. runs two separate models on datasets distinctively')

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
else:
  raise ValueError('Wrong hostname. If running on new machine, add '
                   'application paths in python code above')


plotTrajParams = {}
plotTrajParams['SubfigTrajWinSize'] = (1600,900)
plotTrajParams['nrRows'] = args.nrRows
plotTrajParams['nrCols'] = args.nrCols
plotTrajParams['diagColors'] = {CTL:'b', AD:'r'}
plotTrajParams['legendCols'] = 2
plotTrajParams['diagLabels'] = {CTL:'CTL', AD:'AD'}
plotTrajParams['freesurfPath'] = freesurfPath
# plotTrajParams['ylimitsRandPoints'] = (-3,2)
plotTrajParams['blenderPath'] = blenderPath
plotTrajParams['isSynth'] = True

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
  nrBiomk = 4
  nrTimepts = 4

  lowerAgeLim = 60
  upperAgeLim = 80

  shiftsLowerLim = -13
  shiftsUpperLim = 10

  etaB = 1 * np.ones(nrBiomk)
  lB = 10 * np.ones(nrBiomk)
  epsB = 1 * np.ones(nrBiomk)
  sigmaSB = 2 * np.ones((nrSubjLong, nrBiomk))

  sigmaGfunc = GPModel.genSigmaG
  sigmaEpsfunc = None
  sigmaSfunc  = None

  outFolder = 'resfiles/synth/'

  expName = 'synth1'
  fileName = '%s.npz' % expName

  forceRegenerate = False

  params = {}

  nrFuncUnits = 2
  nrBiomkInFuncUnits = 3

  nrBiomk = nrBiomkInFuncUnits * nrFuncUnits
  mapBiomkToFuncUnits = np.array(list(range(nrFuncUnits)) * nrBiomkInFuncUnits)
  # should give smth like [0,1,2,3,0,1,2,3,0,1,2,3]
  print('mapBiomkToFuncUnits', mapBiomkToFuncUnits)

  plotTrajParams['mapBiomkToFuncUnits'] = mapBiomkToFuncUnits
  plotTrajParams['labels'] = ['b%d' % n for n in range(nrBiomk)]
  plotTrajParams['nrRowsFuncUnit'] = 2
  plotTrajParams['nrColsFuncUnit'] = 3
  plotTrajParams['colorsTraj'] = [colorsys.hsv_to_rgb(hue, 1, 1) for hue in np.linspace(0, 1, num=nrBiomk, endpoint=False)]

  # if False, plot estimated traj. in separate plot from true traj.
  plotTrajParams['allTrajOverlap'] = False

  params['runIndex'] = args.runIndex
  params['nrProc'] = args.nrProc
  params['cluster'] = args.cluster
  params['plotTrajParams'] = plotTrajParams
  params['penalty'] = args.penalty
  params['nrFuncUnits'] = nrFuncUnits
  params['mapBiomkToFuncUnits'] = mapBiomkToFuncUnits

  ##### disease agnostic parameters ###########
  # params of individual biomarkers
  thetas = np.zeros((nrBiomk, 4), float)
  thetas[:, 0] = 1
  thetas[:, 1] = 10
  thetas[:, 3] = 0
  for f in range(nrFuncUnits):
    thetas[mapBiomkToFuncUnits == f, 2] = np.linspace(0.2, 0.9, num=nrBiomkInFuncUnits, endpoint=True)

  sigmaB = 0.1 * np.ones(nrBiomk)

  ##### disease specific parameters ###########
  # params of the dysfunctional trajectories - disease 1
  dysfuncParamsDisOne = np.zeros((nrFuncUnits, 4), float)
  dysfuncParamsDisOne[:, 0] = 1  # ak
  dysfuncParamsDisOne[:, 1] = 0.3  # bk
  dysfuncParamsDisOne[:, 2] = [-3, 7]  # ck
  dysfuncParamsDisOne[:, 3] = 0  # dk

  synthModelDisOne = ParHierModel.ParHierModel(dysfuncParamsDisOne, thetas, mapBiomkToFuncUnits, sigmoidFunc, sigmaB)

  paramsDisOne = genSynthData.generateDataJMD(nrSubjLong, nrBiomk, nrTimepts, lowerAgeLim,
    upperAgeLim, shiftsLowerLim, shiftsUpperLim, synthModelDisOne, outFolder, fileName, forceRegenerate, params)

  paramsDisOne['plotTrajParams']['diagNrs'] = np.unique(paramsDisOne['diag'])
  paramsDisOne['plotTrajParams']['trueParams'] = paramsDisOne['trueParams']

  if forceRegenerate:
    synthPlotter = Plotter.PlotterJDM(paramsDisOne['plotTrajParams'])
    fig = synthPlotter.plotTrajData(paramsDisOne['longData'],
      paramsDisOne['longDiag'], paramsDisOne['trueParams']['dpsLong'],
      synthModelDisOne, replaceFigMode=True)
    fig.savefig('%s/synth1Dis1GenData.png' % outFolder)

  # params of the dysfunctional trajectories - disease 2
  dysfuncParamsDisTwo = copy.deepcopy(dysfuncParamsDisOne)
  dysfuncParamsDisTwo[:, 1] = 1
  dysfuncParamsDisTwo[:, 2] = [8, -4]

  synthModelDisTwo = ParHierModel.ParHierModel(dysfuncParamsDisTwo, thetas, mapBiomkToFuncUnits, sigmoidFunc, sigmaB)

  paramsDisTwo = copy.deepcopy(paramsDisOne)

  paramsDisTwo = genSynthData.generateDataJMD(nrSubjLong, nrBiomk, nrTimepts, lowerAgeLim,
    upperAgeLim, shiftsLowerLim, shiftsUpperLim, synthModelDisTwo, outFolder, fileName, forceRegenerate, paramsDisTwo)


  # for disease two, only keep the second biomarker in each functional unit
  indBiomkInDiseaseTwo = np.array(range(nrFuncUnits,(2*nrFuncUnits)))
  print('indBiomkInDiseaseTwo', indBiomkInDiseaseTwo)
  paramsDisTwo['Xtrue'] = paramsDisTwo['X']
  paramsDisTwo['Ytrue'] = paramsDisTwo['Y']
  paramsDisTwo['X'] = [paramsDisTwo['X'][b] for b in indBiomkInDiseaseTwo]
  paramsDisTwo['Y'] = [paramsDisTwo['Y'][b] for b in indBiomkInDiseaseTwo]
  paramsDisTwo['mapBiomkToFuncUnits'] = np.array([mapBiomkToFuncUnits[b] for b in indBiomkInDiseaseTwo])

  # for disease two, change the format of the X and Y arrays, add the missing biomarkers with empty lists
  XemptyListsAllBiomk = [0 for _ in range(nrBiomk)]
  YemptyListsAllBiomk = [0 for _ in range(nrBiomk)]
  for b in range(nrBiomk):
    XemptyListsAllBiomk[b] = [0 for _ in range(nrSubjLong)]
    YemptyListsAllBiomk[b] = [0 for _ in range(nrSubjLong)]

    for s in range(nrSubjLong):
      if b in indBiomkInDiseaseTwo:
        XemptyListsAllBiomk[b][s] = paramsDisTwo['Xtrue'][b][s]
        YemptyListsAllBiomk[b][s] = paramsDisTwo['Ytrue'][b][s]
      else:
        XemptyListsAllBiomk[b][s] = np.array([])
        YemptyListsAllBiomk[b][s] = np.array([])

  paramsDisTwo['XemptyListsAllBiomk'] = XemptyListsAllBiomk
  paramsDisTwo['YemptyListsAllBiomk'] = YemptyListsAllBiomk

  paramsDisTwo['plotTrajParams']['diagNrs'] = np.unique(paramsDisTwo['diag'])
  paramsDisTwo['plotTrajParams']['trueParams'] = paramsDisTwo['trueParams']
  paramsDisTwo['plotTrajParams']['trueParams']['trueTrajPredXB'] = \
    paramsDisTwo['plotTrajParams']['trueParams']['trueTrajPredXB'][:,indBiomkInDiseaseTwo]

  paramsDisTwo['plotTrajParams']['labels'] = \
    [[paramsDisTwo['plotTrajParams']['labels'][b]] for b in indBiomkInDiseaseTwo]

  if forceRegenerate:
    synthPlotter = Plotter.PlotterJDM(paramsDisTwo['plotTrajParams'])
    fig = synthPlotter.plotTrajData(paramsDisTwo['longData'],
      paramsDisTwo['longDiag'], paramsDisTwo['trueParams']['dpsLong'],
      synthModelDisTwo, replaceFigMode=True)
    fig.savefig('%s/synth1Dis2GenData.png' % outFolder)


  if np.abs(args.penalty - int(args.penalty) < 0.00001):
    expName = '%sPen%d' % (expName, args.penalty)
  else:
    expName = '%sPen%.1f' % (expName, args.penalty)

  paramsDisOne['runPartStd'] = ['L', 'L']
  paramsDisOne['runPartMain'] = ['R', 'I', 'I'] # [mainPart, plot, stage]
  paramsDisOne['masterProcess'] = args.runIndex == 0

  expNameDisOne = '%sDisOne' % expName
  modelNames, res = evaluationFramework.runModels(paramsDisOne, expNameDisOne, args.modelToRun, runAllExpSynth)

  paramsDisTwo['filePathUnitModels'] = '%s/%s_JMD/unitModels.npz' % (outFolder, expNameDisOne)

  paramsDisTwo['runPartStd'] = ['R', 'R']
  paramsDisTwo['runPartMain'] = ['R', 'I', 'I']  # [mainPart, plot, stage]
  paramsDisTwo['masterProcess'] = args.runIndex == 0

  modelDisTwo = 16
  expNameDisTwo = '%sDisTwo' % expName
  modelNames, res = evaluationFramework.runModels(paramsDisTwo, expNameDisTwo, modelDisTwo, runAllExpSynth)


def runAllExpSynth(params, expName, dpmBuilder, compareTrueParamsFunc = None):
  """ runs all experiments"""

  res = {}

  dpmBuilder.plotterObj.plotTrajParams = params['plotTrajParams']

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


