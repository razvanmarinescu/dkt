import sys
import numpy
import numpy as np
import colorsys
from socket import gethostname
import time
import argparse
import os
import colorsys

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

  # params of the dysfunctional trajectories (in the disease specific model)
  dysfuncParams = np.zeros((nrFuncUnits, 4), float)
  dysfuncParams[:, 0] = 1  # ak
  dysfuncParams[:, 1] = 0.3  # bk
  dysfuncParams[:, 2] = [-3, 7]  # ck
  dysfuncParams[:, 3] = 0  # dk

  # params of individual biomarkers
  thetas = np.zeros((nrBiomk, 4), float)
  thetas[:, 0] = 1
  thetas[:, 1] = 10
  thetas[:, 3] = 0
  for f in range(nrFuncUnits):
    thetas[mapBiomkToFuncUnits == f, 2] = np.linspace(0.2, 0.9, num = nrBiomkInFuncUnits, endpoint = True)

  sigmaB = 0.1 * np.ones(nrBiomk)
  synthModel = ParHierModel.ParHierModel(dysfuncParams, thetas, mapBiomkToFuncUnits, sigmoidFunc, sigmaB)

  params = genSynthData.generateDataJMD(nrSubjLong, nrBiomk, nrTimepts, lowerAgeLim,
    upperAgeLim, shiftsLowerLim, shiftsUpperLim, synthModel, outFolder, fileName, forceRegenerate, params)

  plotTrajParams['diagNrs'] = np.unique(params['diag'])
  plotTrajParams['mapBiomkToFuncUnits'] = mapBiomkToFuncUnits
  plotTrajParams['trueParams'] = params['trueParams']
  plotTrajParams['labels'] = ['b%d' % n for n in range(nrBiomk)]
  plotTrajParams['nrRowsFuncUnit'] = 2
  plotTrajParams['nrColsFuncUnit'] = 2
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

  # params['data'] = dataCross
  # params['diag'] = diagCross
  # params['scanTimepts'] = scanTimeptsCross
  # params['partCode'] = partCodeCross
  # params['ageAtScan'] = ageAtScanCrossZ
  # params['trueParams'] = trueParams

  biomkCols = np.array([colorsys.hsv_to_rgb(hue,1,1) for hue in np.linspace(0,1,num=nrBiomk,endpoint=False)])

  if forceRegenerate:
    synthPlotter = Plotter.PlotterJDM(plotTrajParams)
    fig = synthPlotter.plotTrajData(params['longData'], params['longDiag'], params['trueParams']['dpsLong'],
      synthModel, replaceFigMode=True)
    fig.savefig('%s/synth1GeneratedData.png' % outFolder)

  if np.abs(args.penalty - int(args.penalty) < 0.00001):
    expName = '%sPen%d' % (expName, args.penalty)
  else:
    expName = '%sPen%.1f' % (expName, args.penalty)

  params['runPartStd'] = ['L', 'R']
  # [mainPart, plot, stage]
  params['runPartMain'] = ['R', 'R', 'I']

  params['masterProcess'] = args.runIndex == 0

  modelNames, res = evaluationFramework.runModels(params, expName, args.modelToRun, runAllExpSynth)


def runAllExpSynth(params, expName, dpmBuilder, compareTrueParamsFunc = None):
  """ runs all experiments"""

  res = {}

  params['patientID'] = AD
  params['excludeID'] = -1
  params['excludeXvalidID'] = -1
  params['excludeStaging'] = [-1]

  params['outFolder'] = 'resfiles/synth/%s' % expName

  dpmObjStd, res['std'] = evaluationFramework.runStdDPM(params, expName, dpmBuilder,
    params['runPartMain'])

  if 'compareTrueParamsFunc' in params.keys():
    res['resComp'] = params['compareTrueParamsFunc'](dpmObjStd, res['std'])

  # print(res)

  return res

if __name__ == '__main__':
  main()


