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
plotTrajParams['diagColors'] = {CTL:'g', MCI:'y', AD:'r',
  CTL2:'g', PCA:'y', AD2:'r'}
plotTrajParams['diagScatterMarkers'] = {CTL:'o', MCI:'o', AD:'o',
  CTL2:'x', PCA:'x', AD2:'x'}
plotTrajParams['legendCols'] = 4
plotTrajParams['diagLabels'] = {CTL:'CTL', AD:'AD', PCA:'PCA', CTL2:'CTL2'}
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

  sigmaGfunc = GPModel.genSigmaG
  sigmaEpsfunc = None
  sigmaSfunc  = None

  outFolder = 'resfiles/synth/'

  expName = 'synth1'
  fileName = '%s.npz' % expName

  regenerateData = args.regData

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
  # thetas[:, 1] = 10
  thetas[:, 3] = 0
  for f in range(nrFuncUnits):
    thetas[mapBiomkToFuncUnits == f, 2] = np.linspace(0.2, 0.9, num=nrBiomkInFuncUnits, endpoint=True)

  # set first funtional unit to have traj with lower slopes
  thetas[mapBiomkToFuncUnits == 0, 1] = 5
  thetas[mapBiomkToFuncUnits == 1, 1] = 10

  sigmaB = 0.05 * np.ones(nrBiomk)

  ##### disease 1 - disease specific parameters ###########

  # params of the dysfunctional trajectories
  dysfuncParamsDisOne = np.zeros((nrFuncUnits, 4), float)
  dysfuncParamsDisOne[:, 0] = 1  # ak
  dysfuncParamsDisOne[:, 1] = 0.3  # bk
  dysfuncParamsDisOne[:, 2] = [-3, 7]  # ck
  dysfuncParamsDisOne[:, 3] = 0  # dk

  synthModelDisOne = ParHierModel.ParHierModel(dysfuncParamsDisOne, thetas,
    mapBiomkToFuncUnits, sigmoidFunc, sigmaB)

  paramsDisOne = copy.deepcopy(params)

  paramsDisOne = genSynthData.generateDataJMD(nrSubjLong, nrBiomk, nrTimepts,
  shiftsLowerLim, shiftsUpperLim, synthModelDisOne, outFolder, fileName,
    regenerateData, paramsDisOne, ctlDiagNr=CTL, patDiagNr=AD)

  paramsDisOne['plotTrajParams']['trueParams'] = paramsDisOne['trueParams']

  if regenerateData:
    synthPlotter = Plotter.PlotterJDM(paramsDisOne['plotTrajParams'])
    fig = synthPlotter.plotTrajDataMarcoFormat(paramsDisOne['X'], paramsDisOne['Y'],
      paramsDisOne['diag'], paramsDisOne['trueParams']['subShiftsTrueMarcoFormatS'],
      synthModelDisOne, replaceFigMode=True)
    fig.savefig('%s/synth1Dis1GenData.png' % outFolder)

  ##### disease 2 - disease specific parameters ###########

  # params of the dysfunctional trajectories
  dysfuncParamsDisTwo = copy.deepcopy(dysfuncParamsDisOne)
  dysfuncParamsDisTwo[:, 1] = 1
  dysfuncParamsDisTwo[:, 2] = [8, -4]

  synthModelDisTwo = ParHierModel.ParHierModel(dysfuncParamsDisTwo, thetas, mapBiomkToFuncUnits, sigmoidFunc, sigmaB)

  paramsDisTwo = copy.deepcopy(paramsDisOne)
  nrSubjLongDisTwo = 50
  nrTimeptsDisTwo = 4

  paramsDisTwo = genSynthData.generateDataJMD(nrSubjLongDisTwo, nrBiomk,
    nrTimeptsDisTwo, shiftsLowerLim, shiftsUpperLim, synthModelDisTwo,
    outFolder, fileName, regenerateData, paramsDisTwo, ctlDiagNr=CTL2,
    patDiagNr=PCA)

  # for disease two, only keep the second biomarker in each functional unit
  indBiomkInDiseaseTwo = np.array(range(nrFuncUnits,(2*nrFuncUnits)))
  print('indBiomkInDiseaseTwo', indBiomkInDiseaseTwo)
  paramsDisTwo['Xtrue'] = paramsDisTwo['X']
  paramsDisTwo['Ytrue'] = paramsDisTwo['Y']



  # for disease two, change the format of the X and Y arrays, add the missing biomarkers with empty lists
  XemptyListsAllBiomk = [0 for _ in range(nrBiomk)]
  YemptyListsAllBiomk = [0 for _ in range(nrBiomk)]
  for b in range(nrBiomk):
    XemptyListsAllBiomk[b] = [0 for _ in range(nrSubjLongDisTwo)]
    YemptyListsAllBiomk[b] = [0 for _ in range(nrSubjLongDisTwo)]

    for s in range(nrSubjLongDisTwo):
      if b in indBiomkInDiseaseTwo:
        XemptyListsAllBiomk[b][s] = paramsDisTwo['Xtrue'][b][s]
        YemptyListsAllBiomk[b][s] = paramsDisTwo['Ytrue'][b][s]
      else:
        XemptyListsAllBiomk[b][s] = []
        YemptyListsAllBiomk[b][s] = []

  paramsDisTwo['XemptyListsAllBiomk'] = XemptyListsAllBiomk
  paramsDisTwo['YemptyListsAllBiomk'] = YemptyListsAllBiomk

  paramsDisTwo['plotTrajParams']['trueParams'] = paramsDisTwo['trueParams']

  if regenerateData:
    synthPlotter = Plotter.PlotterJDM(paramsDisTwo['plotTrajParams'])
    fig = synthPlotter.plotTrajDataMarcoFormat(paramsDisTwo['Xtrue'],
      paramsDisTwo['Ytrue'], paramsDisTwo['diag'], paramsDisTwo['trueParams']['subShiftsTrueMarcoFormatS'],
      synthModelDisTwo, replaceFigMode=True)
    fig.savefig('%s/synth1Dis2GenDataFull.png' % outFolder)

    synthPlotter = Plotter.PlotterJDM(paramsDisTwo['plotTrajParams'])
    fig = synthPlotter.plotTrajDataMarcoFormat(paramsDisTwo['XemptyListsAllBiomk'],
      paramsDisTwo['YemptyListsAllBiomk'], paramsDisTwo['diag'], paramsDisTwo['trueParams']['subShiftsTrueMarcoFormatS'],
      synthModelDisTwo, replaceFigMode=True)
    fig.savefig('%s/synth1Dis2GenDataMissing.png' % outFolder)

    # pl.pause(100)

  ############### now merge the two datasets ############

  # add the biomarkers from the second dataset, same format as dataset 1
  # but with missing entries
  params = paramsDisOne
  for b in range(nrBiomk):
    params['X'][b] += paramsDisTwo['XemptyListsAllBiomk'][b]
    params['Y'][b] += paramsDisTwo['YemptyListsAllBiomk'][b]

  print(params['RID'].shape)
  print(np.array(paramsDisTwo['RID']))
  print(list(nrSubjLong + np.array(paramsDisTwo['RID'])))
  params['RID'] = np.concatenate((params['RID'],
  nrSubjLong + paramsDisTwo['RID']),axis=0) # RIDs must be different

  print('paramsDisOne[diag]', paramsDisOne['diag'])
  print(paramsDisTwo['diag'])
  params['diag'] = np.concatenate((paramsDisOne['diag'], paramsDisTwo['diag']),axis=0)

  params['trueParams']['subShiftsTrueMarcoFormatS'] = np.array(
    list(params['trueParams']['subShiftsTrueMarcoFormatS']) +
    list(paramsDisTwo['trueParams']['subShiftsTrueMarcoFormatS']))
  params['trueParams']['trueSubjDysfuncScoresSU'] = np.concatenate(
    (params['trueParams']['trueSubjDysfuncScoresSU'],
    paramsDisTwo['trueParams']['trueSubjDysfuncScoresSU']),axis=0)

  params['trueParams']['trueTrajPredXB'] = \
    [paramsDisOne['trueParams']['trueTrajPredXB'], paramsDisTwo['trueParams']['trueTrajPredXB']]
  params['trueParams']['trueDysTrajFromDpsXU'] = \
    [paramsDisOne['trueParams']['trueDysTrajFromDpsXU'], paramsDisTwo['trueParams']['trueDysTrajFromDpsXU']]

  params['plotTrajParams']['diag'] = params['diag']

  # map which diagnoses belong to which disease
  # first disease has CTL+AD, second disease has CTL2+PCA
  params['diagsSetInDis'] = [np.array([CTL, AD]), np.array([CTL2, PCA])]
  params['disLabels'] = ['dis0', 'dis1']


  print('diag', params['diag'].shape[0])
  print('X[0]',len(params['X'][0]))
  assert params['diag'].shape[0] == len(params['X'][0])
  assert params['diag'].shape[0] == len(params['trueParams']['subShiftsTrueMarcoFormatS'])
  assert params['diag'].shape[0] == len(params['trueParams']['trueSubjDysfuncScoresSU'])

  if np.abs(args.penalty - int(args.penalty) < 0.00001):
    expName = '%sPen%d' % (expName, args.penalty)
  else:
    expName = '%sPen%.1f' % (expName, args.penalty)

  params['runPartStd'] = args.runPartStd
  params['runPartMain'] = ['R', 'I', 'I'] # [mainPart, plot, stage]
  params['masterProcess'] = args.runIndex == 0

  expNameDisOne = '%s' % expName
  modelNames, res = evaluationFramework.runModels(params, expName,
    args.modelToRun, runAllExpSynth)



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


