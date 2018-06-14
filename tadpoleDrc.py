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
# plotTrajParams['ylimitsRandPoints'] = (-3,2)
plotTrajParams['blenderPath'] = blenderPath
# plotTrajParams['isSynth'] = False


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

    fig = pl.figure(1)
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
    prepareData(finalDataFile, tinyData)

  ds = pickle.load(open(finalDataFile, 'rb'))
  dataDfAll = ds['dataDfAll']
  regParamsICV = ds['regParamsICV']
  regParamsAge = ds['regParamsAge']
  regParamsGender = ds['regParamsGender']
  regParamsDataset = ds['regParamsDataset']
  X = ds['X']
  Y = ds['Y']
  RID = np.array(ds['RID'])
  labels = ds['list_biomarkers']
  diag = ds['diag']

  visDataHist(dataDfAll)
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

  # nrBiomk = nrBiomkInFuncUnits * nrFuncUnits
  # print(labels)
  # print(adss)
  # mapBiomkToFuncUnits = np.array(list(range(nrFuncUnits)) * nrBiomkInFuncUnits)
  # should give smth like [0,1,2,3,0,1,2,3,0,1,2,3]

  # change the order of the functional units so that the hippocampus and occipital are fitted first
  unitPermutation = [5,3,2,1,4,0]
  mapBiomkToFuncUnits = np.array((unitPermutation * nrBiomkInFuncUnits) + [-1,-1,-1])
  unitNames = [l.split(' ')[-1] for l in labels]
  unitNames = [unitNames[i] for i in unitPermutation]
  nrBiomk = mapBiomkToFuncUnits.shape[0]

  biomkInFuncUnit = [0 for u in range(nrFuncUnits+1)]
  for u in range(nrFuncUnits):
    biomkInFuncUnit[u] = np.where(mapBiomkToFuncUnits == u)[0]
    # biomkInFuncUnit[u] += [nrBiomk-3, nrBiomk-2] # also add CDRSOB and ADAS in order hlp disentangle the trajectories and get better staging

  # add extra entry with other biomks to be added in the disease models
  biomkInFuncUnit[nrFuncUnits] = np.array([nrBiomk-3, nrBiomk-2, nrBiomk-1])


  plotTrajParams['biomkInFuncUnit'] = biomkInFuncUnit
  plotTrajParams['labels'] = labels
  plotTrajParams['nrRowsFuncUnit'] = 3
  plotTrajParams['nrColsFuncUnit'] = 4
  plotTrajParams['colorsTraj'] = [colorsys.hsv_to_rgb(hue, 1, 1) for hue in np.linspace(0, 1, num=nrBiomk, endpoint=False)]

  plotTrajParams['yNormMode'] = 'zScoreTraj'
  # plotTrajParams['yNormMode'] = 'zScoreEarlyStageTraj'
  # plotTrajParams['yNormMode'] = 'unscaled'

  # if False, plot estimated traj. in separate plot from true traj.
  plotTrajParams['allTrajOverlap'] = False

  params['runIndex'] = args.runIndex
  params['nrProc'] = args.nrProc
  params['cluster'] = args.cluster
  params['plotTrajParams'] = plotTrajParams
  params['penaltyUnits'] = args.penalty
  params['penaltyDis'] = args.penalty
  params['nrFuncUnits'] = nrFuncUnits
  params['biomkInFuncUnit'] = biomkInFuncUnit
  params['labels'] = labels

  params['X'] = X
  params['Y'] = Y
  params['RID'] = RID
  params['diag'] = diag
  params['plotTrajParams']['diag'] = params['diag']
  params['Xvalid'] = ds['Xvalid']
  params['Yvalid'] = ds['Yvalid']
  params['RIDvalid'] = ds['RIDvalid']
  params['diagValid'] = ds['diagValid']
  params['dataDfAll'] = dataDfAll

  params['nrGlobIterUnit'] = 10 # these parameters are specific for the Joint Model of Disease (JMD)
  params['iterParamsUnit'] = 60
  params['nrGlobIterDis'] = 10
  params['iterParamsDis'] = 60

  # nrSubj = len(ds['Yvalid'][0])
  # for b in range(6,12):
  #   print(b, labels[b], [ds['Yvalid'][b][s][0] for s in range(nrSubj) if (ds['Yvalid'][b][s] and ds['diag'][s]
  #   == CTL)])
  #   print('mean CTL', np.mean([ds['Yvalid'][b][s][0] for s in range(nrSubj) if (ds['Yvalid'][b][s] and ds['diag'][s]
  #   == CTL)]))
    # print('mean AD', np.mean(ds['Yvalid'][b][ds['diag'] == AD]))

  # print(ads)


  nrBiomkDisModel = nrFuncUnits + 3
  params['nrBiomkDisModel'] = nrBiomkDisModel

  nrXPoints = 50
  nrDis = 2 # nr of diseases
  params['trueParams'] = {}
  params['trueParams']['subShiftsTrueMarcoFormatS'] = np.zeros(RID.shape[0])
  params['trueParams']['trueSubjDysfuncScoresSU'] = np.zeros((RID.shape[0],nrFuncUnits))
  params['trueParams']['trueDysfuncXsX'] = np.linspace(0,1, nrXPoints)
  params['trueParams']['trueTrajXB'] = np.zeros((nrXPoints, nrBiomk))
  params['trueParams']['trueTrajFromDysXB'] = np.zeros((nrXPoints, nrBiomk))

  params['trueParams']['trueLineSpacedDPSsX'] = np.linspace(-10,10, nrXPoints)
  # params['trueParams']['trueTrajPredXB'] = np.zeros((nrXPoints,nrBiomk))
  params['trueParams']['trueDysTrajFromDpsXU'] = [np.zeros((nrXPoints,nrBiomkDisModel)) for d in range(nrDis)]

  params['plotTrajParams']['trueParams'] = params['trueParams']
  params['plotTrajParams']['unitNames'] = unitNames + labels[-3:]

  # map which diagnoses belong to which disease
  # first disease has CTL+AD, second disease has CTL2+PCA
  params['diagsSetInDis'] = [np.array([CTL, MCI, AD]), np.array([CTL2, PCA])]
  params['disLabels'] = ['tAD', 'PCA']
  params['otherBiomkPerDisease'] = [[nrBiomk-3,nrBiomk-2, nrBiomk-1], []]

  print('diag', params['diag'].shape[0])
  print('X[0]',len(params['X'][0]))
  assert params['diag'].shape[0] == len(params['X'][0])
  # assert params['diag'].shape[0] == len(params['trueParams']['subShiftsTrueMarcoFormatS'])
  # assert params['diag'].shape[0] == len(params['trueParams']['trueSubjDysfuncScoresSU'])

  if np.abs(args.penalty - int(args.penalty) < 0.00001):
    expName = '%sPen%d' % (expName, args.penalty)
  else:
    expName = '%sPen%.1f' % (expName, args.penalty)

  # params['runPartStd'] = ['L', 'L']
  params['runPartStd'] = args.runPartStd
  params['runPartMain'] = ['R', 'I', 'I'] # [mainPart, plot, stage]
  params['masterProcess'] = args.runIndex == 0

  expNameDisOne = '%s' % expName
  modelNames, res = evaluationFramework.runModels(params, expName,
   args.modelToRun, runAllExpTadpoleDrc)




def runAllExpTadpoleDrc(params, expName, dpmBuilder, compareTrueParamsFunc = None):
  """ runs all experiments"""

  res = {}

  dpmBuilder.plotterObj.plotTrajParams = params['plotTrajParams']

  params['patientID'] = AD
  params['excludeID'] = -1
  params['excludeXvalidID'] = -1
  params['excludeStaging'] = [-1]

  params['outFolder'] = 'resfiles/%s' % expName
  params['expName'] = expName

  dpmObjStd, res['std'] = evaluationFramework.runStdDPM(params,
    expName, dpmBuilder, params['runPartMain'])

  dpmObjStd.plotter.plotAllBiomkDisSpace(dpmObjStd, params, disNr=0)

  # perform the validation against DRC data
  validateDRCBiomk(dpmObjStd, params)


  return res



if __name__ == '__main__':
  main()


