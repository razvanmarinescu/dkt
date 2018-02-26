from sklearn.metrics import *
from sklearn.model_selection import *
import math
import pickle

import os.path
from auxFunc import *
import sys
from matplotlib import pyplot as pl
import gc



# Differential equation models
from DisProgBuilder import *
#from aligners import *


# joint disease models
import JointModel
import MarcoModelWrapper
import Plotter
import IncompleteModel

from importlib.machinery import SourceFileLoader
from sklearn import linear_model
plotFuncDiffEq = SourceFileLoader("*",
  os.path.abspath("../diffEqModel/plotFunc.py")).load_module()

def runModels(params, expName, modelToRun, runAllExpFunc):
  modelNames = []
  res = []

  if np.any(modelToRun == 0) or np.any(modelToRun == 14):
    # JMD - Joint Model of Diseases
    dpmBuilder = JointModel.JMDBuilder(params['plotTrajParams'])
    modelName = 'JMD'
    expNameCurrModel = '%s_%s' % (expName, modelName)
    params['currModel'] = 14
    res += [runAllExpFunc(params, expNameCurrModel, dpmBuilder)]
    modelNames += [modelName]

  if np.any(modelToRun == 0) or np.any(modelToRun == 15):
    # Marco's Model
    dpmBuilder = MarcoModelWrapper.MarcoModelBuilder(params['plotTrajParams'])
    modelName = 'MarcoModel'
    expNameCurrModel = '%s_%s' % (expName, modelName)
    params['currModel'] = 15
    res += [runAllExpFunc(params, expNameCurrModel, dpmBuilder)]
    modelNames += [modelName]

  if np.any(modelToRun == 0) or np.any(modelToRun == 16):
    # Incomplete JMD - Joint Model of Diseases
    filePathUnitModels = params['filePathUnitModels']
    gpModels = pickle.load(open(filePathUnitModels, 'rb'))
    dpmBuilder = IncompleteModel.IncompleteBuilder(params['plotTrajParams'], gpModels)
    modelName = 'IJDM'
    expNameCurrModel = '%s_%s' % (expName, modelName)
    params['currModel'] = 16
    res += [runAllExpFunc(params, expNameCurrModel, dpmBuilder)]
    modelNames += [modelName]

  return modelNames, res

def runStdDPM(params, expNameCurrModel, dpmBuilder, runPart):
  dataIndices = np.logical_not(np.in1d(params['diag'], params['excludeXvalidID']))
  print(np.sum(np.logical_not(dataIndices)))
  print('excludeID', params['excludeXvalidID'])
  print(params['diag'].shape)
  dpmObj = dpmBuilder.generate(dataIndices, expNameCurrModel, params)
  res = None
  if runPart[0] == 'R':
    res = dpmObj.runStd(params['runPartStd'])

  if runPart[1] == 'R':
    dpmObj.plotTrajectories(res)

  if runPart[2] == 'R':
    # dataIndicesNN = np.logical_and(dataIndices, np.sum(np.isnan(params['data']),1) == 0)
    (maxLikStages, maxStagesIndex, stagingProb, stagingLik, tsStages, _) = dpmObj.stageSubjects(dataIndices)
    print(params['diag'].shape, dataIndices.shape)
    print('maxLikStages min max', np.min(maxLikStages), np.max(maxLikStages))
    fig, lgd = plotFuncDiffEq.plotStagingHist(maxLikStages, diag=params['diag'][dataIndices],
                    plotTrajParams=params['plotTrajParams'], expNameCurrModel=expNameCurrModel)
    stagingHistFigName = '%s/stagingHist.png' % dpmObj.outFolder
    fig.savefig(stagingHistFigName, bbox_extra_artists=(lgd,), bbox_inches='tight')


  return dpmObj, res


