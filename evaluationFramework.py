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
import Plotter
import SigmoidModel
import SigmoidWrapper
import LinearModel

from importlib.machinery import SourceFileLoader
from sklearn import linear_model

def runModels(params, expName, modelToRun, runAllExpFunc):
  modelNames = []
  res = []

  if np.any(modelToRun == 0) or np.any(modelToRun == 14):
    # JMD - Joint Model of Diseases
    unitModelObjList = [SigmoidModel.SigmoidModel for b in range(params['nrFuncUnitsImgOnly'])] + \
                       [LinearModel.LinearModel for b in range(params['nrExtraBiomk'])]
    disModelObj = SigmoidModel.SigmoidModel
    dpmBuilder = JointModel.JMDBuilder(unitModelObjList,
      disModelObj, params['priorsUnitModels'],
      params['priorsDisModels'])
    modelName = 'JMD'
    expNameCurrModel = '%s_%s' % (expName, modelName)
    params['currModel'] = 14
    res += [runAllExpFunc(params, expNameCurrModel, dpmBuilder)]
    modelNames += [modelName]

  if np.any(modelToRun == 16):
    # Joint Model of Diseases - One Pass
    unitModelObjList = [SigmoidModel.SigmoidModel for b in range(params['nrFuncUnitsImgOnly'])] + \
                       [LinearModel.LinearModel for b in range(params['nrExtraBiomk'])]
    disModelObj = SigmoidModel.SigmoidModel
    dpmBuilder = JointModelOnePass.JMDBuilderOnePass(unitModelObjList,
      disModelObj, params['priorsUnitModelsMarcoModel'],
      params['priorsDisModels'])
    modelName = 'JDMOnePass'
    expNameCurrModel = '%s_%s' % (expName, modelName)
    params['currModel'] = 16
    res += [runAllExpFunc(params, expNameCurrModel, dpmBuilder)]
    modelNames += [modelName]

  if np.any(modelToRun == 0) or np.any(modelToRun == 17):
    # Latent Space - Sigmoid Model (Jedynak, 2012, Neuroimage)
    dpmBuilder = SigmoidWrapper.SigmoidModelBuilder(params['plotTrajParams'])
    modelName = 'Sig'
    expNameCurrModel = '%s_%s' % (expName, modelName)
    params['currModel'] = 17
    params['priors'] = params['priorsNoDKTSigmoid']
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

  
  return dpmObj, res


