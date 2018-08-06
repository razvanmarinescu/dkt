import os
import sys
import numpy as np
import math
from auxFunc import *
import scipy
import pickle
import gc
from matplotlib import pyplot as pl
import copy

import Plotter
import SigmoidModel
import DisProgBuilder

class SigmoidModelBuilder(DisProgBuilder.DPMBuilder):

  def __init__(self, plotTrajParams):
    self.plotterObj = Plotter.PlotterFuncUnit(plotTrajParams)

  def generate(self, dataIndices, expName, params):
    return SigmoidModelWrapper(dataIndices, expName, params, self.plotterObj)


class SigmoidModelWrapper(DisProgBuilder.DPMInterface):

  def __init__(self, dataIndices, expName, params, plotterObj):
    self.dataIndices = dataIndices
    self.expName = expName
    self.params = params
    self.outFolder = params['outFolder']
    os.system('mkdir -p %s' % self.outFolder)
    self.params['plotTrajParams']['outFolder'] = self.outFolder
    self.params['plotTrajParams']['expName'] = expName
    self.plotterObj = plotterObj

    self.plotterObj.plotTrajParams['title'] = 'Biomarker traj.'
    self.plotterObj.plotTrajParams['colorsTraj'] = params['plotTrajParams']['colorsTrajBiomkB']


    Xfilt, Yfilt, visitIndicesFilt = filterDataListFormat(params, dataIndices)

    self.model = SigmoidModel.SigmoidModel(Xfilt, Yfilt, visitIndicesFilt,
                                           self.outFolder, plotterObj, self.params['labels'], self.params)

    self.nrBiomk = len(Xfilt)

    self.nrDis = len(self.params['disLabels'])
    # boolean masks
    self.binMaskSubjForEachDisD = [np.in1d(self.params['plotTrajParams']['diag'],
      self.params['diagsSetInDis'][disNr]) for disNr in range(self.nrDis)]

  def runStd(self, runPart):
    self.run(runPart)

  def run(self, runPart):
    filePath = '%s/fittedGPModel.npz' % self.outFolder
    if runPart[0] == 'R':
      N_global_iterations = 5
      self.model.Optimize(N_global_iterations, Plot=True)
      pickle.dump(self.model, open(filePath, 'wb'), protocol = pickle.HIGHEST_PROTOCOL)
    else:
      self.model = pickle.load(open(filePath, 'rb'))

    res = None
    return res

  def plotTrajectories(self, res):
    # fig = self.plotterObj.plotCompWithTrueParams(self.model, replaceFig=True)
    # fig.savefig('%s/compTrueFinal.png' % self.outFolder)
    pass

  def stageSubjects(self, indices):
    pass

  def stageSubjectsData(self, data):
    pass

  def plotTrajSummary(self, res):
    pass

  def predictBiomkSubjGivenXs(self, newXs, disNr):
    return self.model.predictBiomk(newXs)

  def sampleBiomkTrajGivenXs(self, newXs, disNr, nrSamples):
    trajSamplesBXS = np.nan * np.ones((self.nrBiomk, newXs.shape[0], nrSamples))

    for b in range(self.nrBiomk):
      trajSamplesBXS[b,:,:] = self.model.sampleTrajPost(newXs,b,nrSamples)


  def getIndxSubjToKeep(self, disNr):
    return np.where(dpmObj.binMaskSubjForEachDisD[disNr])[0]



