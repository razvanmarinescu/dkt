import os
import sys
import numpy as np
import math
from auxFunc import *
import scipy
#import scipy.cluster.hierarchy
import pickle
import gc
from matplotlib import pyplot as pl
import Plotter

import MarcoModel

import DisProgBuilder

class MarcoModelBuilder(DisProgBuilder.DPMBuilder):
  # builds a voxel-wise disease progression model

  def __init__(self, plotTrajParams):
    self.plotterObj = Plotter.PlotterGP(plotTrajParams)

  def setPlotter(self, plotterObj):
    self.plotterObj = plotterObj

  def generate(self, dataIndices, expName, params):
    return MarcoModelWrapper(dataIndices, expName, params, self.plotterObj)

class MarcoModelWrapper(DisProgBuilder.DPMInterface):

  def __init__(self, dataIndices, expName, params, plotterObj):
    self.dataIndices = dataIndices
    self.expName = expName
    self.params = params
    self.outFolder = params['outFolder']
    os.system('mkdir -p %s' % self.outFolder)
    self.params['plotTrajParams']['outFolder'] = self.outFolder
    self.plotterObj = plotterObj

    Xfilt, Yfilt = filterDataListFormat(params, dataIndices)

    #changed to 2 for testing!!
    N = int(10)  # Number of random features for kernel approximation
    self.gpModel = MarcoModel.GP_progression_model(Xfilt,Yfilt, N, self.outFolder, plotterObj,
      self.params['priors'],  self.params['labels'])

    # set penalty for decreasing trajectories
    self.gpModel.Set_penalty(self.params['penalty'])

  def runStd(self, runPart):
    self.run(runPart)

  def run(self, runPart):
    filePath = '%s/fittedGPModel.npz' % self.outFolder
    if runPart[0] == 'R':
      N_global_iterations = 10
      iterParams = 80
      iterShifts = 50
      self.gpModel.Optimize(N_global_iterations, iterParams, Plot=True)
      pickle.dump(self.gpModel, open(filePath, 'wb'), protocol = pickle.HIGHEST_PROTOCOL)
    else:
      self.gpModel = pickle.load(open(filePath, 'rb'))

    res = None
    return res

  def plotTrajectories(self, res):
    # fig = self.plotterObj.plotTraj(self.gpModel)
    # fig.savefig('%s/allTrajFinal.png' % self.outFolder)

    fig = self.plotterObj.plotCompWithTrueParams(self.gpModel, replaceFig=True)
    fig.savefig('%s/compTrueFinal.png' % self.outFolder)

  def stageSubjects(self, indices):
    pass

  def stageSubjectsData(self, data):
    pass

  def plotTrajSummary(self, res):
    pass








