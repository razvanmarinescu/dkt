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
import copy
import colorsys
import MarcoModel
import JointModelOnePass

import DisProgBuilder

class JMDBuilder(DisProgBuilder.DPMBuilder):
  # builds a Joint Disease model

  def __init__(self, plotTrajParams):
    self.plotterObj = Plotter.PlotterGP(plotTrajParams)

  def setPlotter(self, plotterObj):
    self.plotterObj = plotterObj

  def generate(self, dataIndices, expName, params):
    return JointModel(dataIndices, expName, params, self.plotterObj)

class JointModel(DisProgBuilder.DPMInterface):

  def __init__(self, dataIndices, expName, params, plotterObj):
    self.dataIndices = dataIndices
    self.expName = expName
    self.params = params
    self.outFolder = params['outFolder']
    os.system('mkdir -p %s' % self.outFolder)
    self.params['plotTrajParams']['outFolder'] = self.outFolder
    self.params['plotTrajParams']['expName'] = expName
    self.plotterObj = plotterObj
    self.nrBiomk = len(params['X'])
    self.nrFuncUnits = params['nrFuncUnits']
    self.biomkInFuncUnit = params['biomkInFuncUnit']

    self.unitModels = None # functional unit models
    self.disModels = None # disease specific models

    disLabels = self.params['disLabels']
    self.nrDis = len(disLabels)

    self.indxSubjForEachDisD = [np.in1d(self.params['plotTrajParams']['diag'],
      self.params['diagsSetInDis'][disNr]) for disNr in range(self.nrDis)]

    self.plotter = Plotter.PlotterJDM()


  def runStd(self, runPart):
    self.run(runPart)

  def run(self, runPart):
    filePath = '%s/unitModels.npz' % self.outFolder
    nrRandFeatures = int(3)  # Number of random features for kernel approximation

    plotFigs = True

    if runPart[0] == 'R':
      self.initParams()

    if runPart[1] == 'R':

      if plotFigs:
        fig = self.plotter.plotCompWithTrueParams(self)
        fig.savefig('%s/compTrueParams00_%s.png' % (self.outFolder, self.expName))

      for i in range(nrIt):
        # estimate biomk trajectories - disease agnostic
        for f in range(self.nrFuncUnits):
          self.estimBiomkTraj(unitModels[f])

        fig = self.plotter.plotCompWithTrueParams(self)
        fig.savefig('%s/compTrueParams%d1_%s.png' % (self.outFolder, i, self.expName))

        # estimate unit trajectories - disease specific
        for d in range(self.nrDis):
          self.disModels[d].estimUnitTraj()

        fig = self.plotter.plotCompWithTrueParams(self)
        fig.savefig('%s/compTrueParams%d2_%s.png' % (self.outFolder, i, self.expName))

        # estimate  subject latent variables
        self.estimSubjShifts()

        fig = self.plotter.plotCompWithTrueParams(self)
        fig.savefig('%s/compTrueParams%d3_%s.png' % (self.outFolder, i, self.expName))

    res = None
    return res

  def initParams(self):
    paramsCopy = copy.deepcopy(self.params)
    paramsCopy['nrGlobIterDis'] = 2 # set only two iterations, quick initialisation
    paramsCopy['nrGlobIterUnit'] = 2  # set only two iterations, quick initialisation
    paramsCopy['outFolder'] = '%s/init' % paramsCopy['outFolder']
    plotterObjOnePass = Plotter.PlotterGP(self.params['plotTrajParams'])
    onePassModel = JointModelOnePass.JDMOnePass(self.dataIndices, self.expName, paramsCopy,
      self.plotterObj)

    onePassModel.run(runPart = 'LR')

    self.unitModels = onePassModel.unitModels
    self.disModels = onePassModel.disModels

  def estimBiomkTraj(self, unitModel):

    self.Reset_parameters()

    objFuncGrad = lambda params: self.fullLik(params)

    self.Adadelta(Niterat, objFuncGrad, 0.05, self.parameters, output_grad_penalty=optimize_penalty)


  def predictBiomkSubjGivenXs(self, newXs, disNr):
    """
    predicts biomarkers for given xs (disease progression scores)

    :param newXs: newXs is an array as with np.linspace(minX-unscaled, maxX-unscaled)
    newXs will be scaled to the space of the gpProcess
    :param disNr: index of disease: 0 (tAD) or 1 (PCA)
    :return: biomkPredXB = Ys
    """

    # first predict the dysfunctionality scores in the disease specific model
    dysfuncPredXU = self.disModels[disNr].predictBiomk(newXs)


    # then predict the inidividual biomarkers in the disease agnostic models
    biomkPredXB = np.zeros((newXs.shape[0], self.nrBiomk))
    for u in range(self.nrFuncUnits):
      # dysfScaled = self.unitModels[u].applyScalingX(dysfuncPredXU[:,u])
      biomkPredXB[:, self.biomkInFuncUnit[u]] = \
        self.unitModels[u].predictBiomk(dysfuncPredXU[:,u])


    biomkIndNotInFuncUnits = self.biomkInFuncUnit[-1]
    # assumes these biomarkers are at the end

    nrBiomkNotInUnit = len(biomkIndNotInFuncUnits)
    biomkPredXB[:, biomkIndNotInFuncUnits] = \
      dysfuncPredXU[:,dysfuncPredXU.shape[1] - nrBiomkNotInUnit :]

    print('dysfuncPredXU[:,0]', dysfuncPredXU[:,0])
    print('biomkPredXB[:,0]', biomkPredXB[:,0])
    # print(asds)

    return biomkPredXB

  def sampleBiomkTrajGivenXs(self, newXs, disNr, nrSamples):
    """
    predicts biomarkers for given xs (disease progression scores)

    :param newXs: newXs is an array as with np.linspace(minX-unscaled, maxX-unscaled)
    newXs will be scaled to the space of the gpProcess
    :param disNr: index of disease: 0 (tAD) or 1 (PCA)
    :param nrSamples:

    :return: biomkPredXB = Ys
    """

    # first predict the dysfunctionality scores in the disease specific model
    dysfuncPredXU = self.disModels[disNr].predictBiomk(newXs)

    # then predict the inidividual biomarkers in the disease agnostic models
    trajSamplesBXS = np.nan * np.ones((self.nrBiomk, newXs.shape[0], nrSamples))

    for u in range(self.nrFuncUnits):
      biomkIndInCurrUnit = self.biomkInFuncUnit[u]
      for b in range(biomkIndInCurrUnit.shape[0]):
        trajSamplesBXS[biomkIndInCurrUnit[b],:,:] = \
            self.unitModels[u].sampleTrajPost(dysfuncPredXU[:,u], b, nrSamples)


    biomkIndNotInFuncUnits = self.biomkInFuncUnit[-1]
    nrBiomkNotInUnit = biomkIndNotInFuncUnits.shape[0]

    # assumes these biomarkers are at the end
    indOfRealBiomk =  list(range(dysfuncPredXU.shape[1] - nrBiomkNotInUnit, dysfuncPredXU.shape[1]))
    for b in range(len(biomkIndNotInFuncUnits)):
      trajSamplesBXS[biomkIndNotInFuncUnits[b],:,:] = \
        self.disModels[disNr].sampleTrajPost(newXs, indOfRealBiomk[b], nrSamples)

    assert not np.isnan(trajSamplesBXS).any()

    return trajSamplesBXS

  def createPlotTrajParamsFuncUnit(self, nrCurrFuncUnit):

    plotTrajParamsFuncUnit = copy.deepcopy(self.params['plotTrajParams'])
    plotTrajParamsFuncUnit['nrRows'] = self.params['plotTrajParams']['nrRowsFuncUnit']
    plotTrajParamsFuncUnit['nrCols'] = self.params['plotTrajParams']['nrColsFuncUnit']
    print('plotTrajParamsFuncUnit[nrRows]', plotTrajParamsFuncUnit['nrRows'])
    plotTrajParamsFuncUnit['unitNr'] = nrCurrFuncUnit  # some plotting functions need to know the current unit
    plotTrajParamsFuncUnit['isRunningFuncUnit'] = True


    if 'trueParams' in plotTrajParamsFuncUnit.keys():
        # set the params for plotting true trajectories - the Xs and f(Xs), i.e. trueTraj
      plotTrajParamsFuncUnit['trueParams']['trueXsTrajX'] = self.params['plotTrajParams']['trueParams']['trueDysfuncXsX']
      plotTrajParamsFuncUnit['trueParams']['trueTrajXB'] = \
        self.params['plotTrajParams']['trueParams']['trueTrajFromDysXB'][:, self.biomkInFuncUnit[nrCurrFuncUnit]]
      plotTrajParamsFuncUnit['trueParams']['subShiftsTrueMarcoFormatS'] = \
      plotTrajParamsFuncUnit['trueParams']['trueSubjDysfuncScoresSU'][:, nrCurrFuncUnit]

      # print(self.params['plotTrajParams']['trueParams']['trueTrajFromDysXB'].shape)
      # print(plotTrajParamsFuncUnit['trueParams']['trueTrajXB'].shape)
      # print(asd)



    labels = [self.params['labels'][b] for b in self.biomkInFuncUnit[nrCurrFuncUnit]]
    plotTrajParamsFuncUnit['labels'] = labels
    plotTrajParamsFuncUnit['colorsTraj'] =  [self.params['plotTrajParams']['colorsTraj'][b]
      for b in self.biomkInFuncUnit[nrCurrFuncUnit]]


    # print('plotTrajParamsFuncUnit', plotTrajParamsFuncUnit)
    # print(adsa)
    return plotTrajParamsFuncUnit

  def createPlotTrajParamsDis(self, disNr):

    plotTrajParamsDis = copy.deepcopy(self.params['plotTrajParams'])

    plotTrajParamsDis['diag'] = plotTrajParamsDis['diag'][self.indxSubjForEachDisD[disNr]]

    if 'trueParams' in plotTrajParamsDis.keys():
      # set the params for plotting true trajectories - the Xs and f(Xs), i.e. trueTraj
      plotTrajParamsDis['trueParams']['trueXsTrajX'] = \
        self.params['plotTrajParams']['trueParams']['trueLineSpacedDPSsX']
      plotTrajParamsDis['trueParams']['trueTrajXB'] = \
        self.params['plotTrajParams']['trueParams']['trueDysTrajFromDpsXU'][disNr]
      # need to filter out the subjects with other diseases
      plotTrajParamsDis['trueParams']['subShiftsTrueMarcoFormatS'] = \
        plotTrajParamsDis['trueParams']['subShiftsTrueMarcoFormatS'][self.indxSubjForEachDisD[disNr]]

      # print(plotTrajParamsDis['trueParams']['trueTrajXB'].shape)
      # print(adsa)

    print(self.params['nrBiomkDisModel'])
    # print(adssa)

    plotTrajParamsDis['labels'] = self.params['plotTrajParams']['unitNames']
    plotTrajParamsDis['colorsTraj'] = [colorsys.hsv_to_rgb(hue, 1, 1) for hue in
      np.linspace(0, 1, num = self.params['nrBiomkDisModel'], endpoint = False)]
    # if False, plot estimated traj. in separate plot from true traj. If True, use only one plot
    plotTrajParamsDis['allTrajOverlap'] = False

    # plotTrajParamsDis['yNormMode'] = None

    # print('plotTrajParamsFuncUnit', plotTrajParamsFuncUnit)
    # print(adsa)
    return plotTrajParamsDis


  def plotTrajectories(self, res):
    pass
    # fig = self.plotterObj.plotTraj(self.gpModel)
    # fig.savefig('%s/allTrajFinal.png' % self.outFolder)

  def stageSubjects(self, indices):
    pass

  def stageSubjectsData(self, data):
    pass

  def plotTrajSummary(self, res):
    pass








