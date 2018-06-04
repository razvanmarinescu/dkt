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

  def __init__(self):
    pass

  def generate(self, dataIndices, expName, params):
    return JointModel(dataIndices, expName, params)

class JointModel(DisProgBuilder.DPMInterface):

  def __init__(self, dataIndices, expName, params):
    self.dataIndices = dataIndices
    self.expName = expName
    self.params = params
    self.outFolder = params['outFolder']
    os.system('mkdir -p %s' % self.outFolder)
    self.params['plotTrajParams']['outFolder'] = self.outFolder
    self.params['plotTrajParams']['expName'] = expName
    self.nrBiomk = len(params['X'])
    self.nrFuncUnits = params['nrFuncUnits']
    self.biomkInFuncUnit = params['biomkInFuncUnit']

    self.unitModels = None # functional unit models
    self.disModels = None # disease specific models

    disLabels = self.params['disLabels']
    self.nrDis = len(disLabels)

    self.indxSubjForEachDisD = [np.in1d(self.params['plotTrajParams']['diag'],
      self.params['diagsSetInDis'][disNr]) for disNr in range(self.nrDis)]

    self.plotter = Plotter.PlotterJDM(self.params['plotTrajParams'])

    self.disIdxForEachSubjS = np.zeros(len(params['X'][0]), int)
    for d in range(self.nrDis):
      self.disIdxForEachSubjS[self.indxSubjForEachDisD[d]] = d

    self.ridsPerDisD = [_ for _ in range(self.nrDis)]
    for d in range(self.nrDis):
      self.ridsPerDisD[d] = self.params['RID'][self.indxSubjForEachDisD[d]]

  def runStd(self, runPart):
    self.run(runPart)

  def run(self, runPart):
    filePath = '%s/unitModels.npz' % self.outFolder
    nrRandFeatures = int(3)  # Number of random features for kernel approximation

    plotFigs = True

    if runPart[0] == 'R':
      self.initParams()

    nrIt = 10
    if runPart[1] == 'R':

      # if plotFigs:
      #   fig = self.plotter.plotCompWithTrueParams(self.unitModels, self.disModels, replaceFig=False)
      #   fig.savefig('%s/compTrueParams00_%s.png' % (self.outFolder, self.expName))

      for i in range(nrIt):
        # estimate biomk trajectories - disease agnostic
        self.estimBiomkTraj(self.unitModels, self.disModels)

        if plotFigs:
          fig = self.plotter.plotCompWithTrueParams(self.unitModels, self.disModels, replaceFig=False)
          fig.savefig('%s/compTrueParams%d1_%s.png' % (self.outFolder, i, self.expName))

        # estimate unit trajectories - disease specific
        for d in range(self.nrDis):
          self.disModels[d].estimUnitTraj()

        # estimate  subject latent variables
        self.estimSubjShifts()

    res = None
    return res

  def initParams(self):
    paramsCopy = copy.deepcopy(self.params)
    paramsCopy['nrGlobIterDis'] = 4 # set only two iterations, quick initialisation
    paramsCopy['nrGlobIterUnit'] = 4  # set only two iterations, quick initialisation
    paramsCopy['outFolder'] = '%s/init' % paramsCopy['outFolder']
    paramsCopy['penaltyUnits'] = 1
    onePassModel = JointModelOnePass.JDMOnePass(self.dataIndices, self.expName, paramsCopy)

    onePassModel.run(runPart = 'LL')

    self.unitModels = onePassModel.unitModels
    self.disModels = onePassModel.disModels

  def estimBiomkTraj(self, unitModels, disModels):

    # update the DPS scores of subjects
    predScoresDS = [0 for _ in range(self.nrDis)]

    XshiftedScaledDBS = [0 for _ in range(self.nrDis)]
    XdisDBSX = [0 for _ in range(self.nrDis)]
    for d in range(self.nrDis):
      XshiftedScaledDBS[d], XdisDBSX[d], _ = self.disModels[d].getData()

    XdisSX = [0 for _ in range(self.unitModels[0].nrSubj)]
    nrSubj = self.unitModels[0].nrSubj
    predScoresCurrUSX = [[0 for s in range(nrSubj)] for u in range(self.nrFuncUnits)]
    for s in range(nrSubj):
      currDis = self.disIdxForEachSubjS[s]
      currRID = self.params['RID'][s]
      # print('currDis', currDis)
      idxCurrSubjInDisModel = np.where(self.ridsPerDisD[currDis] == currRID)[0][0]
      # print('idxCurrSubjInDisModel', idxCurrSubjInDisModel)
      XdisSX[s] = XdisDBSX[currDis][0][idxCurrSubjInDisModel]

      # get shifts for curr subj from correct disModel
      currXdataShifted = XshiftedScaledDBS[currDis][0][idxCurrSubjInDisModel]
      # predict dysf scoresf for curr subj
      predScoresCurrXU = disModels[currDis].predictBiomk(currXdataShifted)

      for u in range(self.nrFuncUnits):
        predScoresCurrUSX[u][s] = predScoresCurrXU[:,u]

      assert currXdataShifted.shape[0] == XdisSX[s].shape[0]
      # print('currXdataShifted', currXdataShifted)
      # print('currRID', currRID)
      # print('self.ridsPerDisD', self.ridsPerDisD)

      # print('XshiftedScaledDBS[s] ', XshiftedScaledDBS[currDis][0])
      # print('predScoresCurrXU[0,:]', predScoresCurrXU[:,0])
      # print('XdisSX[s]', XdisSX[s])
      assert predScoresCurrXU[:,0].shape[0] == XdisSX[s].shape[0]
      assert predScoresCurrUSX[0][s].shape[0] == XdisSX[s].shape[0]

    # print(adsa)

    for u in range(self.nrFuncUnits):
      print('updating traj for func unit %d/%d' % (u+1, self.nrFuncUnits))
      # now update the X-values in each unitModel to the updated dysfunc scores
      # not that the X-vals are the same for every biomk within func units,
      # but initial missing values within each biomk are kept
      self.unitModels[u].updateXvals(predScoresCurrUSX[u], XdisSX)
      self.unitModels[u].Optimize_GP_parameters(Niterat = self.params['iterParamsUnit'])

      fig = self.unitModels[u].plotter.plotTraj(self.unitModels[u])
      fig2 = self.unitModels[u].plotter.plotCompWithTrueParams(self.unitModels[u], replaceFig=False)

      # fig.savefig('%s/allTraj%d0_%s.png' % (self.outFolder, i + 1, self.expName))


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


    labels = [self.params['labels'][b] for b in self.biomkInFuncUnit[nrCurrFuncUnit]]
    plotTrajParamsFuncUnit['labels'] = labels
    plotTrajParamsFuncUnit['colorsTraj'] =  [self.params['plotTrajParams']['colorsTrajBiomkB'][b]
      for b in self.biomkInFuncUnit[nrCurrFuncUnit]]

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

    plotTrajParamsDis['labels'] = self.params['plotTrajParams']['unitNames']
    plotTrajParamsDis['colorsTraj'] = plotTrajParamsDis['colorsTrajUnitsU']
    # if False, plot estimated traj. in separate plot from true traj. If True, use only one plot
    plotTrajParamsDis['allTrajOverlap'] = False

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








