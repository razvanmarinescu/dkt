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

import DisProgBuilder

class JMDBuilderOnePass(DisProgBuilder.DPMBuilder):
  # builds a Joint Disease model

  def __init__(self):
    pass

  def generate(self, dataIndices, expName, params):
    return JDMOnePass(dataIndices, expName, params)

class JDMOnePass(DisProgBuilder.DPMInterface):

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

    self.indxSubjForEachDisD = params['indxSubjForEachDisD']

    self.unitModelObj = params['unitModelObj']
    self.disModelObj = params['disModelObj']


  def runStd(self, runPart):
    self.run(runPart)

  def run(self, runPart):
    filePath = '%s/unitModels.npz' % self.outFolder

    if runPart[0] == 'R':
      nrGlobIterUnit = self.params['nrGlobIterUnit']
      iterParamsUnit = self.params['iterParamsUnit']

      Xfilt, Yfilt, visitIndicesFilt = filterDataListFormat(self.params, self.dataIndices)

      self.unitModels = [_ for _ in range(self.nrFuncUnits)]

      # functional units
      for u in range(self.nrFuncUnits):
        plotTrajParamsFuncUnit = JDMOnePass.createPlotTrajParamsFuncUnit(self.params, unitNr=u)
        plotterObjCurrFuncUnit = Plotter.PlotterFuncUnit(plotTrajParamsFuncUnit)  # set separate plotter for the

        XfiltCurrUnit = [Xfilt[b] for b in self.biomkInFuncUnit[u]]
        YfiltCurrUnit = [Yfilt[b] for b in self.biomkInFuncUnit[u]]
        visitIndicesCurrUnit = [visitIndicesFilt[b] for b in self.biomkInFuncUnit[u]]
        outFolderCurrUnit = '%s/unit%d' % (self.outFolder, u)
        os.system('mkdir -p %s' % outFolderCurrUnit)
        self.unitModels[u] = self.unitModelObj(XfiltCurrUnit, YfiltCurrUnit, visitIndicesCurrUnit, outFolderCurrUnit,
          plotterObjCurrFuncUnit, plotTrajParamsFuncUnit['labels'], self.params)

        self.unitModels[u].Set_penalty(self.params['penaltyUnits'])
        self.unitModels[u].Optimize(nrGlobIterUnit, iterParamsUnit, Plot=True)

      pickle.dump(self.unitModels, open(filePath, 'wb'), protocol = pickle.HIGHEST_PROTOCOL)
    else:
      self.unitModels = pickle.load(open(filePath, 'rb'))

      # make sure the data you used has not been changed since fitting this model
      for b in range(len(self.unitModels[0].X)):
        # print('--------------- b', b)
        idxInAllBiomk = self.biomkInFuncUnit[0][b]
        for s in range(len(self.unitModels[0].X[b])):
          assert np.all(self.unitModels[0].X[b][s] == self.params['X'][idxInAllBiomk][s])
          assert np.all(self.unitModels[0].Y[b][s] == self.params['Y'][idxInAllBiomk][s])

      for u in range(self.nrFuncUnits):
        plotTrajParamsFuncUnit = JDMOnePass.createPlotTrajParamsFuncUnit(self.params, unitNr=u)
        plotterObjCurrFuncUnit = Plotter.PlotterFuncUnit(plotTrajParamsFuncUnit)  # set separate plotter for the


    disModelsFile = '%s/disModels.npz' % self.outFolder
    nrSubj = self.unitModels[0].nrSubj

    if runPart[1] == 'R':
      nrGlobIterDis = self.params['nrGlobIterDis']
      iterParamsDis = self.params['iterParamsDis']
      dysfuncScoresU = [0 for x in range(self.nrFuncUnits)]
      xDysfunSubjU = [0 for x in range(self.nrFuncUnits)]

      minDys = np.zeros(self.nrFuncUnits)
      maxDys = np.zeros(self.nrFuncUnits)

      for u in range(self.nrFuncUnits):
        dysfuncScoresU[u] = [[] for _ in range(nrSubj)]
        xDysfunSubjU[u] = [[] for _ in range(nrSubj)]

        XshiftedUnitModel, XunitModel, YunitModel, _ = self.unitModels[u].getData()

        for sub in range(self.unitModels[u].nrSubj):
          for b in range(self.unitModels[u].nrBiomk):
            xDysfunSubjUCurrSubj = XunitModel[b][sub]  # Xs in the unit model
            xDysfunSubjU[u][sub] += list(xDysfunSubjUCurrSubj)
            dysfuncScoresU[u][sub] += list(XshiftedUnitModel[b][sub]) # (Xs + timeShift) in the unit model

          xDysfunSubjU[u][sub] = np.sort(np.unique(xDysfunSubjU[u][sub]))
          dysfuncScoresU[u][sub] = np.sort(np.unique(dysfuncScoresU[u][sub]))

          assert len(dysfuncScoresU[u][sub]) == len(xDysfunSubjU[u][sub])

        dysfuncScoresSerial = [x2 for x in dysfuncScoresU[u] for x2 in x]
        minDys[u] = np.min(dysfuncScoresSerial)
        maxDys[u] = np.max(dysfuncScoresSerial)

        # make the functional scores be between [0,1]
        # 26/02/18: actually this is not needed, re-scaling will be done in the plotting
        # 2 June 2018: actually I need this, otherwise the plotting of unit-traj in dis space
        # will have wrong Y-scale
        dysfuncScoresU[u] = [self.unitModels[u].applyScalingXzeroOneFwd(xs) for xs in dysfuncScoresU[u]]


      # now build separate model for each disease
      disLabels = self.params['disLabels']
      nrDis = len(disLabels)
      self.disModels = [_ for _ in range(nrDis)]


      for disNr in range(nrDis):
        nrBiomkDisModel = len(xDysfunSubjU) + len(self.params['otherBiomkPerDisease'][disNr])

        xDysfunSubjUCopy = copy.deepcopy(xDysfunSubjU)
        dysfuncScoresUCopy = copy.deepcopy(dysfuncScoresU)

        if 'otherBiomkPerDisease' in self.params.keys():
          xDysfunSubjUCopy += [self.params['X'][i] for i in self.params['otherBiomkPerDisease'][disNr]]
          dysfuncScoresUCopy += [self.params['Y'][i] for i in self.params['otherBiomkPerDisease'][disNr]]


        # first filter the data .. keep only subj in current disease
        xDysfunSubjCurrDisU = [_ for _ in range(nrBiomkDisModel)]
        dysfuncScoresCurrDisU = [_ for _ in range(nrBiomkDisModel)]

        for b in range(nrBiomkDisModel):
          xDysfunSubjCurrDisU[b] = [xDysfunSubjUCopy[b][s] for s in
            np.where(self.indxSubjForEachDisD[disNr])[0]]
          dysfuncScoresCurrDisU[b] = [dysfuncScoresUCopy[b][s] for s in
            np.where(self.indxSubjForEachDisD[disNr])[0]]

          for s in range(len(xDysfunSubjCurrDisU[b])):
            visitIndicesCurrDis[b][s] = np.array(range(xDysfunSubjCurrDisU[b][s].shape[0]))

        plotTrajParamsDis = JDMOnePass.createPlotTrajParamsDis(self.params, disNr)
        plotterCurrDis = Plotter.PlotterDis(plotTrajParamsDis)  # set separate plotter for the

        outFolderCurDis = '%s/%s' % (self.outFolder, self.params['disLabels'][disNr])
        os.system('mkdir -p %s' % outFolderCurDis)
        self.disModels[disNr] = self.disModelObj(xDysfunSubjCurrDisU, dysfuncScoresCurrDisU, visitIndicesCurrDis,
          outFolderCurDis, plotterCurrDis, plotTrajParamsDis['labels'], self.params)
        self.disModels[disNr].Optimize(nrGlobIterDis, Plot=True)

        pickle.dump(self.disModels, open(disModelsFile, 'wb'), protocol = pickle.HIGHEST_PROTOCOL)

    elif runPart[1] == 'L':
      self.disModels = pickle.load(open(disModelsFile, 'rb'))

      for disNr in range(self.nrDis):
        plotTrajParamsDis = JDMOnePass.createPlotTrajParamsDis(self.params, disNr)
        plotterCurrDis = Plotter.PlotterDis(plotTrajParamsDis)  # set separate plotter for the

        # fig = plotterCurrDis.plotTrajSameSpace(self.disModels[disNr])
        # fig.savefig('%s/%s_trajSameSpace_%s.png' % (self.outFolder, self.params['disLabels'][disNr],
        #   self.expName))

    res = None
    return res

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
      dysfScaled = self.unitModels[u].applyScalingXzeroOneInv(dysfuncPredXU[:,u])

      biomkPredXB[:, self.biomkInFuncUnit[u]] = \
        self.unitModels[u].predictBiomk(dysfScaled)


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

  @staticmethod
  def createPlotTrajParamsFuncUnit(params, unitNr):

    plotTrajParamsFuncUnit = copy.deepcopy(params['plotTrajParams'])
    plotTrajParamsFuncUnit['nrRows'] = params['plotTrajParams']['nrRowsFuncUnit']
    plotTrajParamsFuncUnit['nrCols'] = params['plotTrajParams']['nrColsFuncUnit']

    if 'trueParamsFuncUnits' in params.keys():
        # set the params for plotting true trajectories - the Xs and f(Xs), i.e. trueTraj
      plotTrajParamsFuncUnit['trueParams'] = params['trueParamsFuncUnits'][unitNr]

    labels = [params['labels'][b] for b in params['biomkInFuncUnit'][unitNr]]
    plotTrajParamsFuncUnit['labels'] = labels
    plotTrajParamsFuncUnit['colorsTraj'] =  [params['plotTrajParams']['colorsTrajBiomkB'][b]
                                             for b in params['biomkInFuncUnit'][unitNr]]
    plotTrajParamsFuncUnit['title'] = params['unitNames'][unitNr]

    return plotTrajParamsFuncUnit

  @staticmethod
  def createPlotTrajParamsDis(params, disNr):

    plotTrajParamsDis = copy.deepcopy(params['plotTrajParams'])

    plotTrajParamsDis['diag'] = plotTrajParamsDis['diag'][params['indxSubjForEachDisD'][disNr]]

    if 'trueParamsDis' in params.keys():
      plotTrajParamsDis['trueParams'] = params['trueParamsDis'][disNr]


    plotTrajParamsDis['labels'] = params['unitNames']
    plotTrajParamsDis['colorsTraj'] =  plotTrajParamsDis['colorsTrajUnitsU']
    # if False, plot estimated traj. in separate plot from true traj. If True, use only one plot
    plotTrajParamsDis['allTrajOverlap'] = False
    plotTrajParamsDis['title'] = params['disLabels'][disNr]


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








