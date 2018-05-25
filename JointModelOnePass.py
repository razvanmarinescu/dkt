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


  def runStd(self, runPart):
    self.run(runPart)

  def run(self, runPart):
    filePath = '%s/unitModels.npz' % self.outFolder
    nrRandFeatures = int(3)  # Number of random features for kernel approximation
    if runPart[0] == 'R':
      nrGlobIterUnit = self.params['nrGlobIterUnit']
      iterParamsUnit = self.params['iterParamsUnit']

      Xfilt, Yfilt = filterDataListFormat(self.params, self.dataIndices)

      # print(adsa)

      self.unitModels = [_ for _ in range(self.nrFuncUnits)]

      # functional units
      for u in range(self.nrFuncUnits):
        plotTrajParamsFuncUnit = self.createPlotTrajParamsFuncUnit(nrCurrFuncUnit=u)
        plotterObjCurrFuncUnit = Plotter.PlotterGP(plotTrajParamsFuncUnit)  # set separate plotter for the

        XfiltCurrUnit = [Xfilt[b] for b in self.biomkInFuncUnit[u]]
        YfiltCurrUnit = [Yfilt[b] for b in self.biomkInFuncUnit[u]]
        outFolderCurrUnit = '%s/unit%d' % (self.outFolder, u)
        os.system('mkdir -p %s' % outFolderCurrUnit)
        self.unitModels[u] = MarcoModel.GP_progression_model(XfiltCurrUnit, YfiltCurrUnit, nrRandFeatures, outFolderCurrUnit,
                                                             plotterObjCurrFuncUnit, plotTrajParamsFuncUnit['labels'])

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
          # print(self.unitModels[0].X[b][s])
          # print(self.params['X'][idxInAllBiomk][s])
          assert np.all(self.unitModels[0].X[b][s] == self.params['X'][idxInAllBiomk][s])
          assert np.all(self.unitModels[0].Y[b][s] == self.params['Y'][idxInAllBiomk][s])

      for u in range(self.nrFuncUnits):
        self.unitModels[u].minScX = self.unitModels[u].applyScalingX(self.unitModels[u].minX)
        self.unitModels[u].maxScX = self.unitModels[u].applyScalingX(self.unitModels[u].maxX)

        plotTrajParamsFuncUnit = self.createPlotTrajParamsFuncUnit(nrCurrFuncUnit=u)
        plotterObjCurrFuncUnit = Plotter.PlotterGP(plotTrajParamsFuncUnit)  # set separate plotter for the

        # fig = plotterObjCurrFuncUnit.plotTraj(self.unitModels[u], replaceFig=True, legendExtraPlot=True)
        # fig.savefig('%s/unit%s_allTraj_%s.png' % (self.outFolder, u, self.expName))



      # print(asda)

    disModelsFile = '%s/disModels.npz' % self.outFolder
    nrSubj = self.unitModels[0].N_samples

    if runPart[1] == 'R':
      nrGlobIterDis = self.params['nrGlobIterDis']
      iterParamsDis = self.params['iterParamsDis']
      dysfuncScoresU = [0 for x in range(self.nrFuncUnits)]
      xDysfunSubjU = [0 for x in range(self.nrFuncUnits)]
      for u in range(self.nrFuncUnits):
        dysfuncScoresU[u] = [[] for _ in range(nrSubj)]
        xDysfunSubjU[u] = [[] for _ in range(nrSubj)]

        XshiftedUnitModel, XunitModel, YunitModel = self.unitModels[u].getData()

        for sub in range(self.unitModels[u].N_samples):
          for b in range(self.unitModels[u].N_biom):
            xDysfunSubjUCurrSubj = XunitModel[b][sub]  # Xs in the unit model
            xDysfunSubjU[u][sub] += list(xDysfunSubjUCurrSubj)

            # dysfuncScoresCurrSubExtr = [self.unitModels[u].X_array[b][k][0] for k in range(int(np.sum(
            #   self.unitModels[u].N_obs_per_sub[b][:sub])), np.sum(self.unitModels[u].N_obs_per_sub[b][:sub + 1]))]

            dysfuncScoresU[u][sub] += list(XshiftedUnitModel[b][sub]) # (Xs + timeShift) in the unit model

          xDysfunSubjU[u][sub] = np.sort(np.unique(xDysfunSubjU[u][sub]))
          dysfuncScoresU[u][sub] = np.sort(np.unique(dysfuncScoresU[u][sub]))


          assert len(dysfuncScoresU[u][sub]) == len(xDysfunSubjU[u][sub])


        dysfuncScoresSerial = [x2 for x in dysfuncScoresU[u] for x2 in x]
        minDys = np.min(dysfuncScoresSerial)
        maxDys = np.max(dysfuncScoresSerial)

        # make the functional scores be between [0,1]
        # 26/02/18: actually this is not needed, re-scaling will be done in the plotting
        # dysfuncScoresU[u] = [ (x - minDys) / (maxDys - minDys) for x in  dysfuncScoresU[u]]



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
          # print(self.params['otherBiomkPerDisease'][disNr])
          # print(asd)

        # first filter the data .. keep only subj in current disease
        xDysfunSubjCurrDisU = [_ for _ in range(nrBiomkDisModel)]
        dysfuncScoresCurrDisU = [_ for _ in range(nrBiomkDisModel)]

        for b in range(nrBiomkDisModel):
          xDysfunSubjCurrDisU[b] = [xDysfunSubjUCopy[b][s] for s in
            np.where(self.indxSubjForEachDisD[disNr])[0]]
          dysfuncScoresCurrDisU[b] = [dysfuncScoresUCopy[b][s] for s in
            np.where(self.indxSubjForEachDisD[disNr])[0]]

        plotTrajParamsDis = self.createPlotTrajParamsDis(disNr)
        plotterCurrDis = Plotter.PlotterGP(plotTrajParamsDis)  # set separate plotter for the


        outFolderCurDis = '%s/%s' % (self.outFolder, self.params['disLabels'][disNr])
        os.system('mkdir -p %s' % outFolderCurDis)
        self.disModels[disNr] = MarcoModel.GP_progression_model(xDysfunSubjCurrDisU,
          dysfuncScoresCurrDisU, nrRandFeatures, outFolderCurDis, plotterCurrDis, plotTrajParamsDis['labels'])

        print('X', [x[0] for x in self.disModels[disNr].X[0]])
        # print(asda)

        self.disModels[disNr].Set_penalty(self.params['penaltyDis'])
        self.disModels[disNr].Optimize(nrGlobIterDis, iterParamsDis, Plot=True)

        pickle.dump(self.disModels, open(disModelsFile, 'wb'), protocol = pickle.HIGHEST_PROTOCOL)

    elif runPart[1] == 'L':
      self.disModels = pickle.load(open(disModelsFile, 'rb'))

      for disNr in range(self.nrDis):
        self.disModels[disNr].minScX = self.disModels[disNr].applyScalingX(self.disModels[disNr].minX)
        self.disModels[disNr].maxScX = self.disModels[disNr].applyScalingX(self.disModels[disNr].maxX)

        plotTrajParamsDis = self.createPlotTrajParamsDis(disNr)
        plotterCurrDis = Plotter.PlotterGP(plotTrajParamsDis)  # set separate plotter for the

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








