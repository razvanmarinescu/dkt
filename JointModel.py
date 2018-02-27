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
    self.plotterObj = Plotter.PlotterJDM(plotTrajParams)

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
    self.mapBiomkToFuncUnits = params['mapBiomkToFuncUnits']

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
    if runPart[0] == 'R':
      nrGlobIter = 10
      iterParams = 80
      N = int(3)  # Number of random features for kernel approximation
      Xfilt, Yfilt = filterDataListFormat(self.params, self.dataIndices)

      # print(adsa)

      self.unitModels = [_ for _ in range(self.nrFuncUnits)]

      # functional units
      for u in range(self.nrFuncUnits):
        plotTrajParamsFuncUnit = self.createPlotTrajParamsFuncUnit(nrCurrFuncUnit=u)
        plotterObjCurrFuncUnit = Plotter.PlotterGP(plotTrajParamsFuncUnit)  # set separate plotter for the

        XfiltCurrUnit = [Xfilt[b] for b in range(self.nrBiomk) if self.mapBiomkToFuncUnits[b] == u]
        YfiltCurrUnit = [Yfilt[b] for b in range(self.nrBiomk) if self.mapBiomkToFuncUnits[b] == u]
        outFolderCurrUnit = '%s/unit%d' % (self.outFolder, u)
        os.system('mkdir -p %s' % outFolderCurrUnit)
        self.unitModels[u] = MarcoModel.GP_progression_model(XfiltCurrUnit, YfiltCurrUnit, N, outFolderCurrUnit,
                                                             plotterObjCurrFuncUnit, plotTrajParamsFuncUnit['labels'])
        print('penalty', self.params['penalty'])
        # print(adsa)
        self.unitModels[u].Set_penalty(self.params['penalty'])
        self.unitModels[u].Optimize(nrGlobIter, iterParams, Plot=True)

      pickle.dump(self.unitModels, open(filePath, 'wb'), protocol = pickle.HIGHEST_PROTOCOL)
    else:
      self.unitModels = pickle.load(open(filePath, 'rb'))


    disModelsFile = '%s/disModels.npz' % self.outFolder
    nrSubj = self.unitModels[0].N_samples

    if runPart[1] == 'R':
      nrGlobIter = 10
      iterParams = 80
      dysfuncScoresU = [0 for x in range(self.nrFuncUnits)]
      xDysfunSubjU = [0 for x in range(self.nrFuncUnits)]
      for u in range(self.nrFuncUnits):
        xs = np.linspace(self.unitModels[u].minX, self.unitModels[u].maxX, num=100).reshape([100, 1])

        dysfuncScoresU[u] = [[] for _ in range(nrSubj)]
        xDysfunSubjU[u] = [[] for _ in range(nrSubj)]
        xsNewGpTest = [[] for _ in range(nrSubj)]

        for sub in range(self.unitModels[u].N_samples):
          for b in range(self.unitModels[u].N_biom):
            xDysfunSubjUCurrSubj = self.unitModels[u].X[b][sub]  # Xs in the unit model
            xDysfunSubjU[u][sub] += list(xDysfunSubjUCurrSubj)

            dysfuncScoresCurrSubExtr = [self.unitModels[u].X_array[b][k][0] for k in range(int(np.sum(
              self.unitModels[u].N_obs_per_sub[b][:sub])), np.sum(self.unitModels[u].N_obs_per_sub[b][:sub + 1]))]

            dysfuncScoresU[u][sub] += dysfuncScoresCurrSubExtr # (Xs + timeShift) in the unit model

            # xsNewGpTestCurrSub = [newGPTest.X_array[b][k][0] for k in range(int(np.sum(
            #   newGPTest.N_obs_per_sub[b][:sub])), np.sum(newGPTest.N_obs_per_sub[b][:sub + 1]))]



            #
            # print('dysfuncScoresUCurrSubCalc', dysfuncScoresCurrSubCalc)
            # print('xsNewGpTestCurrSub', xsNewGpTestCurrSub, np.array(xsNewGpTestCurrSub) + self.unitModels[u].params_time_shift[0][sub])
            # print('dysfuncScoresCurrSubExtr', dysfuncScoresCurrSubExtr)
            # print('params_time_shift[0][sub]',
            #       self.unitModels[u].params_time_shift[0][sub])
            # print('xDysfunSubjU[u][sub]', xDysfunSubjU[u][sub])
            # print(adsa) they are indeed equal if you standardize them.

          # apply the forward scaling transform


          print('xDysfunSubjU[u][sub]', xDysfunSubjU[u][sub])
          print('dysfuncScoresU[u][sub]', dysfuncScoresU[u][sub])

          xDysfunSubjU[u][sub] = np.sort(np.unique(xDysfunSubjU[u][sub]))
          dysfuncScoresU[u][sub] = np.sort(np.unique(dysfuncScoresU[u][sub]))
          # dysfuncScoresU[u][sub] = np.array(xDysfunSubjU[u][sub]) + self.unitModels[u].params_time_shift[0][sub]


          # print('diag Sub', self.params['diag'][sub])
          print('xDysfunSubjU[u][sub]', xDysfunSubjU[u][sub])
          print('dysfuncScoresU[u][sub]', dysfuncScoresU[u][sub])
          print('X[b][sub]', [self.unitModels[u].X[b][sub] for b in range(self.unitModels[u].N_biom)])
          print('-------------------------------------')
          assert len(dysfuncScoresU[u][sub]) == len(xDysfunSubjU[u][sub])
          # print('dysfuncScoresU[u][sub]', dysfuncScoresU[u][sub])

        # print(adsa)

        dysfuncScoresSerial = [x2 for x in dysfuncScoresU[u] for x2 in x]
        minDys = np.min(dysfuncScoresSerial)
        maxDys = np.max(dysfuncScoresSerial)
        # print('dysfuncScoresSerial', dysfuncScoresSerial)
        # print('minDys', minDys)
        # print('maxDys', maxDys)

        # make the functional scores be between [0,1]
        # 26/02/18: actually this is not needed, re-scaling will be done in the plotting
        # dysfuncScoresU[u] = [ (x - minDys) / (maxDys - minDys) for x in  dysfuncScoresU[u]]
        # print('dysfuncScoresU[u]', dysfuncScoresU[u])
        # print(adsa)


      # add any other biomarkers that were not part of any functiona units, e.g. cog tests
      # print(len([self.params['X'][i] for i in np.where(self.mapBiomkToFuncUnits == -1)[0]]))
      # print(len(xDysfunSubjU))

      # print(len(xDysfunSubjU))
      # print(asda)
      nrBiomkDisModel = len(xDysfunSubjU)

      # now build separate model for each disease
      disLabels = self.params['disLabels']
      nrDis = len(disLabels)
      self.disModels = [_ for _ in range(nrDis)]

      for disNr in range(nrDis):

        xDysfunSubjUCopy = copy.deepcopy(xDysfunSubjU)
        dysfuncScoresUCopy = copy.deepcopy(dysfuncScoresU)

        xDysfunSubjUCopy += [self.params['X'][i] for i in self.params['otherBiomkPerDisease'][disNr]]
        dysfuncScoresUCopy += [self.params['Y'][i] for i in self.params['otherBiomkPerDisease'][disNr]]

        # first filter the data .. keep only subj in current disease
        xDysfunSubjCurrDisU = [_ for _ in range(nrBiomkDisModel)]
        dysfuncScoresCurrDisU = [_ for _ in range(nrBiomkDisModel)]

        for b in range(nrBiomkDisModel):
          print('b=', b)
          print(len(xDysfunSubjU[b]), xDysfunSubjU[b][0])
          print('nrSubj', nrSubj)
          print(self.params['diagsSetInDis'][disNr])
          print(self.params['diag'].shape)
          xDysfunSubjCurrDisU[b] = [xDysfunSubjUCopy[b][s] for s in
            np.where(self.indxSubjForEachDisD[disNr])[0]]
          dysfuncScoresCurrDisU[b] = [dysfuncScoresUCopy[b][s] for s in
            np.where(self.indxSubjForEachDisD[disNr])[0]]

        # print('xDysfunSubjCurrDisU', xDysfunSubjCurrDisU)
        # print('dysfuncScoresCurrDisU', dysfuncScoresCurrDisU)
        # print(asds)
        plotTrajParamsDis = self.createPlotTrajParamsDis(disNr)
        plotterCurrDis = Plotter.PlotterGP(plotTrajParamsDis)  # set separate plotter for the

        # print(plotTrajParamsDis['trueParams']['subShiftsTrueMarcoFormatS'].shape[0])
        # print(len(xDysfunSubjCurrDisU[0]))
        # print(ad)

        print('xDysfunSubjCurrDisU[0]', xDysfunSubjCurrDisU[0][:5])
        print('dysfuncScoresCurrDisU[0]', dysfuncScoresCurrDisU[0][:5])
        print('xDysfunSubjCurrDisU[1]', xDysfunSubjCurrDisU[1][:5])
        print('dysfuncScoresCurrDisU[1]', dysfuncScoresCurrDisU[1][:5])

        print('dysfuncScoresU[:][:5]', [xDysfunSubjCurrDisU[u][:5] for u in range(nrBiomkDisModel)])
        print('dysfuncScoresU[:][:5]', [dysfuncScoresCurrDisU[u][:5] for u in range(nrBiomkDisModel)])
        # print(ads)

        outFolderCurDis = '%s/%s' % (self.outFolder, self.params['disLabels'][disNr])
        os.system('mkdir -p %s' % outFolderCurDis)
        self.disModels[disNr] = MarcoModel.GP_progression_model(xDysfunSubjCurrDisU,
          dysfuncScoresCurrDisU, nrGlobIter, outFolderCurDis, plotterCurrDis, plotTrajParamsDis['labels'])

        print('X', [x[0] for x in self.disModels[disNr].X[0]])
        # print(asda)

        self.disModels[disNr].Set_penalty(self.params['penalty'])
        self.disModels[disNr].Optimize(nrGlobIter, iterParams, Plot=True)

        pickle.dump(self.disModels, open(disModelsFile, 'wb'), protocol = pickle.HIGHEST_PROTOCOL)

    elif runPart[1] == 'L':
      self.disModels = pickle.load(open(disModelsFile, 'rb'))


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
    dysfuncPredXU = self.disModels[disNr].predictBiomkAndScale(newXs)


    # then predict the inidividual biomarkers in the disease agnostic models
    biomkPredXB = np.zeros((newXs.shape[0], self.nrBiomk))
    for u in range(self.nrFuncUnits):
      biomkPredXB[:, self.mapBiomkToFuncUnits == u] = \
        self.unitModels[u].predictBiomkAndScale(dysfuncPredXU[:,u])


    biomkIndNotInFuncUnits = np.where(self.mapBiomkToFuncUnits == -1)[0]
    # assumes these biomarkers are at the end

    nrBiomkNotInUnit = biomkIndNotInFuncUnits.shape[0]
    biomkPredXB[:, biomkIndNotInFuncUnits] = \
      dysfuncPredXU[:,dysfuncPredXU.shape[1] - nrBiomkNotInUnit :]

    return biomkPredXB

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
        self.params['plotTrajParams']['trueParams']['trueTrajFromDysXB'][:, self.mapBiomkToFuncUnits == nrCurrFuncUnit]
      plotTrajParamsFuncUnit['trueParams']['subShiftsTrueMarcoFormatS'] = \
      plotTrajParamsFuncUnit['trueParams']['trueSubjDysfuncScoresSU'][:, nrCurrFuncUnit]

      # print(self.params['plotTrajParams']['trueParams']['trueTrajFromDysXB'].shape)
      # print(plotTrajParamsFuncUnit['trueParams']['trueTrajXB'].shape)
      # print(asd)



    labels = [self.params['labels'][b] for b in range(len(self.params['labels']))
      if self.mapBiomkToFuncUnits[b] == nrCurrFuncUnit]
    plotTrajParamsFuncUnit['labels'] = labels
    plotTrajParamsFuncUnit['colorsTraj'] =  [self.params['plotTrajParams']['colorsTraj'][b] for b in range(len(self.params['labels']))
      if self.mapBiomkToFuncUnits[b] == nrCurrFuncUnit]


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


    plotTrajParamsDis['labels'] = self.params['plotTrajParams']['unitNames']
    plotTrajParamsDis['colorsTraj'] = [colorsys.hsv_to_rgb(hue, 1, 1) for hue in
      np.linspace(0, 1, num = self.params['nrBiomkDisModel'], endpoint = False)]
    # if False, plot estimated traj. in separate plot from true traj. If True, use only one plot
    plotTrajParamsDis['allTrajOverlap'] = False

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








