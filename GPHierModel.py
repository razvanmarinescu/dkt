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

class GPHierBuilder(DisProgBuilder.DPMBuilder):
  # builds a voxel-wise disease progression model

  def __init__(self, plotTrajParams):
    self.plotterObj = Plotter.PlotterJDM(plotTrajParams)

  def setPlotter(self, plotterObj):
    self.plotterObj = plotterObj

  def generate(self, dataIndices, expName, params):
    return JointModel(dataIndices, expName, params, self.plotterObj)

class GPHierModel(DisProgBuilder.DPMInterface):

  def __init__(self, dataIndices, expName, params, plotterObj):
    self.dataIndices = dataIndices
    self.expName = expName
    self.params = params
    self.outFolder = params['outFolder']
    os.system('mkdir -p %s' % self.outFolder)
    self.params['plotTrajParams']['outFolder'] = self.outFolder
    self.plotterObj = plotterObj
    self.nrBiomk = len(params['X'])
    self.nrFuncUnits = params['nrFuncUnits']
    self.mapBiomkToFuncUnits = params['mapBiomkToFuncUnits']

    self.gpModels = None
    self.grandModel = None


  def runStd(self, runPart):
    self.run(runPart)

  def run(self, runPart):
    filePath = '%s/unitModels.npz' % self.outFolder
    if runPart[0] == 'R':
      N_global_iterations = 50
      iterParams = 50
      iterShifts = 30
      N = int(10)  # Number of random features for kernel approximation
      Xfilt, Yfilt = filterDataListFormat(self.params, self.dataIndices)

      # print(adsa)

      self.gpModels = [_ for _ in range(self.nrFuncUnits)]

      # functional units
      for u in range(self.nrFuncUnits):
        plotTrajParamsFuncUnit = self.createPlotTrajParamsFuncUnit(nrCurrFuncUnit=u)
        plotterObjCurrFuncUnit = Plotter.PlotterGP(plotTrajParamsFuncUnit)  # set separate plotter for the

        XfiltCurrUnit = [Xfilt[b] for b in range(self.nrBiomk) if self.mapBiomkToFuncUnits[b] == u]
        YfiltCurrUnit = [Yfilt[b] for b in range(self.nrBiomk) if self.mapBiomkToFuncUnits[b] == u]
        outFolderCurrUnit = '%s/unit%d' % (self.outFolder, u)
        os.system('mkdir -p %s' % outFolderCurrUnit)
        self.gpModels[u] = MarcoModel.GP_progression_model(XfiltCurrUnit, YfiltCurrUnit, N, outFolderCurrUnit,
           plotterObjCurrFuncUnit, plotTrajParamsFuncUnit['labels'])
        self.gpModels[u].Set_penalty(self.params['penalty'])
        self.gpModels[u].Optimize(N_global_iterations, [iterParams, iterShifts], Plot=True)

      pickle.dump(self.gpModels, open(filePath, 'wb'), protocol = pickle.HIGHEST_PROTOCOL)
    else:
      self.gpModels = pickle.load(open(filePath, 'rb'))
      # for u in range(self.nrFuncUnits):
      #   plotTrajParamsFuncUnit = self.createPlotTrajParamsFuncUnit(nrCurrFuncUnit = u)
      #   plotterObjCurrFuncUnit = Plotter.PlotterGP(plotTrajParamsFuncUnit)  # set separate plotter for the
      #   fig = plotterObjCurrFuncUnit.plotCompWithTrueParams(self.gpModels[u], replaceFig=True)
      #   outFolderCurrUnit = '%s/unit%d' % (self.outFolder, u)
      #   fig.savefig('%s/compTrueFinal.png' % outFolderCurrUnit)

    grandModelFile = '%s/grandModel.npz' % self.outFolder
    if runPart[1] == 'R':
      dysfuncScoresU = [0 for x in range(self.nrFuncUnits)]
      xDysfunSubjU = [0 for x in range(self.nrFuncUnits)]
      for u in range(self.nrFuncUnits):
        xs = np.linspace(self.gpModels[u].minX, self.gpModels[u].maxX, num=100).reshape([100, 1])

        # _, meanStagesS = self.gpModels[u].StageSubjects(self.gpModels[u].X,self.gpModels[u].Y, xs)


        nrSubj = self.gpModels[u].N_samples
        dysfuncScoresU[u] = [[] for _ in range(nrSubj)]
        xDysfunSubjU[u] = [[] for _ in range(nrSubj)]
        for sub in range(self.gpModels[u].N_samples):
          for b in range(self.gpModels[u].N_biom):
            dysfuncScoresU[u][sub] += [self.gpModels[u].X_array[b][k][0] for k in range(int(np.sum(
              self.gpModels[u].N_obs_per_sub[b][:sub])), np.sum(self.gpModels[u].N_obs_per_sub[b][:sub + 1]))]
            xDysfunSubjU[u][sub] += [self.gpModels[u].X[b][sub]]

            # dysfuncScoresCurrSubCalc = self.params['X'][b][sub] + meanStagesS[sub]
            # dysfuncScoresCurrSubExtr = [self.gpModels[u].X_array[b][k][0] for k in range(int(np.sum(self.gpModels[
            # u].N_obs_per_sub[b][:sub])), np.sum(self.gpModels[u].N_obs_per_sub[b][:sub + 1]))]
            #
            # print('dysfuncScoresUCurrSubCalc', dysfuncScoresCurrSubCalc)
            # print('dysfuncScoresCurrSubExtr', dysfuncScoresCurrSubExtr)
            # print(adsa) they are indeed equal if you standardize them.

          dysfuncScoresU[u][sub] = np.unique(dysfuncScoresU[u][sub])
          xDysfunSubjU[u][sub] = np.unique(xDysfunSubjU[u][sub])
          assert len(dysfuncScoresU[u][sub]) == len(xDysfunSubjU[u][sub])
          print('xDysfunSubjU[u][sub]', xDysfunSubjU[u][sub])
          # print('dysfuncScoresU[u][sub]', dysfuncScoresU[u][sub])

        dysfuncScoresSerial = [x2 for x in dysfuncScoresU[u] for x2 in x]
        minDys = np.min(dysfuncScoresSerial)
        maxDys = np.max(dysfuncScoresSerial)
        print('dysfuncScoresSerial', dysfuncScoresSerial)
        print('minDys', minDys)
        print('maxDys', maxDys)

        # make the functional scores be between [0,1]
        dysfuncScoresU[u] = [ (x - minDys) / (maxDys - minDys) for x in  dysfuncScoresU[u]]
        print('dysfuncScoresU[u]', dysfuncScoresU[u])
        # print(adsa)

        # nrBiomkCurrUnit = len(self.gpModels[u].X)
        # for sub in range(self.gpModels[u].N_samples):
        #   assert dysfuncScoresU[u][sub].shape[0] == self.params['X'][u][sub].shape[0]
        #
        #
        #   listOfXpointsCurrSubj = [x for b in range(nrBiomkCurrUnit) for x in list(self.gpModels[u].X[b][sub])]
        #   xsTmp = np.unique(listOfXpointsCurrSubj)
        #   xDysfunSubjU[u] += xsTmp
        #
        #   yDysfunSubj[u] += dysfuncScoresU[u][sub]

      plotTrajParamsGrand = copy.deepcopy(self.params['plotTrajParams'])
      plotTrajParamsGrand['trueParams']['trueTrajPredXB'] = \
        self.params['plotTrajParams']['trueParams']['trueDysfuncPredXU']
      potentialLayout = [(2,2), (2,3), (3,3), (3,4), (4,4), (4,5), (4,6), (5,6)]

      optimalRows, optimalCols = ([potentialLayout[i] for i in range(len(potentialLayout))
        if potentialLayout[i][0] * potentialLayout[i][1] > (self.nrFuncUnits + 2)])[0]
      plotTrajParamsGrand['nrRows'] = optimalRows
      plotTrajParamsGrand['nrCols'] = optimalCols
      plotTrajParamsGrand['colorsTraj'] = [colorsys.hsv_to_rgb(hue, 1, 1) for hue in
        np.linspace(0, 1, num=self.nrFuncUnits, endpoint=False)]

      # if False, plot estimated traj. in separate plot from true traj. If True, use only one plot
      plotTrajParamsGrand['allTrajOverlap'] = True
      labelsGrand = ['unit %d' % u for u in range(self.nrFuncUnits)]
      print('labelsGrand', labelsGrand)
      # print(asdas)
      plotTrajParamsGrand['labels'] = labelsGrand
      plotterGrand = Plotter.PlotterGP(plotTrajParamsGrand)  # set separate plotter for the

      N_global_iterations = 50
      iterParams = 50
      iterShifts = 30
      self.grandModel = MarcoModel.GP_progression_model(xDysfunSubjU, dysfuncScoresU, N_global_iterations,
        self.outFolder, plotterGrand, self.params['labels'])
      self.grandModel.Set_penalty(self.params['penalty'])
      self.grandModel.Optimize(N_global_iterations, [iterParams, iterShifts], Plot=True)

      pickle.dump(self.grandModel, open(grandModelFile, 'wb'), protocol = pickle.HIGHEST_PROTOCOL)

    elif runPart[1] == 'L':
      self.grandModel = pickle.load(open(grandModelFile, 'rb'))

    res = None
    return res

  def createPlotTrajParamsFuncUnit(self, nrCurrFuncUnit):

    plotTrajParamsFuncUnit = copy.deepcopy(self.params['plotTrajParams'])
    plotTrajParamsFuncUnit['nrRows'] = self.params['plotTrajParams']['nrRowsFuncUnit']
    plotTrajParamsFuncUnit['nrCols'] = self.params['plotTrajParams']['nrColsFuncUnit']
    print('plotTrajParamsFuncUnit[nrRows]', plotTrajParamsFuncUnit['nrRows'])
    plotTrajParamsFuncUnit['unitNr'] = nrCurrFuncUnit  # some plotting functions need to know the current unit
    plotTrajParamsFuncUnit['isRunningFuncUnit'] = True
    plotTrajParamsFuncUnit['trueParams']['trueTrajPredXB'] = \
    self.params['plotTrajParams']['trueParams']['trueTrajPredXB'][:, self.mapBiomkToFuncUnits == nrCurrFuncUnit]
    plotTrajParamsFuncUnit['trueParams']['subShiftsTrueMarcoFormat'] = \
    plotTrajParamsFuncUnit['trueParams']['trueSubjDysfuncScoresSU'][:, nrCurrFuncUnit]
    print('self.mapBiomkToFuncUnits', self.mapBiomkToFuncUnits)
    labels = [self.params['labels'][b] for b in range(len(self.params['labels']))
      if self.mapBiomkToFuncUnits[b] == nrCurrFuncUnit]
    plotTrajParamsFuncUnit['labels'] = labels
    # print('plotTrajParamsFuncUnit', plotTrajParamsFuncUnit)
    # print(adsa)
    return plotTrajParamsFuncUnit

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








