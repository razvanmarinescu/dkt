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
import JointModel
import DisProgBuilder

class IncompleteBuilder(DisProgBuilder.DPMBuilder):
  # builds a voxel-wise disease progression model

  def __init__(self, plotTrajParams, gpModels):
    self.plotterObj = Plotter.PlotterJDM(plotTrajParams)
    self.gpModels = gpModels

  def setPlotter(self, plotterObj):
    self.plotterObj = plotterObj

  # set complete model
  def setGPModels(self, gpModels):
    self.gpModels = gpModels

  def generate(self, dataIndices, expName, params):
    return IncompleteModel(dataIndices, expName, params, self.plotterObj, self.gpModels)

class IncompleteModel(JointModel.JointModel):

  def __init__(self, dataIndices, expName, params, plotterObj, gpModels):
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

    self.gpModels = gpModels
    self.grandModel = None


  def runStd(self, runPart):
    self.run(runPart)

  def run(self, runPart):

    unitsStagingFile = '%s/unitsStaging.npz' % self.outFolder
    if runPart[0] == 'R':
      meanStagesUS = [0 for x in range(self.nrFuncUnits)]#
      xsU = [0 for x in range(self.nrFuncUnits)]
      for u in range(self.nrFuncUnits):
        deltaX = (self.gpModels[u].maxX - self.gpModels[u].minX)
        xsU[u] = np.linspace(self.gpModels[u].minX, self.gpModels[u].maxX + deltaX, num=100).reshape([100, 1])

        plotTrajParamsFuncUnit = self.createPlotTrajParamsFuncUnit(nrCurrFuncUnit = u)
        plotterObjCurrFuncUnit = Plotter.PlotterGP(plotTrajParamsFuncUnit)  # set separate plotter for the

        nrSubjToSkip = 1
        xSmall = [b[::nrSubjToSkip] for b in self.params['XemptyListsAllBiomk']]
        ySmall = [b[::nrSubjToSkip] for b in self.params['YemptyListsAllBiomk']]
        _, meanStagesUS[u] = self.gpModels[u].StageSubjects(xSmall, ySmall, xsU[u])

        for l in range(self.gpModels[u].N_biom):
            # Creating 1d arrays of individuals' time points and observations
            self.gpModels[u].X_array.append([np.float(item) for sublist in X[l] for item in sublist])
            self.gpModels[u].Y_array.append([np.float(item) for sublist in Y[l] for item in sublist])
            self.gpModels[u].N_obs_per_sub.append([len(X[l][j]) for j in range(len(X[l]))])



        # fig = plotterObjCurrFuncUnit.plotCompWithTrueParams(self.gpModels[u],
        #   replaceFig = True, subjStagesEstim=meanStagesUS[u])
        fig = plotterObjCurrFuncUnit.scatterPlotShifts(self.gpModels[u], meanStagesUS[u])
        fig.savefig('%s/shiftsScatterUnit%d.png' % (self.outFolder, u))


      dataStruct = dict(meanStagesUS=meanStagesUS,xsU=xsU)
      pickle.dump(dataStruct, open(unitsStagingFile, 'wb'), protocol=pickle.HIGHEST_PROTOCOL)
      # pl.pause(100)
    else:
      dataStruct = pickle.load(open(unitsStagingFile, 'rb'))
      meanStagesUS = dataStruct['meanStagesUS']
      xsU = dataStruct['xsU']

    grandModelFile = '%s/grandModel.npz' % self.outFolder
    if runPart[1] == 'R':
      dysfuncScoresU = [0 for x in range(self.nrFuncUnits)]
      xDysfunSubjU = [0 for x in range(self.nrFuncUnits)]
      for u in range(self.nrFuncUnits):
        nrSubj = self.gpModels[u].N_samples
        dysfuncScoresU[u] = [[] for _ in range(nrSubj)]
        xDysfunSubjU[u] = [[] for _ in range(nrSubj)]
        for sub in range(self.gpModels[u].N_samples):
          for b in range(self.gpModels[u].N_biom):
            # dysfuncScoresU[u][sub] += [self.gpModels[u].X_array[b][k][0] for k in range(int(np.sum(
            #   self.gpModels[u].N_obs_per_sub[b][:sub])), np.sum(self.gpModels[u].N_obs_per_sub[b][:sub + 1]))]
            xDysfunSubjU[u][sub] += [self.gpModels[u].X[b][sub]]


          xDysfunSubjU[u][sub] = np.unique(xDysfunSubjU[u][sub])

          dysfuncScoresU[u][sub] = xDysfunSubjU[u][sub] + meanStagesUS[u][sub]

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

    elif runPart[0] == 'L':
      self.grandModel = pickle.load(open(grandModelFile, 'rb'))

    res = None
    return res


  def createPlotTrajParamsFuncUnit(self, nrCurrFuncUnit):
    plotTrajParamsFuncUnit = copy.deepcopy(self.params['plotTrajParams'])
    plotTrajParamsFuncUnit['nrRows'] = self.params['plotTrajParams']['nrRowsFuncUnit']
    plotTrajParamsFuncUnit['nrCols'] = self.params['plotTrajParams']['nrColsFuncUnit']
    plotTrajParamsFuncUnit['unitNr'] = nrCurrFuncUnit  # some plotting functions need to know the current unit
    plotTrajParamsFuncUnit['isRunningFuncUnit'] = True
    plotTrajParamsFuncUnit['trueParams']['subShiftsTrueMarcoFormat'] = \
    plotTrajParamsFuncUnit['trueParams']['trueSubjDysfuncScoresSU'][:, nrCurrFuncUnit]
    # print('self.mapBiomkToFuncUnits', self.mapBiomkToFuncUnits)
    # labels = [self.params['labels'][b] for b in range(len(self.params['labels'])) if
    #           self.mapBiomkToFuncUnits[b] == nrCurrFuncUnit]
    # plotTrajParamsFuncUnit['labels'] = labels
    # # print('plotTrajParamsFuncUnit', plotTrajParamsFuncUnit)
    # print(adsa)
    return plotTrajParamsFuncUnit







