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
import DPMModelGeneric

class SigmoidModelBuilder(DisProgBuilder.DPMBuilder):

  def __init__(self, plotTrajParams):
    self.plotterObj = Plotter.PlotterFuncUnit(plotTrajParams)

  def generate(self, dataIndices, expName, params):
    return SigmoidModelWrapper(dataIndices, expName, params, self.plotterObj)


class SigmoidModelWrapper(DisProgBuilder.DPMInterface):

  def __init__(self, dataIndices, expName, params, plotter):
    self.dataIndices = dataIndices
    self.expName = expName
    self.params = params
    self.outFolder = params['outFolder']
    os.system('mkdir -p %s' % self.outFolder)
    self.params['plotTrajParams']['outFolder'] = self.outFolder
    self.params['plotTrajParams']['expName'] = expName
    self.plotter = plotter

    self.plotter.plotTrajParams['title'] = 'Biomarker traj.'
    self.plotter.plotTrajParams['colorsTraj'] = params['plotTrajParams']['colorsTrajBiomkB']


    Xfilt, Yfilt, visitIndicesFilt = filterDataListFormat(params, dataIndices)

    self.model = SigmoidModel.SigmoidModel(Xfilt, Yfilt, visitIndicesFilt,
                                           self.outFolder, plotter, self.params['labels'], self.params)

    self.nrBiomk = len(Xfilt)

    self.nrDis = len(self.params['disLabels'])
    # boolean masks
    self.binMaskSubjForEachDisD = [np.in1d(self.params['plotTrajParams']['diag'],
      self.params['diagsSetInDis'][disNr]) for disNr in range(self.nrDis)]

    # integer arrays
    self.disIdxForEachSubjS = np.nan * np.ones(len(params['X'][0]), int)
    self.indSubjForEachDisD = [0 for _ in range(self.nrDis)]
    for d in range(self.nrDis):
      assert self.binMaskSubjForEachDisD[d].shape[0] == self.disIdxForEachSubjS.shape[0]
      self.disIdxForEachSubjS[self.binMaskSubjForEachDisD[d]] = d
      self.indSubjForEachDisD[d] = np.where(self.binMaskSubjForEachDisD[d])[0]

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
    # fig = self.plotter.plotCompWithTrueParams(self.model, replaceFig=True)
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

    return trajSamplesBXS

  def getIndxSubjToKeep(self, disNr):
    # listOfLists = [np.where(self.binMaskSubjForEachDisD[disNr])[0] for disNr in disNrs]
    # return np.sort([l for l2 in listOfLists for l in l2])
    return np.where(self.binMaskSubjForEachDisD[disNr])[0]

  def getDataDis(self, disNr):
    XshiftedBS, X_BS, _, _ =  self.model.getData()
    indCurrDis = self.getIndxSubjToKeep(disNr)
    XshiftedCurrDisBS, XcurrDis_BS = DPMModelGeneric.DPMModelGeneric.filterXYsubjInd(XshiftedBS, X_BS, indCurrDis)
    return XshiftedCurrDisBS, XcurrDis_BS

  def getDataDisOverBiomk(self, disNr):

    """
    get sub-shifts for each biomarker (not for functional units or dis units),
    only for the diseases disNr

    :param disNr:
    :return:
    """
    indxSubjToKeep = self.getIndxSubjToKeep(disNr)
    nrSubCurrDis = indxSubjToKeep.shape[0]
    XshiftedAllTimeptsBS = [[] for b in range(self.nrBiomk)]
    ysPredBS = [[] for b in range(self.nrBiomk)]
    XshiftedCurrDisBS, XcurrDisBS = self.getDataDis(disNr)
    xsOrigPred1S = XcurrDisBS[0]  # all biomarkers should contain all timepoints in the disease model

    #### construct sub-shifts for each biomarker
    for s in range(nrSubCurrDis):
      bTmp = 0  # some biomarker, doesn't matter which one

      allXsCurrSubj = []
      for b in range(self.nrBiomk):
        allXsCurrSubj += list(XshiftedCurrDisBS[b][s])

      for b in range(self.nrBiomk):
        XshiftedAllTimeptsBS[b] += [np.sort(np.unique(allXsCurrSubj))]

      # print('XshiftedAllTimeptsBS', XshiftedAllTimeptsBS[0][s])
      # print('XshiftedCurrDisBS[b][s]', [XshiftedCurrDisBS[b][s] for b in range(self.nrBiomk)])
      # print(dasa)
      disNrCurrSubj = self.disIdxForEachSubjS[indxSubjToKeep][s]
      assert disNrCurrSubj == disNr

      ysCurrSubXB = self.predictBiomkSubjGivenXs(XshiftedAllTimeptsBS[0][s], disNrCurrSubj)

      for b in range(self.nrBiomk):
        ysPredBS[b] += [ysCurrSubXB[:, b]]

    return XshiftedCurrDisBS, ysPredBS, xsOrigPred1S

  def getXsMinMaxRange(self, _):
    return self.model.getXsMinMaxRange()