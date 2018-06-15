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

    # boolean masks
    self.indxSubjForEachDisD = [np.in1d(self.params['plotTrajParams']['diag'],
      self.params['diagsSetInDis'][disNr]) for disNr in range(self.nrDis)]

    self.plotter = Plotter.PlotterJDM(self.params['plotTrajParams'])

    # integer arrays
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

      self.makePlots(plotFigs, 0, 0)

      for i in range(nrIt):
        # # estimate biomk trajectories - disease agnostic
        # self.estimBiomkTraj(self.unitModels, self.disModels)
        # self.makePlots(plotFigs, i, 1)
        #
        # # estimate unit trajectories - disease specific
        # self.estimTrajWithinDisModel(self.unitModels, self.disModels)
        # self.makePlots(plotFigs, i, 2)

        # estimate subject latent variables
        self.estimSubjShifts(self.unitModels, self.disModels)
        self.makePlots(plotFigs, i, 3)


    res = None
    return res

  def makePlots(self, plotFigs, iterNr, picNr):
    if plotFigs:
      fig = self.plotter.plotCompWithTrueParams(self.unitModels, self.disModels, replaceFig=True)
      fig.savefig('%s/compTrueParams%d%d_%s.png' % (self.outFolder, iterNr, picNr, self.expName))

      for d in range(self.nrDis):
        fig = self.plotter.plotAllBiomkDisSpace(self, self.params, d)
        fig.savefig('%s/trajDisSpace%s_%d%d_%s.png' % (self.outFolder, self.params['disLabels'][d],
          iterNr, picNr, self.expName))

  def initParams(self):
    paramsCopy = copy.deepcopy(self.params)
    paramsCopy['nrGlobIterDis'] = 4 # set only two iterations, quick initialisation
    paramsCopy['nrGlobIterUnit'] = 4  # set only two iterations, quick initialisation
    paramsCopy['outFolder'] = '%s/init' % paramsCopy['outFolder']
    paramsCopy['penaltyUnits'] = 1
    # paramsCopy['priors'] = dict(prior_length_scale_mean_ratio=0.2,  # mean_length_scale = (self.maxX-self.minX)/3
    #                         prior_length_scale_std=1e-4, prior_sigma_mean=0.005, prior_sigma_std=1e-9,
    #                         prior_eps_mean=0.1, prior_eps_std=1e-6)
    # paramsCopy['priors'] = dict(prior_length_scale_mean_ratio=0.9,  # mean_length_scale = (self.maxX-self.minX)/3
    #                         prior_length_scale_std=1e-4, prior_sigma_mean=3, prior_sigma_std=1e-3,
    #                         prior_eps_mean=0.1, prior_eps_std=1e-6)
    onePassModel = JointModelOnePass.JDMOnePass(self.dataIndices, self.expName, paramsCopy)

    onePassModel.run(runPart = 'LL')

    self.unitModels = onePassModel.unitModels
    self.disModels = onePassModel.disModels

  def estimSubjShifts(self, unitModels, disModels):

    XshiftedScaledDBSX = [0 for _ in range(self.nrDis)]
    XdisDBSX = [0 for _ in range(self.nrDis)]
    X_arrayScaledDB = [0 for _ in range(self.nrDis)]
    for d in range(self.nrDis):
      XshiftedScaledDBSX[d], XdisDBSX[d], _, X_arrayScaledDB[d] = disModels[d].getData()
      nrSubjCurr = len(XshiftedScaledDBSX[d][0])
      for s in range(nrSubjCurr):

        TODO: add the subject shift optimisation

        objectiveFun = lambda time_shift_one_sub: -self.log_posterior_time_shift_Raz(
          time_shift_one_sub, s)[0]

        print('initLik', likJDMobjFunc(initParams))
        print('initParams', initParams)
        resStruct = scipy.optimize.minimize(likJDMobjFunc, initParams, method='Nelder-Mead',
          options={'disp': True, 'maxiter':50})


        disModels[d].parameters[u] = [resStruct.x, initVariance]


  def estimBiomkTraj(self, unitModels, disModels):

    XshiftedScaledDBS = [0 for _ in range(self.nrDis)]
    XdisDBSX = [0 for _ in range(self.nrDis)]
    for d in range(self.nrDis):
      XshiftedScaledDBS[d], XdisDBSX[d], _, _ = disModels[d].getData()

    XdisSX = [0 for _ in range(self.unitModels[0].nrSubj)]
    nrSubj = unitModels[0].nrSubj
    predScoresCurrUSX = [[0 for s in range(nrSubj)] for u in range(self.nrFuncUnits)]
    for s in range(nrSubj):
      currDis = self.disIdxForEachSubjS[s]
      currRID = self.params['RID'][s]
      # print('currDis', currDis)
      idxCurrSubjInDisModel = np.where(self.ridsPerDisD[currDis] == currRID)[0][0]
      XdisSX[s] = XdisDBSX[currDis][0][idxCurrSubjInDisModel]

      # get shifts for curr subj from correct disModel
      currXdataShifted = XshiftedScaledDBS[currDis][0][idxCurrSubjInDisModel]
      # predict dysf scoresf for curr subj
      predScoresCurrXU = disModels[currDis].predictBiomk(currXdataShifted)

      for u in range(self.nrFuncUnits):
        predScoresCurrUSX[u][s] = predScoresCurrXU[:,u]

      assert currXdataShifted.shape[0] == XdisSX[s].shape[0]
      assert predScoresCurrXU[:,0].shape[0] == XdisSX[s].shape[0]
      assert predScoresCurrUSX[0][s].shape[0] == XdisSX[s].shape[0]

    for u in range(self.nrFuncUnits):
      print('updating traj for func unit %d/%d' % (u+1, self.nrFuncUnits))
      # now update the X-values in each unitModel to the updated dysfunc scores
      # not that the X-vals are the same for every biomk within func units,
      # but initial missing values within each biomk are kept
      self.unitModels[u].updateXvals(predScoresCurrUSX[u], XdisSX)
      self.unitModels[u].priors = self.params['priorsJMD']
      self.unitModels[u].Set_penalty(2)
      self.unitModels[u].Optimize_GP_parameters(Niterat = 70)

      # fig = self.unitModels[u].plotter.plotTraj(self.unitModels[u])
      # fig2 = self.unitModels[u].plotter.plotCompWithTrueParams(self.unitModels[u], replaceFig=False)

      # fig.savefig('%s/allTraj%d0_%s.png' % (self.outFolder, i + 1, self.expName))


  def estimTrajWithinDisModel(self, unitModels, disModels):
    """ estimates trajectory parameters within the disease specific models """

    XshiftedScaledDBSX = [0 for _ in range(self.nrDis)]
    XdisDBSX = [0 for _ in range(self.nrDis)]
    X_arrayScaledDB = [0 for _ in range(self.nrDis)]
    for d in range(self.nrDis):
      XshiftedScaledDBSX[d], XdisDBSX[d], _, X_arrayScaledDB[d] = disModels[d].getData()

      # update each unit-traj independently
      for u in range(self.nrFuncUnits):
        Y_arrayCurDis, indFiltToMisCurDisB = unitModels[u].filterLongArray(unitModels[u].Y_array,
          unitModels[u].N_obs_per_sub, self.indxSubjForEachDisD[d],unitModels[u].visitIndices)

        for b in range(len(Y_arrayCurDis)):
          assert Y_arrayCurDis[b].shape[0] == indFiltToMisCurDisB[b].shape[0]
          print(Y_arrayCurDis[b].shape)

        # print(adsa)

        # build function that within a disease model takes a unit traj , and predicts lik of given
        # unit-traj params in corresponding unitModel.
        likJDMobjFunc = lambda paramsCurrTraj: self.likJDM(disModels[d], unitModels[u],
          X_arrayScaledDB[d], XdisDBSX, paramsCurrTraj, u, d, Y_arrayCurDis, indFiltToMisCurDisB)
        initParams, initVariance = disModels[d].unpack_parameters(disModels[d].parameters[u])


        print('initLik', likJDMobjFunc(initParams))
        # print(adsa)
        print('initParams', initParams)
        # print('10 evals', [likJDMobjFunc(initParams + perturb) for perturb in
        #   np.random.normal(0,0.01,10)])
        # print(adsa)
        # print(ads)
        resStruct = scipy.optimize.minimize(likJDMobjFunc, initParams, method='Nelder-Mead',
          options={'disp': True, 'maxiter':50})

        print('resStruct', resStruct)
        likJDMobjFuncBreak = lambda paramsCurrTraj: self.likJDM(disModels[d], unitModels[u],
          X_arrayScaledDB[d], XdisDBSX, paramsCurrTraj, u, d, Y_arrayCurDis, indFiltToMisCurDisB, breakPoint=True)
        print('finalLik', likJDMobjFuncBreak(resStruct.x), likJDMobjFunc(resStruct.x))

        disModels[d].parameters[u] = [resStruct.x, initVariance]

        # import pdb
        # pdb.set_trace()

      # for s in range(len(XshiftedScaledDBSX[d][0])):
      #   biomkPredXB = self.predictBiomkSubjGivenXs(XshiftedScaledDBSX[d][0][s], d)
      #   print('biomkPredXB', biomkPredXB)
      #   print('')



  def likJDM(self, disModel, unitModel, X_arrayScaledDisB, XdisDBSX, params, unitNr, disNr, Y_arrayCurDis,
             indFiltToMisCurDisB, breakPoint=False):
    """
    function computes the likelihood of params of one traj within one disModel.

    :param disModel:
    :param unitModel:
    :param X_arrayScaledDisB: X_array values (scaled) for patients withing current disease
    :param XdisDBSX:
    :param params: parameters for one unit-trajectory to optimise, within a diseModel
    :param unitNr:
    :param disNr:
    :param Y_arrayCurDis:

    :return: log-likelihood
    """

    ####### first predict dysScores in disModel for curr functional unit ##########

    # XdisSX = [0 for _ in range(unitModel.nrSubj)]
    # nrSubjDisModel = disModel.nrSubj
    # predScoresCurrSX = [0 for s in range(nrSubjDisModel)]
    # for s in range(nrSubjDisModel):
      # currDis = self.ridsPerDisD[currDis][s]
      # currRID = self.params['RID'][s]
      # idxCurrSubjInDisModel = np.where(self.ridsPerDisD[currDis] == currRID)[0][0]
      # idxSubjInBigArray = self.indxSubjForEachDisD[disNr][s]
      # XdisSX[s] = XdisDBSX[disNr][0][idxSubjInBigArray]

    # get shifts for curr subj from correct disModel
    # currXdataShifted = XshiftedScaledDBSX[disNr][0][idxSubjInBigArray]
    # predict dysf scoresf for curr subj
    paramsAllU = disModel.parameters
    paramsAllU[unitNr] = [params, paramsAllU[unitNr][1]]
    # print('paramsAllU', paramsAllU)
    predScoresCurrXU = disModel.predictBiomkWithParams(X_arrayScaledDisB[0], paramsAllU)

    if breakPoint:
      print('predScoresCurrXU', predScoresCurrXU)
      # import pdb
      # pdb.set_trace()
      # print(adsa)

    # predScoresCurrSX[s] = predScoresCurrXU[:,unitNr]

    # assert currXdataShifted.shape[0] == XdisSX[s].shape[0]
    # assert predScoresCurrXU[:,0].shape[0] == XdisSX[s].shape[0]
    # assert predScoresCurrSX[0][s].shape[0] == XdisSX[s].shape[0]

    #### then sum log-liks of each func unit.

    # turn predScoresCurrSX[u] into predX_arrayUnitModel
    # predX_arrayUnitModel, N_obs_per_subj = unitModel.convertLongToArray(predScoresCurrSX)

    predX_arrayUnitModel = [0 for b in range(unitModel.nrBiomk)]

    # print('indFiltToMisCurDisB', indFiltToMisCurDisB)
    # print(ads)

    sumLik =0
    for b in range(unitModel.nrBiomk):
      predScoresCurrBiomk = predScoresCurrXU[:,unitNr].reshape(-1,1)
      # print('b', b)
      # print('predScoresCurrBiomk', predScoresCurrBiomk.shape)
      # print('indFiltToMisCurDisB[b]', indFiltToMisCurDisB[b].shape)
      # print('Y_arrayCurDis[b]', Y_arrayCurDis[b].shape)
      predX_arrayUnitModel[b] = unitModel.filterZarray(predScoresCurrBiomk,
        indFiltToMisCurDisB[b])
      # print('predX_arrayUnitModel', predX_arrayUnitModel[0].shape)
      print('Y_arrayCurDis', Y_arrayCurDis[0].shape)
      assert predX_arrayUnitModel[b].shape[0] == Y_arrayCurDis[b].shape[0]

      if len(Y_arrayCurDis[b]) > 0:
        lik = unitModel.stochastic_grad_manual_onebiomk(unitModel.parameters[b],
        predX_arrayUnitModel[b], Y_arrayCurDis[b], unitModel.penalty[b], fixSeed=True)[0]
        sumLik += lik

    return sumLik # sum the likelihood from each trajectory where there is data


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
      # dysfuncPredXU[:, u] = self.unitModels[u].applyScalingXzeroOneInv(dysfuncPredXU[:,u])
      biomkPredXB[:, self.biomkInFuncUnit[u]] = \
        self.unitModels[u].predictBiomk(dysfuncPredXU[:,u])


    # assumes these biomarkers are at the end
    biomkIndNotInFuncUnits = self.biomkInFuncUnit[-1]
    nrBiomkNotInUnit = biomkIndNotInFuncUnits.shape[0]
    if nrBiomkNotInUnit > 0:
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
      # dysfuncPredXU[:, u] = self.unitModels[u].applyScalingXzeroOneInv(dysfuncPredXU[:, u])
      for b in range(biomkIndInCurrUnit.shape[0]):
        trajSamplesBXS[biomkIndInCurrUnit[b],:,:] = \
            self.unitModels[u].sampleTrajPost(dysfuncPredXU[:,u], b, nrSamples)


    biomkIndNotInFuncUnits = self.biomkInFuncUnit[-1]
    nrBiomkNotInUnit = biomkIndNotInFuncUnits.shape[0]
    if nrBiomkNotInUnit > 0:
      # assumes these biomarkers are at the end
      indOfRealBiomk =  list(range(dysfuncPredXU.shape[1] - nrBiomkNotInUnit, dysfuncPredXU.shape[1]))
      for b in range(len(biomkIndNotInFuncUnits)):
        trajSamplesBXS[biomkIndNotInFuncUnits[b],:,:] = \
          self.disModels[disNr].sampleTrajPost(newXs, indOfRealBiomk[b], nrSamples)

    assert not np.isnan(trajSamplesBXS).any()

    return trajSamplesBXS


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








