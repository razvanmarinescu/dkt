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

  def __init__(self, unitModelObj, disModelObj, priorsUnitModels, priorsDisModels):
    self.unitModelObj = unitModelObj
    self.disModelObj = disModelObj
    self.priorsUnitModels = priorsUnitModels
    self.priorsDisModels = priorsDisModels

  def generate(self, dataIndices, expName, params):
    return JointModel(dataIndices, expName, params,
      self.unitModelObj, self.disModelObj, self.priorsUnitModels, self.priorsDisModels)

class JointModel(DisProgBuilder.DPMInterface):

  def __init__(self, dataIndices, expName, params, unitModelObj, disModelObj, priorsUnitModels, priorsDisModels):
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
    self.binMaskSubjForEachDisD = [np.in1d(self.params['plotTrajParams']['diag'],
      self.params['diagsSetInDis'][disNr]) for disNr in range(self.nrDis)]

    self.plotter = Plotter.PlotterJDM(self.params['plotTrajParams'])

    # integer arrays
    self.disIdxForEachSubjS = np.zeros(len(params['X'][0]), int)
    self.indSubjForEachDisD = [0 for _ in range(self.nrDis)]
    for d in range(self.nrDis):
      self.disIdxForEachSubjS[self.binMaskSubjForEachDisD[d]] = d
      self.indSubjForEachDisD[d] = np.where(self.binMaskSubjForEachDisD[d])[0]

    self.ridsPerDisD = [_ for _ in range(self.nrDis)]
    for d in range(self.nrDis):
      self.ridsPerDisD[d] = self.params['RID'][self.binMaskSubjForEachDisD[d]]

    self.priorsUnitModels = priorsUnitModels
    self.priorsDisModels = priorsDisModels

  def runStd(self, runPart):
    self.run(runPart)

  def run(self, runPart):
    filePath = '%s/unitModels.npz' % self.outFolder
    nrRandFeatures = int(3)  # Number of random features for kernel approximation

    plotFigs = True

    if runPart[0] == 'R':
      self.initParams()

    nrIt = 100
    if runPart[1] == 'R':

      # self.makePlots(plotFigs, 0, 0)

      # i = 4
      # self.loadCheckpoint(i, 4)

      i = 0
      while i < nrIt:

        # estimate biomk trajectories - disease agnostic
        self.estimBiomkTraj(self.unitModels, self.disModels, i)
        self.makePlots(plotFigs, i, 1)
        self.saveCheckpoint(i, 1)
        # self.loadCheckpoint(i, 1)

        # estimate subject latent variables
        self.estimSubjShifts(self.unitModels, self.disModels)
        self.makePlots(plotFigs, i, 2)
        self.saveCheckpoint(i, 2)
        # # self.loadCheckpoint(i, 2)
        #
        # # estimate unit trajectories - disease specific
        self.estimTrajWithinDisModel(self.unitModels, self.disModels)
        self.makePlots(plotFigs, i, 3)
        self.saveCheckpoint(i, 3)
        #
        # # estimate subject latent variables
        self.estimSubjShifts(self.unitModels, self.disModels)
        self.makePlots(plotFigs, i, 4)
        self.saveCheckpoint(i, 4)
        # self.loadCheckpoint(i, 4)

        # try to vary the scaling of the sub shifts, optimise dis traj at the same time.
        # the reason for this is because the !"range"! of the estimated shift is smaller than the true range.
        self.varyShiftsScale(self.unitModels, self.disModels)
        self.makePlots(plotFigs, i, 5)
        self.saveCheckpoint(i, 5)

        i += 1

    res = None
    return res

  def saveCheckpoint(self, iterNr, stepNr):
    fileRes = '%s/fitRes%d%d.npz' % (self.outFolder, iterNr, stepNr)
    pickle.dump(dict(unitModels=self.unitModels, disModels=self.disModels),
                open(fileRes, 'wb'), protocol=pickle.HIGHEST_PROTOCOL)

  def loadCheckpoint(self, iterNr, stepNr):
    fileRes = '%s/fitRes%d%d.npz' % (self.outFolder, iterNr, stepNr)
    ds = pickle.load(open(fileRes, 'rb'))
    self.unitModels = ds['unitModels']
    self.disModels = ds['disModels']

  def makePlots(self, plotFigs, iterNr, picNr):
    if plotFigs:
      fig = self.plotter.plotCompWithTrueParams(self.unitModels, self.disModels, replaceFig=True)
      fig.savefig('%s/compTrueParams%d%d_%s.png' % (self.outFolder, iterNr, picNr, self.expName))

      fig = self.plotter.plotHierData(self.unitModels, self.disModels, replaceFig=True)
      fig.savefig('%s/plotHierData%d%d_%s.png' % (self.outFolder, iterNr, picNr, self.expName))

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
    onePassModel = JointModelOnePass.JDMOnePass(self.dataIndices, self.expName, paramsCopy,
      self.priorsUnitModels, self.priorsDisModels, self.priorsUnitModels, self.priorsDisModels)

    onePassModel.run(runPart = 'LL')

    self.unitModels = onePassModel.unitModels
    self.disModels = onePassModel.disModels

    for d in range(self.nrDis):
      self.disModels[d].priors = self.params['priorsDisModels'][d]


  def varyShiftsScale(self, unitModels, disModels):
    """
    For each disease, multiply the DPS scores by a factor X, then re-estimate the traj within disModels.
    Try 10 different scaling factors and select the one that maximises the model likelihood.

    :param unitModels:
    :param disModels:
    :return:
    """

    # scalingFactors = [0.3, 1, 3]
    scalingFactors = [0.6,0.9,1,1.1,1.5]
    nrScFact = len(scalingFactors)

    XshiftedScScFDBSX = [[0 for _ in range(self.nrDis)] for sc in range(nrScFact)]
    XshiftedScaledDBSX = [0 for _ in range(self.nrDis)]
    X_arrayScaledDB = [0 for _ in range(self.nrDis)]
    optimalShiftsDisModels = [0 for _ in range(self.nrDis)]
    bestScalingIndD = np.zeros(self.nrDis, int)
    informPriorTrajDisModels = [True, True] # set informative priors only for the first disease
    nrSubjAllDis = unitModels[0].nrSubj
    optimalShiftsAll = np.zeros(nrSubjAllDis)
    for d in range(self.nrDis):
      optimalShiftsDisModels[d] = np.zeros((2, self.indSubjForEachDisD[d].shape[0]))


    for d in range(self.nrDis):
      XshiftedScaledDBSX[d], _, _, X_arrayScaledDB[d] = disModels[d].getData()
      optimParams = [[0 for u in range(self.nrFuncUnits)] for _ in range(nrScFact)]

      sumSSD = np.zeros(nrScFact)
      X_arrayScScFBX = [0 for sc in range(nrScFact)]
      for sc in range(nrScFact):

        XshiftedScScFDBSX[sc][d] = self.addScalingFactor(XshiftedScaledDBSX[d], scalingFactors[sc])
        X_arrayScScFBX[sc], _, _ = disModels[d].convertLongToArray(XshiftedScScFDBSX[sc][d], disModels[d].visitIndices)

        timeShiftPriorStd = disModels[d].priors['timeShiftStd']
        priorSubShifts = np.sum([(XshiftedScScFDBSX[sc][d][0][s][0] - 0) ** 2 /timeShiftPriorStd
          for s in range(disModels[d].nrSubj)])

        for u in range(self.nrFuncUnits):
          Y_arrayCurDis, indFiltToMisCurDisB = unitModels[u].filterLongArray(unitModels[u].Y_array,
            unitModels[u].N_obs_per_sub, self.binMaskSubjForEachDisD[d],unitModels[u].visitIndices)

          # for b in range(len(Y_arrayCurDis)):
          #   assert Y_arrayCurDis[b].shape[0] == indFiltToMisCurDisB[b].shape[0]
          #   # print(Y_arrayCurDis[b].shape)

          # build function that within a disease model takes a unit traj, and predicts lik of given
          # unit-traj params in corresponding unitModel.


          ssdJDMobjFunc = lambda paramsCurrTraj: self.ssdJDMOneUnitTraj(disModels[d], unitModels[u],
            X_arrayScScFBX[sc], [paramsCurrTraj, None], u, d, Y_arrayCurDis, indFiltToMisCurDisB,
            informPriorTrajDisModels[d])
          initParams, initVariance = disModels[d].unpack_parameters(disModels[d].parameters[u])

          print('initSSD', ssdJDMobjFunc(initParams))

          resStruct = scipy.optimize.minimize(ssdJDMobjFunc, initParams, method='Nelder-Mead',
            options={'disp': True, 'maxiter':1000})
          optimParams[sc][u] = [resStruct.x, initVariance]
          sumSSD[sc] += ssdJDMobjFunc(resStruct.x)

          print('resStruct', resStruct)


        # sumSSD[sc] += priorSubShifts

      # print('sumSSD', sumSSD, np.argsort(sumSSD))

      bestScalingIndD[d] = int(np.argmin(sumSSD))

      # print('bestScalingIndD[d]', bestScalingIndD[d])
      # print('best scaling factor', scalingFactors[bestScalingIndD[d]])
      # print('X_arrayScScFBX[sc] range', np.min(X_arrayScScFBX[bestScalingIndD[d]][0]),
      #   np.max(X_arrayScScFBX[bestScalingIndD[d]][0]))
      # print(adsa)
      disModels[d].parameters = optimParams[bestScalingIndD[d]]

    # update optimal time shifts and Y-values in disease model

    for s in range(nrSubjAllDis):
      d = self.disIdxForEachSubjS[s]
      subjCurrIndInDisModel = np.where(self.indSubjForEachDisD[d] == s)[0][0]
      oldStage = XshiftedScaledDBSX[d][0][subjCurrIndInDisModel][0]
      newStage = XshiftedScScFDBSX[bestScalingIndD[d]][d][0][subjCurrIndInDisModel][0]

      optimalShiftsAll[s] = newStage - oldStage
      optimalShiftsDisModels[d][0, subjCurrIndInDisModel] = optimalShiftsAll[s]

      # print('abs(oldStage + optimalShiftsAll[s] - newStage)', abs(oldStage + optimalShiftsAll[s] - newStage))
      if abs(oldStage + optimalShiftsAll[s] - newStage) > 0.001:
        print('oldStage', oldStage)
        print('optimalShiftsAll[s]', optimalShiftsAll[s])
        print('newStage', newStage)
        raise ValueError('Shifts dont match')


    for d in range(self.nrDis):
      disModels[d].updateTimeShiftsAndData(optimalShiftsDisModels[d])

    # update optimal time shifts in all functional models
    # actually not needed because when I optimise the unit traj I automatically update the X-vals then
    self.updateXvalsInFuncModels(optimalShiftsAll, disModels, unitModels, XshiftedScaledDBSX)

    # print('dis0 final range', np.min(disModels[0].X_array[0]), np.max(disModels[0].X_array[0]))
    # print(ads)

  def addScalingFactor(self, X_BSX, scaleFactor):
    newX_BSX = [[0 for s in range(len(X_BSX[0]))] for b in range(len(X_BSX))]
    for b in range(len(X_BSX)):
      for s in range(len(X_BSX[0])):
        newX_BSX[b][s] = ((scaleFactor - 1) * X_BSX[b][s][0]) + X_BSX[b][s]

    return newX_BSX

  def estimSubjShifts(self, unitModels, disModels):

    YunitUBSX = [0 for _ in range(self.nrFuncUnits)]
    trajParamsU = [0 for _ in range(self.nrFuncUnits)]
    optimalShiftsDisModels = [0 for _ in range(self.nrDis)]

    for u in range(self.nrFuncUnits):
      _, _, YunitUBSX[u], _ = unitModels[u].getData()
      trajParamsU[u] = unitModels[u].computeTrajParamsForTimeShifts()

    XshiftedScaledDBSX = [0 for _ in range(self.nrDis)]
    XdisDBSX = [0 for _ in range(self.nrDis)]
    X_arrayScaledDB = [0 for _ in range(self.nrDis)]
    for d in range(self.nrDis):
      XshiftedScaledDBSX[d], XdisDBSX[d], _, X_arrayScaledDB[d] = disModels[d].getData()
      optimalShiftsDisModels[d] = np.zeros((2, self.indSubjForEachDisD[d].shape[0]))

    nrSubjAllDis = len(YunitUBSX[0][0])
    optimalShiftsAll = np.zeros(nrSubjAllDis)

    for s in range(nrSubjAllDis):
      d = self.disIdxForEachSubjS[s]
      subjCurrIndInDisModel = np.where(self.indSubjForEachDisD[d] == s)[0][0]
      XshiftedScaledCurrX = XshiftedScaledDBSX[d][0][subjCurrIndInDisModel]
      likJDMShiftsObjFunc = lambda time_shift_delta_one_sub: self.log_posterior_time_shift_Raz(
        time_shift_delta_one_sub, disModels[d], unitModels, XshiftedScaledCurrX, YunitUBSX,
        s, trajParamsU)

      initDeltaShift = 0 # this shift is added on top of the existing shift

      resStruct = scipy.optimize.minimize(likJDMShiftsObjFunc, initDeltaShift, method='Nelder-Mead',
        options={'disp': True})

      optimalShiftsAll[s] = resStruct.x

      # self.plotShiftBeforeAfter(optimalShiftsAll[s], XshiftedScaledCurrX, YunitUBSX, d, s)

      # now update the disease model and the unitModels with the optimal shift
      optimalShiftsDisModels[d][0, subjCurrIndInDisModel] = optimalShiftsAll[s]

    # update optimal time shifts and Y-values in disease model
    for d in range(self.nrDis):
      disModels[d].updateTimeShiftsAndData(optimalShiftsDisModels[d])

    # update optimal time shifts in all functional models
    # actually not needed because when I optimise the unit traj I automatically update the X-vals then
    self.updateXvalsInFuncModels(optimalShiftsAll, disModels, unitModels, XshiftedScaledDBSX)

  def plotShiftBeforeAfter(self, tOptim, XshiftedScaledCurrX, YunitUBSX, d, s):
    tShifts = [0, tOptim]
    nrTimeShifts = len(tShifts)
    xsDisBSX = [[0 for t in range(nrTimeShifts)] for b in range(self.disModels[d].nrBiomk)]
    ysDisBSX = [[0 for t in range(nrTimeShifts)] for b in range(self.disModels[d].nrBiomk)]
    xsFuncUBSX = [[[0 for t in range(nrTimeShifts)] for b in range(self.unitModels[u].nrBiomk)] for u in
                  range(self.nrFuncUnits)]
    ysFuncUBSX = [[[0 for t in range(nrTimeShifts)] for b in range(self.unitModels[u].nrBiomk)] for u in
                  range(self.nrFuncUnits)]

    for t in range(len(tShifts)):

      predBiomksJointXB = self.predictBiomkSubjGivenXs(tShifts[t] + XshiftedScaledCurrX, d)
      for b in range(self.disModels[d].nrBiomk):
        xsDisBSX[b][t] = tShifts[t] + XshiftedScaledCurrX

      predBiomksXU = self.disModels[d].predictBiomk(xsDisBSX[0][t])

      for b in range(self.disModels[d].nrBiomk):
        ysDisBSX[b][t] = predBiomksXU[:, b]

      for u in range(self.nrFuncUnits):
        for b in range(self.unitModels[u].nrBiomk):
          Xdata = predBiomksXU[self.unitModels[u].visitIndices[b][s], u]
          Ypred = self.unitModels[u].predictBiomk(Xdata)
          xsFuncUBSX[u][b][t] = Xdata
          ysFuncUBSX[u][b][t] = YunitUBSX[u][b][s]

    self.plotter.plotHierGivenData(self.unitModels, self.disModels[d], xsDisBSX, ysDisBSX,
                                   xsFuncUBSX, ysFuncUBSX, replaceFig=False)


  def log_posterior_time_shift_Raz(self, time_shift_delta_one_sub, disModel, unitModels,
                                   XshiftedScaledX, YunitUBSX, s, trajParamsUB):

    timeShiftPriorSpread = disModel.priors['timeShiftStd']
    totalShift = time_shift_delta_one_sub + XshiftedScaledX[0]
    prior_time_shift = (totalShift[0] - 0) ** 2 / timeShiftPriorSpread
    predBiomksXU = disModel.predictBiomk(time_shift_delta_one_sub + XshiftedScaledX)

    logliks = []

    for u in range(self.nrFuncUnits):

      # Shifting data according to current time-shift estimate
      for b in range(unitModels[u].nrBiomk):
        if unitModels[u].visitIndices[b][s].shape[0] > 0:
          Xdata = predBiomksXU[unitModels[u].visitIndices[b][s], u].reshape(-1,1)
          # Xdata = Xdata # here need to apply scaling, if identity map was NOT used
          Ydata = YunitUBSX[u][b][s]
          Xdata = unitModels[u].applyScalingXForward(Xdata, biomk=0)
          Ypred = unitModels[u].predictBiomk(Xdata)

          YdataScaled = self.unitModels[u].applyScalingYInv(Ydata.reshape(-1,1), b)

          loglikCurr = unitModels[u].log_posterior_time_shift_onebiomk_given_arrays(
            Xdata, YdataScaled, trajParamsUB[u][b])

          logliks += [loglikCurr]


    return np.sum(logliks) + prior_time_shift

  def updateXvalsInFuncModels(self, time_shift_delta_all, disModels, unitModels,
                              XshiftedScaledDBSX):
    """
    after the correct time-shifts in the disease models have been estimated,
    function updates the X_values within the function models with the predictions
    from the disease models

    :param time_shift_delta_all:
    :param disModels:
    :param unitModels:
    :param XshiftedScaledDBSX:
    :return:
    """

    XdataNewUBSX = [[[np.array([]) for s in range(unitModels[u].nrSubj)] for b in
                     range(unitModels[u].nrBiomk)] for u in range(self.nrFuncUnits)]

    nrSubjAllDis = self.disIdxForEachSubjS.shape[0]
    for s in range(nrSubjAllDis):
      d = self.disIdxForEachSubjS[s]
      subjCurrIndInDisModel = np.where(self.indSubjForEachDisD[d] == s)[0][0]

      predBiomksXU = disModels[d].predictBiomk(time_shift_delta_all[s] +
        XshiftedScaledDBSX[d][0][subjCurrIndInDisModel])

      for u in range(self.nrFuncUnits):

        # Shifting data according to current time-shift estimate
        for b in range(unitModels[u].nrBiomk):
          print('unitModels[u].visitIndices[b][s]', unitModels[u].visitIndices[b][0])
          if unitModels[u].visitIndices[b][s].shape[0] > 0:
            XdataNewUBSX[u][b][s] = predBiomksXU[unitModels[u].visitIndices[b][s], u]
            # XdataNewUBSX[u][b][s] = XdataNewUBSX[u][b][s] # here need to apply scaling, if identity map was NOT used

    # update X_array for the current unitModel
    for u in range(self.nrFuncUnits):
      newXarray, _, _ = \
        unitModels[u].convertLongToArray(XdataNewUBSX[u], unitModels[u].visitIndices)

      for b in range(unitModels[u].nrBiomk):
        for s in range(unitModels[u].nrSubj):
          if XdataNewUBSX[u][b][s].shape[0] != unitModels[u].X[b][s].shape[0]:
            print('u b s', u, b, s)
            print(XdataNewUBSX[u][b][s].shape[0])
            print(unitModels[u].X[b][s].shape[0])
            raise ValueError('Shape of new array doesnt match')



      # dimension check
      assert newXarray[0].shape[0] == unitModels[u].X_array[0].shape[0]

      # check that at least the first value changed during the update
      # assert newXarray[0][0] != unitModels[u].X_array[0][0]
      unitModels[u].X_array = newXarray

      unitModels[u].mean_std_X = []
      unitModels[u].max_X = []
      for b in range(unitModels[u].nrBiomk):
        unitModels[u].mean_std_X.append([0, 1])
        unitModels[u].max_X.append(1)

      unitModels[u].xsUpdateSetLimits()


  def estimBiomkTraj(self, unitModels, disModels, iterNr):

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
      if iterNr == 0:
        self.unitModels[u].updateXvals(predScoresCurrUSX[u], XdisSX)

      self.unitModels[u].priors = self.params['priorsUnitModels'][u]
      # self.unitModels[u].Set_penalty(2)
      # self.unitModels[u].estimTrajParams(Niterat = 70)
      self.unitModels[u].estimTrajParams()

      # fig = self.unitModels[u].plotter.plotTraj(self.unitModels[u])
      # fig2 = self.unitModels[u].plotter.plotCompWithTrueParams(self.unitModels[u], replaceFig=False)

      # fig.savefig('%s/allTraj%d0_%s.png' % (self.outFolder, i + 1, self.expName))


  def estimTrajWithinDisModel(self, unitModels, disModels):
    """ estimates trajectory parameters within the disease specific models:

        We constrain Dis1-unit-traj lower/upper bounds to be [0-1]
    """

    XshiftedScaledDBSX = [0 for _ in range(self.nrDis)]
    X_arrayScaledDB = [0 for _ in range(self.nrDis)]
    informPriorTrajDisModels = [True, True] # set informative priors only for the first disease
    for d in range(self.nrDis):
      XshiftedScaledDBSX[d], _, _, X_arrayScaledDB[d] = disModels[d].getData()

      # update each unit-traj independently
      for u in range(self.nrFuncUnits):
        Y_arrayCurDis, indFiltToMisCurDisB = unitModels[u].filterLongArray(unitModels[u].Y_array,
          unitModels[u].N_obs_per_sub, self.binMaskSubjForEachDisD[d],unitModels[u].visitIndices)

        for b in range(len(Y_arrayCurDis)):
          assert Y_arrayCurDis[b].shape[0] == indFiltToMisCurDisB[b].shape[0]
          print(Y_arrayCurDis[b].shape)

        # build function that within a disease model takes a unit traj, and predicts lik of given
        # unit-traj params in corresponding unitModel.
        likJDMobjFunc = lambda paramsCurrTraj: self.ssdJDMOneUnitTraj(disModels[d], unitModels[u],
          X_arrayScaledDB[d], [paramsCurrTraj, None], u, d, Y_arrayCurDis, indFiltToMisCurDisB, informPriorTrajDisModels[d])
        initParams, initVariance = disModels[d].unpack_parameters(disModels[d].parameters[u])

        resStruct = scipy.optimize.minimize(likJDMobjFunc, initParams, method='Nelder-Mead',
          options={'disp': True, 'maxiter':1000})

        print('resStruct', resStruct)

        # print('finalLik', likJDMobjFunc(resStruct.x))
        disModels[d].parameters[u] = [resStruct.x, initVariance]


  def ssdJDMOneUnitTraj(self, disModel, unitModel, X_arrayScaledDisB, params, unitNr, disNr, Y_arrayCurDis,
                        indFiltToMisCurDisB, informPriorTraj, breakPoint=False):
    """
    function computes the likelihood of params of one traj within one disModel.

    :param disModel:
    :param unitModel:
    :param X_arrayScaledDisB: X_array values (scaled) for patients withing current disease
    :param params: parameters for one unit-trajectory to optimise, within a diseModel
    :param unitNr:
    :param disNr:
    :param Y_arrayCurDis:
    :param indFiltToMisCurDisB
    :param informPriorTraj
    :param breakPoint

    :return: log-likelihood
    """

    assert type(X_arrayScaledDisB) == list

    ####### first predict dysScores in disModel for curr functional unit ##########
    paramsAllU = disModel.parameters
    paramsAllU[unitNr] = params
    # print('X_arrayScaledDisB[0]', X_arrayScaledDisB[0])
    predScoresCurrXU = disModel.predictBiomkWithParams(X_arrayScaledDisB[0], paramsAllU)

    if breakPoint:
      print('predScoresCurrXU', predScoresCurrXU)


    #### then sum log-liks of each func unit.

    predX_arrayUnitModel = [0 for b in range(unitModel.nrBiomk)]

    sumSSD =0
    for b in range(unitModel.nrBiomk):
      predScoresCurrBiomk = predScoresCurrXU[:,unitNr].reshape(-1,1)
      predX_arrayUnitModel[b] = unitModel.filterZarray(predScoresCurrBiomk,
        indFiltToMisCurDisB[b])
      assert predX_arrayUnitModel[b].shape[0] == Y_arrayCurDis[b].shape[0]

      if len(Y_arrayCurDis[b]) > 0:
        ssd = unitModel.ssdTrajOneBiomk(unitModel.parameters[b],
                                        predX_arrayUnitModel[b], Y_arrayCurDis[b], b)[0]
        sumSSD += ssd

    # here we also add the prior on the disease trajectories.
    prior_traj = 0
    if informPriorTraj:
      prior_traj = disModel.computePriorTrajOneBiomk(params)

    return sumSSD + prior_traj # sum the likelihood from each trajectory where there is data


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

    # print('dysfuncPredXU[:,0]', dysfuncPredXU[:,0])
    # print('biomkPredXB[:,0]', biomkPredXB[:,0])
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








