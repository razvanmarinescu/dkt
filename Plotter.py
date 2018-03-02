from matplotlib import pyplot as pl
import numpy as np
import sys
import colorsys
from env import *
import copy
import auxFunc

class PlotterJDM:

  def __init__(self, plotTrajParams):
    self.plotTrajParams = plotTrajParams

  def plotTrajDataMarcoFormat(self, X, Y, diag, subShiftsMarcoFormat, model,
    replaceFigMode=True, yLimUseData=False, showConfInt=False, adjustBottomHeight=0.1):
    figSizeInch = (self.plotTrajParams['SubfigTrajWinSize'][0] / 100, self.plotTrajParams['SubfigTrajWinSize'][1] / 100)
    fig = pl.figure(1, figsize=figSizeInch)
    pl.clf()
    nrRows = self.plotTrajParams['nrRows']
    nrCols = self.plotTrajParams['nrCols']

    nrBiomk = len(X)

    minX = np.min([np.min(dpsCurr) for dpsCurr in subShiftsMarcoFormat])
    maxX = np.max([np.max(dpsCurr) for dpsCurr in subShiftsMarcoFormat])

    diagNrs = np.unique(diag)
    nrSubjLong = len(X[0])

    # xs = np.linspace(minX, maxX, 100)
    # dysScoresSF = model.predPopDys(xs)
    # modelPredSB = model.predPop(xs)

    xs = self.plotTrajParams['trueParams']['trueLineSpacedDPSsX']
    dysScoresSF = self.plotTrajParams['trueParams']['trueDysTrajFromDpsXU']
    modelPredSB = self.plotTrajParams['trueParams']['trueTrajPredXB']

    scalingBiomk2B = self.plotTrajParams['trueParams']['scalingBiomk2B']
    modelPredSB = auxFunc.applyInverseScalingToBiomk(modelPredSB, scalingBiomk2B)

    YScaled = [[] for b in range(nrBiomk)]
    for b in range(nrBiomk):
      if len(Y[b][0]) > 0:
        YScaled[b] = [auxFunc.applyInverseScalingToBiomk(ys.reshape(-1,1),
        scalingBiomk2B[:,b].reshape(-1,1)) for ys in Y[b]]
      else:
        YScaled[b] = Y[b]

    # print('xs', xs)
    # print('dysScoresSF', dysScoresSF)
    # print('modelPredSB[:,0]', modelPredSB[:, 0])
    # print(ads)

    lw = 3.0

    # first plot the dysfunctionality biomarkers
    ax = pl.subplot(nrRows, nrCols, 1)
    ax.set_title('dysfunc all')
    moveTicksInside(ax)
    for f in range(model.nrFuncUnits):
      ax.plot(xs, dysScoresSF[:, f], 'k-', linewidth=lw)

    # for each unit, plot all biomarkers against the dysfunctional scores
    for f in range(model.nrFuncUnits):
      ax2 = pl.subplot(nrRows, nrCols, f + 2)
      ax2.set_title('dysfunc %d' % f)
      moveTicksInside(ax2)
      biomkInCurrUnit = np.where(self.plotTrajParams['mapBiomkToFuncUnits'] == f)[0]
      for b in range(len(biomkInCurrUnit)):
        ax2.plot(dysScoresSF[:, f], modelPredSB[:, biomkInCurrUnit[b]], 'k-', linewidth=lw)

      ax2.set_xlim((0, 1))

    for b in range(nrBiomk):
      ax = pl.subplot(nrRows, nrCols, b + model.nrFuncUnits + 2)
      ax.set_title('biomk %d func %d' % (b, self.plotTrajParams['mapBiomkToFuncUnits'][b]))
      moveTicksInside(ax)

      pl.plot(xs, modelPredSB[:, b], 'k-', linewidth=lw)  # label='sigmoid traj %d' % b
      if showConfInt:
        pass
        # pl.fill(np.concatenate([xs, xs[::-1]]), np.concatenate([fsCurr - 1.9600 * stdDevs[b],
        #   (fsCurr + 1.9600 * stdDevs[b])[::-1]]), alpha=.3, fc='b', ec='None')
        # label='conf interval (1.96*std)')

      # print(xs[50:60], fsCurr[50:60], thetas[b,:])
      # print(asda)


      ############# spagetti plot subjects ######################
      counterDiagLegend = dict(zip(diagNrs, [0 for x in range(diagNrs.shape[0])]))
      for s in range(nrSubjLong):
        labelCurr = None
        # print(diag[s])
        # print(counterDiagLegend[diag[s]])
        # print(self.plotTrajParams['diagLabels'])
        if counterDiagLegend[diag[s]] == 0:
          labelCurr = self.plotTrajParams['diagLabels'][diag[s]]
          counterDiagLegend[diag[s]] += 1

        # print('X[b][s]', X[b][s])
        # print('X[b][s] + subShiftsMarcoFormat[s]', X[b][s] + subShiftsMarcoFormat[s])
        # print('Y[b][s]', Y[b][s])
        pl.plot(X[b][s] + subShiftsMarcoFormat[s], YScaled[b][s],
          c=self.plotTrajParams['diagColors'][diag[s]],
          label=labelCurr, alpha=0.5)

      pl.xlim(np.min(minX), np.max(maxX))

      minY = np.min([np.min(modelPredSB[:, b])] + [np.min(dataCurr) for dataCurr in YScaled[b] if len(dataCurr) > 0])
      maxY = np.max([np.max(modelPredSB[:, b])] + [np.max(dataCurr) for dataCurr in YScaled[b] if len(dataCurr) > 0])

      delta = (maxY - minY) / 10
      pl.ylim(minY - delta, maxY + delta)

    fs = 15

    fig.text(0.02, 0.6, 'Z-score of biomarker', rotation='vertical', fontsize=fs)
    fig.text(0.4, 0.052, 'disease progression score', fontsize=fs)

    # adjustCurrFig(self.plotTrajParams)
    pl.gcf().subplots_adjust(bottom=adjustBottomHeight, left=0.05, right=0.95)

    # pl.tight_layout(pad=30)
    # fig.suptitle('cluster trajectories', fontsize=20)

    h, axisLabels = ax.get_legend_handles_labels()
    # print(h[2:4], labels[2:4])
    # legend =  pl.legend(handles=h, bbox_to_anchor=self.plotTrajParams['legendPos'], loc='upper center', ncol=plotTrajParams['legendCols'])
    # legend = pl.legend(handles=h, loc='upper center', ncol=self.plotTrajParams['legendCols'])

    legend = pl.figlegend(h, axisLabels, loc='lower center', ncol=self.plotTrajParams['legendCols'], labelspacing=0.)
    # set the linewidth of each legend object
    # for i,legobj in enumerate(legend.legendHandles):
    #   legobj.set_linewidth(4.0)
    #   legobj.set_color(self.plotTrajParams['diagColors'][diagNrs[i]])

    # mng = pl.get_current_fig_manager()
    # print(self.plotTrajParams['SubfigClustMaxWinSize'])
    # print(adsds)
    # mng.resize(*self.plotTrajParams['SubfigClustMaxWinSize'])

    if replaceFigMode:
      fig.show()
    else:
      pl.show()

    # print("Plotting results .... ")
    pl.pause(0.05)
    return fig


class PlotterGP:

  def __init__(self, plotTrajParams):
    self.plotTrajParams = plotTrajParams

  def plotTraj(self, gpModel, replaceFig=True, legendExtraPlot=False):
    nrBiomk = gpModel.N_biom
    # Plot method

    font = {'family': 'normal',
      'size': 13}

    import matplotlib
    matplotlib.rc('font', **font)

    figSizeInch = (self.plotTrajParams['SubfigTrajWinSize'][0] / 100, self.plotTrajParams['SubfigTrajWinSize'][1] / 100)
    fig = pl.figure(1, figsize = figSizeInch)
    pl.clf()
    fig.show()

    print(self.plotTrajParams['labels'])
    print('nrBiomk', nrBiomk)
    # print(asda)

    diagNrs = np.unique(self.plotTrajParams['diag'])
    nrDiags = diagNrs.shape[0]

    min_yB, max_yB = gpModel.getMinMaxY_B(extraDelta=0)
    deltaB = [(max_yB[b] - min_yB[b]) * 0.2 for b in range(nrBiomk)]

    # nrRows = self.plotTrajParams['nrRows']
    # nrCols = self.plotTrajParams['nrCols']

    nrRows = 2
    nrCols = 3

    newX = gpModel.getXsMinMaxRange()
    predBiomksYscaledXB = gpModel.predictBiomk(newX)

    Xshifted, X, Y = gpModel.getData()

    for b in range(nrBiomk):
      ax = pl.subplot(nrRows, nrCols, b + 1)
      pl.title(self.plotTrajParams['labels'][b])

      # plot traj samples
      nrSamples = 100
      trajSamplesXS = gpModel.sampleTrajPost(newX, b, nrSamples)
      for i in range(nrSamples):
        ax.plot(newX, trajSamplesXS[:,i], lw = 0.05,
          color = 'red', alpha=1)

      # plot subject data
      diagCounters = dict([(k,0) for k in self.plotTrajParams['diagLabels'].keys()])
      for sub in range(gpModel.N_samples):
        diagCurrSubj = self.plotTrajParams['diag'][sub]
        currLabel = None
        if diagCounters[diagCurrSubj] == 0:
          currLabel = self.plotTrajParams['diagLabels'][diagCurrSubj]
          pass

        ax.plot(Xshifted[b][sub], Y[b][sub],
          color=self.plotTrajParams['diagColors'][diagCurrSubj], lw=0.5)

        ax.scatter(Xshifted[b][sub], Y[b][sub],
          marker=self.plotTrajParams['diagScatterMarkers'][diagCurrSubj],
          color=self.plotTrajParams['diagColors'][diagCurrSubj], lw=2.5,
          label=currLabel)

        diagCounters[diagCurrSubj] += 1

      ax.plot(newX,predBiomksYscaledXB[:,b], lw=2, color='black', label='estimated trajectory')

      # ax.plot(np.array([np.min(newX), np.max(newX)]), [min_yB[b], max_yB[b]],
      #   color=(0.5,0.5,0.5), lw=2)

      ax.set_ylim([min_yB[b]-deltaB[b], max_yB[b]+deltaB[b]])

      # ax.legend(ncol=1)

    if legendExtraPlot:
      ax = pl.subplot(nrRows, nrCols, nrBiomk+1)
      pl.axis('off')

      # plot subject data
      diagCounters = dict([(k,0) for k in self.plotTrajParams['diagLabels'].keys()])
      for sub in range(gpModel.N_samples):
        diagCurrSubj = self.plotTrajParams['diag'][sub]
        currLabel = None
        if diagCounters[diagCurrSubj] == 0:
          currLabel = self.plotTrajParams['diagLabels'][diagCurrSubj]
          pass

        ax.plot(Xshifted[b][sub], Y[b][sub],
          color=self.plotTrajParams['diagColors'][diagCurrSubj], lw=0.5)

        ax.scatter(Xshifted[b][sub], Y[b][sub],
          marker=self.plotTrajParams['diagScatterMarkers'][diagCurrSubj],
          color=self.plotTrajParams['diagColors'][diagCurrSubj], lw=2.5,
          label=currLabel)

        diagCounters[diagCurrSubj] += 1

      ax.plot(newX,predBiomksYscaledXB[:,b], lw=2, color='black', label='estimated trajectory')


      ax.set_ylim([1000, 10000])

      ax.legend(ncol=1, loc='upper left')


    fig.text(0.5, 0.04, 'Dysfunctionality Score', ha='center')
    fig.text(0.08, 0.5, 'Biomarker Value (normalised)', va='center', rotation='vertical')


    if replaceFig:
      fig.show()
    else:
      pl.show()
    pl.pause(0.05)

    return fig

  def plotCompWithTrueParams(self, gpModel, replaceFig=True, subjStagesEstim=None):

    nrBiomk = gpModel.N_biom
    figSizeInch = (self.plotTrajParams['SubfigTrajWinSize'][0] / 100, self.plotTrajParams['SubfigTrajWinSize'][1] / 100)
    fig = pl.figure(2, figsize = figSizeInch)
    pl.clf()

    nrRows = self.plotTrajParams['nrRows']
    nrCols = self.plotTrajParams['nrCols']


    if (nrBiomk + 3) > nrRows * nrCols:
      print('nrRows', nrRows)
      print('nrCols', nrCols)
      print('nrBiomk', nrBiomk)
      print('labels', self.plotTrajParams['labels'])
      raise ValueError('too few nrRows and nrCols')

    ######### compare subject shifts ##########

    subShiftsTrueMarcoFormatS = self.plotTrajParams['trueParams']['subShiftsTrueMarcoFormatS']
    # estimShifts = gpModel.params_time_shift[0,:]

    nrSubjLong = len(gpModel.X[0])


    if subjStagesEstim is None:
      subjStagesEstim = gpModel.getSubShiftsLong()

    ax = pl.subplot(nrRows, nrCols, 1)

    diagNrs = np.unique(self.plotTrajParams['diag'])
    nrDiags = diagNrs.shape[0]
    for d in range(nrDiags):
      pl.scatter(subjStagesEstim[self.plotTrajParams['diag'] == diagNrs[d]],
        subShiftsTrueMarcoFormatS[self.plotTrajParams['diag'] == diagNrs[d]],
        c=self.plotTrajParams['diagColors'][diagNrs[d]],
        label=self.plotTrajParams['diagLabels'][diagNrs[d]])


    pl.title('Subject shifts')
    pl.xlabel('estimated shifts')
    pl.ylabel('true shifts')
    ax.set_ylim([np.min(subShiftsTrueMarcoFormatS), np.max(subShiftsTrueMarcoFormatS)])
    ax.legend(ncol=1)

    ######### compare all trajectories ##########

    newXTraj = gpModel.getXsMinMaxRange()
    newXTrajScaledZeroOne = (newXTraj - np.min(newXTraj)) / (np.max(newXTraj) - np.min(newXTraj))
    trueXsTrajX = self.plotTrajParams['trueParams']['trueXsTrajX']
    trueXsScaledZeroOne = (trueXsTrajX - np.min(trueXsTrajX)) / (np.max(trueXsTrajX) - np.min(trueXsTrajX))

    predTrajScaledXB = gpModel.predictBiomk(newXTraj)
    trueTrajXB = self.plotTrajParams['trueParams']['trueTrajXB']
    trueTrajScaledXB = copy.deepcopy(trueTrajXB)

    if self.plotTrajParams['yNormMode'] == 'zScoreTraj':
      idxZscore = [np.where(np.in1d(self.plotTrajParams['diag'], [CTL, CTL2]))[0]
        for b in range(nrBiomk)]
    elif self.plotTrajParams['yNormMode'] == 'zScoreEarlyStageTraj':
      # take only the first 10 percentile at earliest stages and set that
      # as the true control group
      idxZscore = [0 for b in range(nrBiomk)]
      for b in range(nrBiomk):
        sortedIndx = np.argsort(subjStagesEstim)
        sortedIndx =np.array([i for i in sortedIndx if gpModel.Y[b][i].shape[0] > 0])
        percPoint = 0.1
        if sortedIndx.shape[0] < 50:
          # increase percPoint due to not enough samples
          # (with 10% of subj then at most 5 subj will be used to estimate the Z-scores)
          percPoint = 0.2

        idxZscore[b] = sortedIndx[:int(sortedIndx.shape[0] * percPoint)]

        if len(idxZscore) == 0:
          idxZscore[b] = list(range(int(subjStagesEstim.shape[0]/20)))
    elif self.plotTrajParams['yNormMode'] == 'unscaled':
      pass
    else:
      raise ValueError('plotTrajParams[yNormMode] should be either unscaled, zScoreTraj or zScoreEarlyStageTraj')


    if self.plotTrajParams['yNormMode'] in ['zScoreTraj', 'zScoreEarlyStageTraj']:
      meanCtlB = np.zeros(gpModel.N_biom)
      stdCtlB = np.zeros(gpModel.N_biom)

      for b in range(gpModel.N_biom):
        print('idxZscore', idxZscore)
        yValsOfCtl = [gpModel.Y[b][s] for s in idxZscore[b]]
        yValsOfCtl = [l2 for l in yValsOfCtl for l2 in l]
        meanCtlB[b] = np.mean(yValsOfCtl)
        stdCtlB[b] = np.std(yValsOfCtl)

        predTrajScaledXB[:, b] = gpModel.applyGivenScalingY(predTrajScaledXB[:, b], meanCtlB[b], stdCtlB[b])
        trueTrajScaledXB[:, b] = gpModel.applyGivenScalingY(trueTrajXB[:, b], meanCtlB[b], stdCtlB[b])

      yMinPred = np.min(predTrajScaledXB, axis = (0, 1))
      yMinTrue = np.min(trueTrajScaledXB, axis = (0, 1))
      yMinAll = np.min([yMinPred, yMinTrue])

      yMaxPred = np.max(predTrajScaledXB, axis = (0, 1))
      yMaxTrue = np.max(trueTrajScaledXB, axis = (0, 1))
      yMaxAll = np.max([yMaxPred, yMaxTrue])

      min_yB = yMinAll * np.ones(nrBiomk)
      max_yB = yMaxAll * np.ones(nrBiomk)

      deltaB = deltaB = [(max_yB[b] - min_yB[b]) * 0.2 for b in range(nrBiomk)]
      min_yB = min_yB - deltaB
      max_yB = max_yB + deltaB

    elif self.plotTrajParams['yNormMode'] == 'unscaled':
      scaledYarrayB = [gpModel.applyScalingY(gpModel.Y_array[b], b) for b in range(nrBiomk)]
      min_yB, max_yB = gpModel.getMinMaxY_B(extraDelta=0.2)

      yMinAll = np.min(min_yB)
      yMaxAll = np.max(max_yB)
    else:
      raise ValueError('plotTrajParams[yNormMode] should be either unscaled, zScoreTraj or zScoreEarlyStageTraj')


    # print('predTrajScaledXB[:, 0]', predTrajScaledXB[:, 0])
    # print('trueTrajScaledXB[:, 0]', trueTrajScaledXB[:, 0])
    # print('yMinAll', yMinAll)
    # print('yMaxAll', yMaxAll)
    # print('yMinPred', yMinPred)
    # print('yMinTrue', yMinTrue)
    # print('meanCtlB', meanCtlB)
    # print('stdCtlB', stdCtlB)
    # print('idxZscore', idxZscore)
    # print('yValsOfCtl', yValsOfCtl)
    # print('gpModel.Y', gpModel.Y[-1])
    # print(asda)

    if self.plotTrajParams['allTrajOverlap']:
      ax2 = pl.subplot(nrRows, nrCols, 2)
      pl.title('all trajectories')
      ax2.set_ylim([yMinAll, yMaxAll])
      for b in range(gpModel.N_biom):

        ax2.plot(newXTrajScaledZeroOne, predTrajScaledXB[:, b], '-',lw=2
          ,c=self.plotTrajParams['colorsTraj'][b], label=self.plotTrajParams['labels'][b])
        print('trueXsScaledZeroOne trueTrajXB', trueXsScaledZeroOne.shape, trueTrajXB.shape)
        ax2.plot(trueXsScaledZeroOne, trueTrajScaledXB[:,b], '--', lw=2
          ,c=self.plotTrajParams['colorsTraj'][b])

      # ax2.legend(loc='lower right',ncol=4)
      nrPlotsSoFar = 2
    else:
      ax2 = pl.subplot(nrRows, nrCols, 2)
      pl.title('all estimated trajectories')
      ax2.set_ylim([yMinAll, yMaxAll])
      for b in range(gpModel.N_biom):
        print('colTraj', len(self.plotTrajParams['colorsTraj']))
        print('labels', len(self.plotTrajParams['labels']))
        print(b)
        ax2.plot(newXTrajScaledZeroOne, predTrajScaledXB[:, b], '-',lw=2
          ,c=self.plotTrajParams['colorsTraj'][b], label=self.plotTrajParams['labels'][b])

      # ax2.legend(loc='lower right',ncol=4)

      ax3 = pl.subplot(nrRows, nrCols, 3)
      pl.title('all true trajectories')
      ax3.set_ylim([yMinAll, yMaxAll])
      for b in range(gpModel.N_biom):

        ax3.plot(trueXsScaledZeroOne, trueTrajScaledXB[:,b], '--', lw=2
          ,c=self.plotTrajParams['colorsTraj'][b])

      # ax3.legend(loc='lower right',ncol=4)

      nrPlotsSoFar = 3

    ######### compare biomarker trajectories one by one ##########

    for b in range(gpModel.N_biom):
      ax4 = pl.subplot(nrRows, nrCols, b+nrPlotsSoFar+1)
      pl.title(self.plotTrajParams['labels'][b])

      ax4.plot(newXTrajScaledZeroOne, predTrajScaledXB[:, b], '-',lw=2,
        c=self.plotTrajParams['colorsTraj'][b], label='estimated')

      ax4.plot(trueXsScaledZeroOne, trueTrajScaledXB[:,b], '--', lw=2,
        c=self.plotTrajParams['colorsTraj'][b], label='true')

      ax4.set_ylim([min_yB[b], max_yB[b]])
      ax4.legend(loc='lower right')

    if replaceFig:
      fig.show()
    else:
      pl.show()
    pl.pause(0.05)

    # print(ads)
    return fig

  def plotTrajSameSpace(self, gpModel, replaceFig=True, subjStagesEstim=None):

    # nrBiomk = gpModel.N_biom
    nrBiomk = 6
    figSizeInch = (6, 4)
    fig = pl.figure(2, figsize = figSizeInch)
    pl.clf()

    font = {'family': 'normal',
      'size': 11.8}

    import matplotlib
    matplotlib.rc('font', **font)

    ######### compare subject shifts ##########

    colorsTraj = [colorsys.hsv_to_rgb(hue, 1, 1) for hue in
      np.linspace(0, 1, num=nrBiomk, endpoint=False)]

    nrSubjLong = len(gpModel.X[0])


    if subjStagesEstim is None:
      subjStagesEstim = gpModel.getSubShiftsLong()

    ######### compare all trajectories ##########

    newXTraj = gpModel.getXsMinMaxRange()
    # newXTrajScaledZeroOne = (newXTraj - np.min(newXTraj)) / (np.max(newXTraj) - np.min(newXTraj))

    # Xshifted, X, Y = gpModel.getData()
    # XshiftedCtl = Xshifted[0][s] if s in np.where()

    predTrajScaledXB = gpModel.predictBiomk(newXTraj)

    if self.plotTrajParams['yNormMode'] == 'zScoreTraj':
      idxZscore = [np.where(np.in1d(self.plotTrajParams['diag'], [CTL, CTL2]))[0]
        for b in range(nrBiomk)]
    elif self.plotTrajParams['yNormMode'] == 'zScoreEarlyStageTraj':
      # take only the first 10 percentile at earliest stages and set that
      # as the true control group
      idxZscore = [0 for b in range(nrBiomk)]
      for b in range(nrBiomk):
        sortedIndx = np.argsort(subjStagesEstim)
        sortedIndx =np.array([i for i in sortedIndx if gpModel.Y[b][i].shape[0] > 0])
        percPoint = 0.1
        if sortedIndx.shape[0] < 50:
          # increase percPoint due to not enough samples
          # (with 10% of subj then at most 5 subj will be used to estimate the Z-scores)
          percPoint = 0.2

        idxZscore[b] = sortedIndx[:int(sortedIndx.shape[0] * percPoint)]

        if len(idxZscore) == 0:
          idxZscore[b] = list(range(int(subjStagesEstim.shape[0]/20)))
    elif self.plotTrajParams['yNormMode'] == 'unscaled':
      pass
    else:
      raise ValueError('plotTrajParams[yNormMode] should be either unscaled, zScoreTraj or zScoreEarlyStageTraj')


    if self.plotTrajParams['yNormMode'] in ['zScoreTraj', 'zScoreEarlyStageTraj']:
      meanCtlB = np.zeros(nrBiomk)
      stdCtlB = np.zeros(nrBiomk)

      for b in range(nrBiomk):
        print('idxZscore', idxZscore)
        yValsOfCtl = [gpModel.Y[b][s] for s in idxZscore[b]]
        yValsOfCtl = [l2 for l in yValsOfCtl for l2 in l]
        meanCtlB[b] = np.mean(yValsOfCtl)
        stdCtlB[b] = np.std(yValsOfCtl)

        predTrajScaledXB[:, b] = gpModel.applyGivenScalingY(predTrajScaledXB[:, b], meanCtlB[b], stdCtlB[b])

      yMinPred = np.min(predTrajScaledXB, axis = (0, 1))
      yMinAll = np.min([yMinPred])

      yMaxPred = np.max(predTrajScaledXB, axis = (0, 1))
      yMaxAll = np.max([yMaxPred])

      min_yB = yMinAll * np.ones(nrBiomk)
      max_yB = yMaxAll * np.ones(nrBiomk)

      deltaB = deltaB = [(max_yB[b] - min_yB[b]) * 0.2 for b in range(nrBiomk)]
      min_yB = min_yB - deltaB
      max_yB = max_yB + deltaB

    elif self.plotTrajParams['yNormMode'] == 'unscaled':
      scaledYarrayB = [gpModel.applyScalingY(gpModel.Y_array[b], b) for b in range(nrBiomk)]
      min_yB, max_yB = gpModel.getMinMaxY_B(extraDelta=0.2)

      yMinAll = np.min(min_yB)
      yMaxAll = np.max(max_yB)
    else:
      raise ValueError('plotTrajParams[yNormMode] should be either unscaled, zScoreTraj or zScoreEarlyStageTraj')




    # pl.gca().set_ylim([yMinAll, yMaxAll])
    for b in range(nrBiomk):
      pl.plot(newXTraj, predTrajScaledXB[:, b], '-',lw=2
        ,c=colorsTraj[b], label=self.plotTrajParams['labels'][b])

    pl.gcf().subplots_adjust(bottom=0.14)

    pl.xlabel('Disease Progression (months)')
    pl.ylabel('Dysfunctionality Scores (normalised)')

    pl.legend()

    if replaceFig:
      fig.show()
    else:
      pl.show()
    pl.pause(2)

    # print(ads)
    return fig

  def plotTrajInDisSpace(self, xsTrajX, predTrajXB, trajSamplesBXS,
      XsubjBSX, YsubjBSX, diagS, XsubjValidBSX, YsubjValidBSX, diagValidS, labels, replaceFig=True):
    """
    plot biomarker traj and subject data in disease space. function doesn't do any scaling of xs or ys

    :param xsTrajX:
    :param predTrajXB:
    :param trajSamplesBXS:
    :param XsubjBSX:
    :param YsubjBSX:
    :param diagS:
    :param XsubjValidBSX:
    :param YsubjValidBSX:
    :param diagValidS:
    :param replaceFig:
    :return:
    """

    font = {'family': 'normal',
      'size': 13}

    import matplotlib
    matplotlib.rc('font', **font)

    # Plot method
    figSizeInch = (self.plotTrajParams['SubfigTrajWinSize'][0] / 100, self.plotTrajParams['SubfigTrajWinSize'][1] / 100)
    fig = pl.figure(1, figsize = figSizeInch)
    pl.clf()
    fig.show()

    diagNrs = np.unique(self.plotTrajParams['diag'])
    nrDiags = diagNrs.shape[0]

    nrBiomk = len(XsubjBSX)
    # nrBiomk = 18
    nrRows, nrCols = auxFunc.findOptimalRowsCols(nrBiomk)

    nrSamples = trajSamplesBXS.shape[2]

    min_yB = np.zeros(nrBiomk)
    max_yB = np.zeros(nrBiomk)
    for b in range(nrBiomk):
      # print([np.min(yS) for yS in YsubjBSX[b] if len(yS) > 0])
      # print([np.min(predTrajXB[:,b])])
      # print([np.min(yS) for yS in YsubjBSX[b] if len(yS) > 0] + [np.min(predTrajXB[:,b])])
      listsMin = [np.min(yS) for yS in YsubjBSX[b] if len(yS) > 0] + [np.min(predTrajXB[:,b])]
      listsMax = [np.max(yS) for yS in YsubjBSX[b] if len(yS) > 0] + [np.max(predTrajXB[:,b])]
      if YsubjValidBSX is not None:
        listsMin += [np.min(yS) for yS in YsubjValidBSX[b] if len(yS) > 0]
        listsMax += [np.max(yS) for yS in YsubjValidBSX[b] if len(yS) > 0]

      min_yB[b] = np.min(listsMin)
      max_yB[b] = np.max(listsMax)

    deltaB = (max_yB - min_yB)/5

    for b in range(nrBiomk):
      ax = pl.subplot(nrRows, nrCols, b + 1)
      pl.title(labels[b])
      # print('--------------b', b)
      # plot traj samples
      for i in range(nrSamples):
        ax.plot(xsTrajX, trajSamplesBXS[b,:,i], lw = 0.05,
          color = 'red', alpha=1)

      self.plotSubjData(ax, XsubjBSX[b], YsubjBSX[b], diagS, labelExtra = '')
      # print('-------------- validation data')
      if XsubjValidBSX is not None:
        self.plotSubjData(ax, XsubjValidBSX[b], YsubjValidBSX[b], diagValidS, labelExtra = '')

      ax.plot(xsTrajX,predTrajXB[:,b],
        lw=2, color='black', label='estim. traj.')

      # ax.plot([np.min(xsTrajX), np.max(xsTrajX)], [min_yB[b], max_yB[b]],
      #   color=(0.5,0.5,0.5), lw=2)

      ax.set_ylim([min_yB[b]-deltaB[b], max_yB[b]+deltaB[b]])

      pl.legend(ncol=1,fontsize=12)

    # pl.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)


    fig.text(0.5, 0.04, 'Disease Progression (months)', ha='center')
    fig.text(0.08, 0.5, 'Biomarker Value (normalised)', va='center', rotation='vertical')


    if replaceFig:
      fig.show()
    else:
      pl.show()
    pl.pause(0.05)

    return fig


  def plotTrajInDisSpaceOverlap(self, xsTrajX, predTrajXB, trajSamplesBXS,params,
    replaceFig=True):
    """
    plot biomarker traj and subject data in disease space. function doesn't do any scaling of xs or ys

    :param xsTrajX:
    :param predTrajXB:
    :param trajSamplesBXS:
    :param replaceFig:
    :return:
    """

    font = {'family': 'normal',
      'size': 13}

    import matplotlib
    matplotlib.rc('font', **font)

    # Plot method
    figSizeInch = (self.plotTrajParams['SubfigTrajWinSize'][0] / 100, self.plotTrajParams['SubfigTrajWinSize'][1] / 100)
    fig = pl.figure(1, figsize = figSizeInch)
    # pl.subplots(nrows = 3, ncols = 3, sharex = True, sharey = True)
    pl.clf()
    fig.show()

    diagNrs = np.unique(self.plotTrajParams['diag'])
    nrDiags = diagNrs.shape[0]

    # nrBiomk = 18
    nrBiomk = predTrajXB.shape[1]


    nrSamples = trajSamplesBXS.shape[2]

    nrBiomkGroups = 5 # DTI, FDG, MRI, AV45, AV1451
    biomkIndInGr = [[] for g in range(nrBiomkGroups)]
    nrBiomkPerGroup = 6 # Frontal, Parietal, Temporal, Cingulate, Occipital, Hippocampus

    nrRows, nrCols = auxFunc.findOptimalRowsCols(nrBiomkGroups)

    colorsTraj = [colorsys.hsv_to_rgb(hue, 1, 1) for hue in
      np.linspace(0, 1, num=nrBiomkPerGroup, endpoint=False)]


    for g in range(nrBiomkGroups):
      biomkIndInGr[g] = np.array(range(g * nrBiomkPerGroup, (g+1)*nrBiomkPerGroup))

      groupName = params['labels'][biomkIndInGr[g][0]].split(' ')[0]

      print('groupName', groupName)
      print('biomkIndInGr[g]', biomkIndInGr[g])
      print('labels in group', [params['labels'][b] for b in biomkIndInGr[g]])
      # print(asda)

      predTrajXBCurrGr = predTrajXB[:,biomkIndInGr[g]]
      trajSamplesBXSCurrGr = trajSamplesBXS[biomkIndInGr[g],:,:]

      min_yB = np.zeros(nrBiomk)
      max_yB = np.zeros(nrBiomk)
      for b in range(nrBiomkPerGroup):
        listsMin = [np.min(predTrajXBCurrGr[:,b])]
        listsMax = [np.max(predTrajXBCurrGr[:,b])]

        min_yB[b] = np.min(listsMin)
        max_yB[b] = np.max(listsMax)

      deltaB = (max_yB - min_yB)/5

      min_y = np.min(min_yB)
      max_y = np.max(max_yB)

      ax = pl.subplot(nrRows, nrCols, g+1)
      pl.title(groupName)
      # ax.xlabel()

      for b in range(nrBiomkPerGroup):

        for i in range(nrSamples):
          ax.plot(xsTrajX, trajSamplesBXSCurrGr[b,:,i], lw = 0.05,
            color = colorsTraj[b], alpha=0.7)

        currLabel = self.plotTrajParams['labels'][biomkIndInGr[g][b]]
        ax.plot(xsTrajX,predTrajXBCurrGr[:,b],
          lw=2, c=colorsTraj[b], label=currLabel.split(' ')[-1])

        # ax.plot([np.min(xsTrajX), np.max(xsTrajX)], [min_yB[b], max_yB[b]],
        #   color=(0.5,0.5,0.5), lw=2)

        ax.set_ylim([min_yB[b]-deltaB[b], max_yB[b]+deltaB[b]])

      # pl.tight_layout(pad=1, w_pad=0.5, h_pad=1.0)

    axe = pl.subplot(nrRows, nrCols, nrBiomkGroups+1)
    for b in range(nrBiomkPerGroup):

      currLabel = self.plotTrajParams['labels'][biomkIndInGr[0][b]]
      plH = axe.plot(xsTrajX, predTrajXBCurrGr[:, b],
        lw=2, c=colorsTraj[b], label=currLabel.split(' ')[-1], alpha=1)

      axe.set_ylim([1000, 100000])

      pl.legend(loc='upper left')

      pl.axis('off')

      # for group in plH:
      #   for x in group:
      #     x.set_visible(False)

    fig.text(0.55, 0.04, 'Disease Progression (months)', ha='center')
    fig.text(0.08, 0.5, 'Biomarker Value (normalised)', va='center', rotation='vertical')



    if replaceFig:
      fig.show()
    else:
      pl.show()
    pl.pause(0.05)

    return fig


  def plotSubjData(self, ax, XsubjSX, YsubjSX, diag, labelExtra):
    # plot subject data
    nrSubData = len(XsubjSX[0])
    diagCounters = dict([(k, 0) for k in self.plotTrajParams['diagLabels'].keys()])
    for sub in range(len(XsubjSX)):

      diagCurrSubj = diag[sub]
      currLabel = None
      if diagCounters[diagCurrSubj] == 0:
        currLabel = labelExtra + self.plotTrajParams['diagLabels'][diagCurrSubj]
      # print('XsubjSX[sub]', XsubjSX[sub])
      # print('YsubjSX[sub]', YsubjSX[sub])

      ax.plot(XsubjSX[sub], YsubjSX[sub],
              color=self.plotTrajParams['diagColors'][diagCurrSubj], lw=0.5)

      ax.scatter(XsubjSX[sub], YsubjSX[sub],
                 marker=self.plotTrajParams['diagScatterMarkers'][diagCurrSubj],
                 color=self.plotTrajParams['diagColors'][diagCurrSubj], lw=2.5,
                 label=currLabel)

      diagCounters[diagCurrSubj] += 1


  def scatterPlotShifts(self, gpModel, subjStagesEstim):

    figSizeInch = (self.plotTrajParams['SubfigTrajWinSize'][0] / 100, self.plotTrajParams['SubfigTrajWinSize'][1] / 100)
    fig = pl.figure(2, figsize = figSizeInch)
    pl.clf()

    ax = pl.gca()

    subShiftsTrueMarcoFormatS = self.plotTrajParams['trueParams']['subShiftsTrueMarcoFormatS']
    pl.scatter(subjStagesEstim, subShiftsTrueMarcoFormatS[::1], c='b')
    # pl.scatter(meanStagesS, subjStagesEstim, c = 'r')
    pl.title('Subject shifts)' )
    pl.xlabel('estimated shifts')
    pl.ylabel('true shifts')
    ax.set_ylim([np.min(subShiftsTrueMarcoFormatS), np.max(subShiftsTrueMarcoFormatS)])

    fig.show()


    return fig



  def Plot_predictions(self, gpModel, final_predSX, Xrange, names=[]):
    scaling = gpModel.mean_std_X[0][1] * gpModel.max_X[0]
    for i in range(len(final_predSX)):
      valid_indices = np.where(np.array(final_predSX[i]) != 0)
      print('predictions[i]', final_predSX[i].shape)
      print('Xrange', Xrange.shape)
      assert final_predSX[i].shape[0] == Xrange.shape[0]
      # print('Xrange[valid_indices]', Xrange[valid_indices])
      # print('Xrange[valid_indices] * scaling', Xrange[valid_indices] * scaling)
      # print('Xrange[valid_indices] * scaling + gpModel.mean_std_X[0][0]', Xrange[valid_indices] * scaling + gpModel.mean_std_X[0][0])
      # print('np.array(predictions[i])[valid_indices]', np.array(final_predSX[i])[valid_indices])
      pl.plot(Xrange[valid_indices] * scaling + gpModel.mean_std_X[0][0], np.array(final_predSX[i])[valid_indices])
      if len(names) > 0:
        print(np.max(final_predSX[i]))
        print(final_predSX[i])
        print(final_predSX[i] == np.max(final_predSX[i]))
        max = np.int(np.where(final_predSX[i] == np.max(final_predSX[i]))[0])
        pl.annotate(names[i], xy=(Xrange[max] * scaling + gpModel.mean_std_X[0][0], final_predSX[i][max]))
    pl.show()

def adjustCurrFig(plotTrajParams):
  fig = pl.gcf()
  # fig.set_size_inches(180/fig.dpi, 100/fig.dpi)

  mng = pl.get_current_fig_manager()
  if plotTrajParams['agg']:  # if only printing images
    pass
  else:
    maxSize = mng.window.maxsize()
    maxSize = (maxSize[0] / 2.1, maxSize[1] / 1.1)
    # print(maxSize)
    mng.resize(*maxSize)

    # mng.window.SetPosition((500, 0))
    mng.window.wm_geometry("+200+50")

  # pl.tight_layout()
  pl.gcf().subplots_adjust(bottom=0.25)

  # pl.tight_layout(pad=50, w_pad=25, h_pad=25)

def moveTicksInside(ax):
  ax.tick_params(axis='y', direction='in', pad=-30)
  ax.tick_params(axis='x', direction='in', pad=-15)