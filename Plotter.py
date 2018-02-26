from matplotlib import pyplot as pl
import numpy as np
import sys
import colorsys
from env import *

class PlotterJDM:

  def __init__(self, plotTrajParams):
    self.plotTrajParams = plotTrajParams

  #
  # def plotTrajData(self, longData, longDiag, longDPS, model,
  #   replaceFigMode=True, yLimUseData=False,showConfInt=False, adjustBottomHeight=0.1):
  #   figSizeInch = (self.plotTrajParams['SubfigTrajWinSize'][0] / 100, self.plotTrajParams['SubfigTrajWinSize'][1] / 100)
  #   fig = pl.figure(1, figsize=figSizeInch)
  #   pl.clf()
  #   nrRows = self.plotTrajParams['nrRows']
  #   nrCols = self.plotTrajParams['nrCols']
  #
  #   nrBiomk = longData[0].shape[1]
  #
  #   minX = np.min([np.min(dpsCurr) for dpsCurr in longDPS])
  #   maxX = np.max([np.max(dpsCurr) for dpsCurr in longDPS])
  #
  #   xs = np.linspace(minX, maxX, 100)
  #   diagNrs = self.plotTrajParams['diagNrs']
  #
  #   nrSubjLong = len(longData)
  #
  #   dysScoresSF = model.predPopDys(xs)
  #   modelPredSB = model.predPop(xs)
  #
  #   print('xs', xs)
  #   print('modelPredSB[:,0]', modelPredSB[:,0])
  #
  #   lw = 3.0
  #
  #   # first plot the dysfunctionality biomarkers
  #   ax = pl.subplot(nrRows, nrCols, 1)
  #   ax.set_title('dysfunc all')
  #   moveTicksInside(ax)
  #   for f in range(model.nrFuncUnits):
  #     ax.plot(xs, dysScoresSF[:,f], 'k-', linewidth=lw)
  #
  #   # for each unit, plot all biomarkers against the dysfunctional scores
  #   for f in range(model.nrFuncUnits):
  #     ax2 = pl.subplot(nrRows, nrCols, f+2)
  #     ax2.set_title('dysfunc %d' % f)
  #     moveTicksInside(ax2)
  #     biomkInCurrUnit = np.where(self.plotTrajParams['mapBiomkToFuncUnits'] == f)[0]
  #     for b in range(len(biomkInCurrUnit)):
  #       ax2.plot(dysScoresSF[:,f], modelPredSB[:,biomkInCurrUnit[b]], 'k-', linewidth=lw)
  #
  #     ax2.set_xlim((0,1))
  #
  #   for b in range(nrBiomk):
  #     ax = pl.subplot(nrRows, nrCols, b + model.nrFuncUnits + 2)
  #     ax.set_title('biomk %d func %d' % (b, self.plotTrajParams['mapBiomkToFuncUnits'][b]))
  #     moveTicksInside(ax)
  #
  #     pl.plot(xs, modelPredSB[:,b], 'k-', linewidth=lw)  # label='sigmoid traj %d' % b
  #     if showConfInt:
  #       pass
  #       # pl.fill(np.concatenate([xs, xs[::-1]]), np.concatenate([fsCurr - 1.9600 * stdDevs[b],
  #       #   (fsCurr + 1.9600 * stdDevs[b])[::-1]]), alpha=.3, fc='b', ec='None')
  #       # label='conf interval (1.96*std)')
  #
  #     # print(xs[50:60], fsCurr[50:60], thetas[b,:])
  #     # print(asda)
  #
  #
  #     ############# spagetti plot subjects ######################
  #     counterDiagLegend = dict(zip(diagNrs, [0 for x in range(diagNrs.shape[0])]))
  #     for s in range(nrSubjLong):
  #       labelCurr = None
  #       if counterDiagLegend[longDiag[s]] == 0:
  #         labelCurr = self.plotTrajParams['diagLabels'][longDiag[s]]
  #         counterDiagLegend[longDiag[s]] += 1
  #
  #       # print('longDPS', longDPS)
  #       # print('len(longDPS)', len(longDPS))
  #       # print('longDPS[s].shape', longDPS[s].shape)
  #       # print('longData[s][:, b]', longData[s][:, b].shape)
  #       pl.plot(longDPS[s], longData[s][:, b],
  #         c=self.plotTrajParams['diagColors'][longDiag[s]],
  #         label=labelCurr,alpha=0.5)
  #
  #     pl.xlim(np.min(minX), np.max(maxX))
  #
  #     minY = np.min([np.min(dataCurr[:,b]) for dataCurr in longData])
  #     maxY = np.max([np.max(dataCurr[:,b]) for dataCurr in longData])
  #     delta = (maxY - minY) / 10
  #     pl.ylim(minY - delta, maxY + delta)
  #
  #   fs = 15
  #
  #   fig.text(0.02, 0.6, 'Z-score of biomarker', rotation='vertical', fontsize=fs)
  #   fig.text(0.4, 0.052, 'disease progression score', fontsize=fs)
  #
  #   # adjustCurrFig(self.plotTrajParams)
  #   pl.gcf().subplots_adjust(bottom=adjustBottomHeight, left=0.05, right=0.95)
  #
  #   # pl.tight_layout(pad=30)
  #   # fig.suptitle('cluster trajectories', fontsize=20)
  #
  #   h, axisLabels = ax.get_legend_handles_labels()
  #   # print(h[2:4], labels[2:4])
  #   # legend =  pl.legend(handles=h, bbox_to_anchor=self.plotTrajParams['legendPos'], loc='upper center', ncol=plotTrajParams['legendCols'])
  #   # legend = pl.legend(handles=h, loc='upper center', ncol=self.plotTrajParams['legendCols'])
  #
  #   legend = pl.figlegend(h, axisLabels, loc='lower center', ncol=self.plotTrajParams['legendCols'], labelspacing=0.)
  #   # set the linewidth of each legend object
  #   # for i,legobj in enumerate(legend.legendHandles):
  #   #   legobj.set_linewidth(4.0)
  #   #   legobj.set_color(self.plotTrajParams['diagColors'][diagNrs[i]])
  #
  #   # mng = pl.get_current_fig_manager()
  #   # print(self.plotTrajParams['SubfigClustMaxWinSize'])
  #   # print(adsds)
  #   # mng.resize(*self.plotTrajParams['SubfigClustMaxWinSize'])
  #
  #   if replaceFigMode:
  #     fig.show()
  #   else:
  #     pl.show()
  #
  #   # print("Plotting results .... ")
  #   pl.pause(0.05)
  #   return fig


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

    xs = np.linspace(minX, maxX, 100)
    # diagNrs = self.plotTrajParams['diagNrs']
    diagNrs = np.unique(diag)

    nrSubjLong = len(X[0])

    dysScoresSF = model.predPopDys(xs)
    modelPredSB = model.predPop(xs)

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
        pl.plot(X[b][s] + subShiftsMarcoFormat[s], Y[b][s],
          c=self.plotTrajParams['diagColors'][diag[s]],
          label=labelCurr, alpha=0.5)

      pl.xlim(np.min(minX), np.max(maxX))

      minY = np.min([np.min(modelPredSB[:, b])] + [np.min(dataCurr) for dataCurr in Y[b] if len(dataCurr) > 0])
      maxY = np.max([np.max(modelPredSB[:, b])] + [np.max(dataCurr) for dataCurr in Y[b] if len(dataCurr) > 0])

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

  def plotTraj(self, gpModel, replaceFig=True):
    nrBiomk = gpModel.N_biom
    # Plot method
    newX = np.linspace(gpModel.minX, gpModel.maxX, 30).reshape([30, 1])
    figSizeInch = (self.plotTrajParams['SubfigTrajWinSize'][0] / 100, self.plotTrajParams['SubfigTrajWinSize'][1] / 100)
    fig = pl.figure(1, figsize = figSizeInch)
    pl.clf()
    fig.show()

    diagNrs = np.unique(self.plotTrajParams['diag'])
    nrDiags = diagNrs.shape[0]

    scaledYarrayB = [gpModel.applyScalingY(gpModel.Y_array[b], b) for b in range(nrBiomk)]
    min_yB = [np.min(scaledYarrayB[b].reshape(-1)) for b in range(nrBiomk)]
    max_yB = [np.max(scaledYarrayB[b].reshape(-1)) for b in range(nrBiomk)]
    deltaB = [(max_yB[b] - min_yB[b])/5 for b in range(nrBiomk)]

    # max_y = np.max([np.float(item) for sublist in gpModel.Y_array for item in sublist])
    # min_y = np.min([np.float(item) for sublist in gpModel.Y_array for item in sublist])

    nrRows = self.plotTrajParams['nrRows']
    nrCols = self.plotTrajParams['nrCols']

    predictedBiomksXB = gpModel.predictBiomk(newX)

    # print('nrBiomk', nrBiomk)
    # print(nrRows, nrCols)
    # print(adsa)

    for b in range(nrBiomk):
      ax = pl.subplot(nrRows, nrCols, b + 1)
      pl.title(self.plotTrajParams['labels'][b])

      # plot traj samples
      nrSamples = 100
      newXScaledX, trajSamplesXS = gpModel.sampleBiomkTrajPosterior(newX, b, nrSamples)
      for i in range(nrSamples):
        # ax.plot(gpModel.applyScalingX(newX, b), gpModel.applyScalingY(trajSamplesXS[:,i], b), lw = 0.05,
        #   color = 'red')
        ax.plot(newX, gpModel.applyScalingY(trajSamplesXS[:,i], b), lw = 0.05,
          color = 'red', alpha=1)

      # plot subject data
      diagCounters = dict([(k,0) for k in self.plotTrajParams['diagLabels'].keys()])
      for sub in range(gpModel.N_samples):
        x_data = np.array([gpModel.X_array[b][k][0] for k in range(int(np.sum(gpModel.N_obs_per_sub[b][:sub])),
          np.sum(gpModel.N_obs_per_sub[b][:sub + 1]))])
        y_data = np.array([gpModel.Y_array[b][k][0] for k in range(int(np.sum(gpModel.N_obs_per_sub[b][:sub])),
          np.sum(gpModel.N_obs_per_sub[b][:sub + 1]))])

        # print('x_data', x_data)


        diagCurrSubj = self.plotTrajParams['diag'][sub]
        currLabel = None
        if diagCounters[diagCurrSubj] == 0:
          currLabel = self.plotTrajParams['diagLabels'][diagCurrSubj]
          pass
        # ax.plot(gpModel.applyScalingX(x_data, b),
        #   gpModel.applyScalingY(y_data, b),
        #   color = self.plotTrajParams['diagColors'][diagCurrSubj], lw = 0.5)

        # ax.scatter(gpModel.applyScalingX(x_data, b),
        #   gpModel.applyScalingY(y_data, b),
        #   marker=self.plotTrajParams['diagScatterMarkers'][diagCurrSubj],
        #   color=self.plotTrajParams['diagColors'][diagCurrSubj], lw=2.5,
        #   label=currLabel)

        ax.plot(x_data, gpModel.applyScalingY(y_data, b),
          color=self.plotTrajParams['diagColors'][diagCurrSubj], lw=0.5)

        ax.scatter(x_data, gpModel.applyScalingY(y_data, b),
          marker=self.plotTrajParams['diagScatterMarkers'][diagCurrSubj],
          color=self.plotTrajParams['diagColors'][diagCurrSubj], lw=2.5,
          label=currLabel)


        diagCounters[diagCurrSubj] += 1

        # pl.pause(0.5)

      # plot main traj
      # ax.plot(gpModel.applyScalingX(newX, b),
      #   gpModel.applyScalingY(predictedBiomksXB[:,b], b),
      #   lw = 2, color = 'black', label='estim traj')

      # ax.plot(gpModel.applyScalingX(np.array([gpModel.minX, gpModel.maxX]), b), [min_yB[b], max_yB[b]],
      #   color=(0.5,0.5,0.5), lw=2)

      ax.plot(newX,gpModel.applyScalingY(predictedBiomksXB[:,b], b),
        lw=2, color='black', label='estim traj')

      ax.plot(np.array([gpModel.minX, gpModel.maxX]), [min_yB[b], max_yB[b]],
        color=(0.5,0.5,0.5), lw=2)

      ax.set_ylim([min_yB[b]-deltaB[b], max_yB[b]+deltaB[b]])

      # ax.legend(loc='upper right')

      # print(asda)

      # import pdb
      # pdb.set_trace()

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

    # max_y = np.max([np.float(item) for sublist in gpModel.Y_array for item in sublist])
    # min_y = np.min([np.float(item) for sublist in gpModel.Y_array for item in sublist])

    scaledYarrayB = [gpModel.applyScalingY(gpModel.Y_array[b], b) for b in range(nrBiomk)]
    min_yB = [np.min(scaledYarrayB[b].reshape(-1)) for b in range(nrBiomk)]
    max_yB = [np.max(scaledYarrayB[b].reshape(-1)) for b in range(nrBiomk)]
    deltaB = [(max_yB[b] - min_yB[b])/5 for b in range(nrBiomk)]


    nrRows = self.plotTrajParams['nrRows']
    nrCols = self.plotTrajParams['nrCols']

    ######### compare subject shifts ##########

    subShiftsTrueMarcoFormatS = self.plotTrajParams['trueParams']['subShiftsTrueMarcoFormatS']
    # estimShifts = gpModel.params_time_shift[0,:]
    delta = (gpModel.maxX - gpModel.minX) * 0
    newXs = np.linspace(gpModel.minX - delta, gpModel.maxX + delta, num=100).reshape([100,1])
    newXsScaled = gpModel.applyScalingX(newXs, biomk=0)
    # print('gpModel.X', len(gpModel.X), len(gpModel.X[0]))
    nrSubjToSkip = 10
    nrSubjLong = len(gpModel.X[0])
    xSmall = [b[::nrSubjToSkip] for b in gpModel.X]
    ySmall = [b[::nrSubjToSkip] for b in gpModel.Y]
    # stagingDistSX, meanStagesS = gpModel.StageSubjects(xSmall, ySmall, newXs.reshape(-1, 1))
    stagesExtractedS = np.zeros(nrSubjLong, float)

    # for s in range(nrSubjLong):
    #   x_data = []
    #   for b in range(nrBiomk):
    #     x_data += [gpModel.X_array[b][k][0] for k in
    #   range(int(np.sum(gpModel.N_obs_per_sub[b][:s])),np.sum(gpModel.N_obs_per_sub[b][:s + 1]))]
    #
    #   sortedXdata = np.sort(x_data)
    #   stagesExtractedS[s] = sortedXdata[0]

    # stagesExtractedS = stagesExtractedS[::nrSubjToSkip]
    # stagesExtractedStdS = (stagesExtractedS - np.mean(stagesExtractedS))/np.std(stagesExtractedS)
    # print('stagesExtractedS', stagesExtractedS)
    # print('stagesExtractedStdS', stagesExtractedStdS)
    #
    # subjStagesEstim = gpModel.params_time_shift[0]
    # stagesExtractedStdS2 = (subjStagesEstim - np.mean(subjStagesEstim))/np.std(subjStagesEstim)
    # print('subjStagesEstim', subjStagesEstim)
    # print('stagesExtractedStdS2', stagesExtractedStdS2)
    #
    # meanStagesStdS = (meanStagesS - np.mean(meanStagesS))/np.std(meanStagesS)
    # print('meanStagesStdS', meanStagesStdS)
    # print('diff', meanStagesStdS - stagesExtractedStdS)
    # print('diff2', meanStagesStdS - stagesExtractedStdS2[::nrSubjToSkip])
    # # print(adsa)

    # meanStagesS = np.array(meanStagesS)
    # print('stagingDistSX', len(stagingDistSX), stagingDistSX[0])
    # print('meanStagesS', meanStagesS.shape, meanStagesS)
    # print('maxLikStages', [newXsScaled[np.argmax(stagingDistSX[s])] for s in range(len(stagingDistSX))])
    # print('subShiftsTrueMarcoFormatS', subShiftsTrueMarcoFormatS.shape, subShiftsTrueMarcoFormatS[::nrSubjToSkip])
    sys.stdout.flush()
    # print(ads)
    if subjStagesEstim is None:
      subjStagesEstim = gpModel.params_time_shift[0]

    ax = pl.subplot(nrRows, nrCols, 1)

    diagNrs = np.unique(self.plotTrajParams['diag'])
    nrDiags = diagNrs.shape[0]
    for d in range(nrDiags):
      pl.scatter(subjStagesEstim[self.plotTrajParams['diag'] == diagNrs[d]],
        subShiftsTrueMarcoFormatS[self.plotTrajParams['diag'] == diagNrs[d]],
        c=self.plotTrajParams['diagColors'][diagNrs[d]],
        label=self.plotTrajParams['diagLabels'][diagNrs[d]])


    # pl.scatter(meanStagesS, subjStagesEstim, c = 'r')
    percSubjUsed = int(100/nrSubjToSkip)
    pl.title('Subject shifts (%d %% of subj.)' % percSubjUsed)
    pl.xlabel('estimated shifts')
    pl.ylabel('true shifts')
    ax.set_ylim([np.min(subShiftsTrueMarcoFormatS), np.max(subShiftsTrueMarcoFormatS)])
    ax.legend(ncol=1)

    ######### compare all trajectories ##########
    trueTrajXB = self.plotTrajParams['trueParams']['trueTrajXB']
    newXTraj = np.linspace(gpModel.minX, gpModel.maxX, 30).reshape([30, 1])
    newXTrajScaledZeroOne = (newXTraj - np.min(newXTraj)) / (np.max(newXTraj) - np.min(newXTraj))

    predictedBiomksXB = gpModel.predictBiomk(newXTraj)
    predTrajScaledXB = gpModel.applyScalingYAllBiomk(predictedBiomksXB)

      if self.plotTrajParams['zScoreTraj']:
        meanCtlB = np.zeros(gpModel.N_biom)
        stdCtlB = np.zeros(gpModel.N_biom)
        yValsOfCtl = [gpModel.Y[b][s] for s in range(len(gpModel.Y[b]))
          if np.in1d(self.plotTrajParams['diag'][s], [CTL, CTL2])]
        yValsOfCtl = [l2 for l in yValsOfCtl for l2 in l]
        meanCtlB[b] = np.mean(yValsOfCtl)
        stdCtlB[b] = np.std(yValsOfCtl)

        predTrajScaledXB[:,b] = gpModel.applyGivenScalingY(predTrajScaledXB[:,b],
          meanCtlB[b], stdCtlB[b])


    trueXsTrajX = self.plotTrajParams['trueParams']['trueXsTrajX']

    trueXsScaledZeroOne = (trueXsTrajX - np.min(trueXsTrajX)) / (np.max(trueXsTrajX) - np.min(trueXsTrajX))

    if self.plotTrajParams['zScoreTraj']:
      yMinAll = np.min(predTrajScaledXB, axis = (0, 1))
      yMaxAll = np.max(predTrajScaledXB, axis = (0, 1))
    else:
      yMinAll = np.min(min_yB)
      yMaxAll = np.max(max_yB)


    deltaAll = (yMaxAll - yMinAll) / 5



    if self.plotTrajParams['allTrajOverlap']:
      ax2 = pl.subplot(nrRows, nrCols, 2)
      pl.title('all trajectories')
      ax2.set_ylim([yMinAll - deltaAll, yMaxAll + deltaAll])
      for b in range(gpModel.N_biom):

        ax2.plot(newXTrajScaledZeroOne, predTrajScaledXB[:, b], '-',lw=2
          ,c=self.plotTrajParams['colorsTraj'][b], label=self.plotTrajParams['labels'][b])
        print('trueXsScaledZeroOne trueTrajXB', trueXsScaledZeroOne.shape, trueTrajXB.shape)
        ax2.plot(trueXsScaledZeroOne, trueTrajXB[:,b], '--', lw=2
          ,c=self.plotTrajParams['colorsTraj'][b])

      # ax2.legend(loc='lower right',ncol=4)
      nrPlotsSoFar = 2
    else:
      ax2 = pl.subplot(nrRows, nrCols, 2)
      pl.title('all estimated trajectories')
      ax2.set_ylim([yMinAll - deltaAll, yMaxAll + deltaAll])
      for b in range(gpModel.N_biom):
        print('colTraj', len(self.plotTrajParams['colorsTraj']))
        print('labels', len(self.plotTrajParams['labels']))
        print(b)
        ax2.plot(newXTrajScaledZeroOne, predTrajScaledXB[:, b], '-',lw=2
          ,c=self.plotTrajParams['colorsTraj'][b], label=self.plotTrajParams['labels'][b])

      # ax2.legend(loc='lower right',ncol=4)

      ax3 = pl.subplot(nrRows, nrCols, 3)
      pl.title('all true trajectories')
      ax3.set_ylim([yMinAll - deltaAll, yMaxAll + deltaAll])
      for b in range(gpModel.N_biom):

        ax3.plot(trueXsScaledZeroOne, trueTrajXB[:,b], '--', lw=2
          ,c=self.plotTrajParams['colorsTraj'][b])

      # ax3.legend(loc='lower right',ncol=4)

      nrPlotsSoFar = 3

    ######### compare biomarker trajectories one by one ##########

    for b in range(gpModel.N_biom):
      ax4 = pl.subplot(nrRows, nrCols, b+nrPlotsSoFar+1)
      pl.title(self.plotTrajParams['labels'][b])

      ax4.plot(newXTrajScaledZeroOne,
               gpModel.applyScalingY(predictedBiomksXB[:,b], b), '-',lw=2,
        c=self.plotTrajParams['colorsTraj'][b], label='estimated')

      ax4.plot(trueXsScaledZeroOne, trueTrajXB[:,b], '--', lw=2,
        c=self.plotTrajParams['colorsTraj'][b], label='true')

      ax4.set_ylim([min_yB[b] - deltaB[b], max_yB[b] + deltaB[b]])
      ax4.legend(loc='lower right')

    if replaceFig:
      fig.show()
    else:
      pl.show()
    pl.pause(0.05)

    # print(ads)
    return fig

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