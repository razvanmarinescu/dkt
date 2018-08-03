from matplotlib import pyplot as pl
import numpy as np
import sys
import colorsys
from env import *
import copy
import auxFunc
from abc import ABC, ABCMeta, abstractmethod

class PlotterJDM:

  def __init__(self, plotTrajParams):
    self.plotTrajParams = plotTrajParams




  def plotTrajDataMarcoFormat(self, X, Y, diag, model, trueParamsDis,
    replaceFigMode=True, yLimUseData=False, showConfInt=False, adjustBottomHeight=0.1):
    figSizeInch = (self.plotTrajParams['SubfigTrajWinSize'][0] / 100, self.plotTrajParams['SubfigTrajWinSize'][1] / 100)
    fig = pl.figure(1, figsize=figSizeInch)
    pl.clf()
    nrRows = self.plotTrajParams['nrRows']
    nrCols = self.plotTrajParams['nrCols']

    nrBiomk = len(X)

    xsX = trueParamsDis['xsX']
    ysXU = trueParamsDis['ysXU']
    ysXB = trueParamsDis['ysXB']
    subShiftsMarcoFormat = trueParamsDis['subShiftsS']

    minX = np.min([np.min(dpsCurr) for dpsCurr in subShiftsMarcoFormat])
    maxX = np.max([np.max(dpsCurr) for dpsCurr in subShiftsMarcoFormat])

    diagNrs = np.unique(diag)
    nrSubjLong = len(X[0])

    scalingBiomk2B = trueParamsDis['scalingBiomk2B']
    ysXB = auxFunc.applyInverseScalingToBiomk(ysXB, scalingBiomk2B)

    YScaled = [[] for b in range(nrBiomk)]
    for b in range(nrBiomk):
      if len(Y[b][0]) > 0:
        YScaled[b] = [auxFunc.applyInverseScalingToBiomk(ys.reshape(-1,1),
        scalingBiomk2B[:,b].reshape(-1,1)) for ys in Y[b]]
      else:
        YScaled[b] = Y[b]

    lw = 3.0

    # first plot the dysfunctionality biomarkers
    ax = pl.subplot(nrRows, nrCols, 1)
    ax.set_title('dysfunc all')
    moveTicksInside(ax)
    for f in range(model.nrFuncUnits):
      ax.plot(xsX, ysXU[:, f], '-',color=self.plotTrajParams['colorsTrajUnitsU'][f],
       linewidth=lw)

    # for each unit, plot all biomarkers against the dysfunctional scores
    for f in range(model.nrFuncUnits):
      ax2 = pl.subplot(nrRows, nrCols, f + 2)
      ax2.set_title('dysfunc %d' % f)
      moveTicksInside(ax2)
      biomkInCurrUnit = self.plotTrajParams['biomkInFuncUnit'][f]
      for b in range(len(biomkInCurrUnit)):
        ax2.plot(ysXU[:, f], ysXB[:, biomkInCurrUnit[b]], '-',
          color=self.plotTrajParams['colorsTrajBiomkB'][biomkInCurrUnit[b]],  linewidth=lw)

      ax2.set_xlim((0, 1))

    for b in range(nrBiomk):
      ax = pl.subplot(nrRows, nrCols, b + model.nrFuncUnits + 2)
      ax.set_title('biomk %d func %d' % (b, [f for f in range(model.nrFuncUnits) if b in self.plotTrajParams['biomkInFuncUnit'][f]][0]))
      moveTicksInside(ax)

      ############# spagetti plot subjects ######################
      counterDiagLegend = dict(zip(diagNrs, [0 for x in range(diagNrs.shape[0])]))
      for s in range(nrSubjLong):
        labelCurr = None
        if counterDiagLegend[diag[s]] == 0:
          labelCurr = self.plotTrajParams['diagLabels'][diag[s]]
          counterDiagLegend[diag[s]] += 1

        pl.plot(X[b][s] + subShiftsMarcoFormat[s], YScaled[b][s],
          c=self.plotTrajParams['diagColors'][diag[s]],
          label=labelCurr, alpha=0.5)


      pl.plot(xsX, ysXB[:, b], '-', color=self.plotTrajParams['colorsTrajBiomkB'][b], linewidth=lw)  # label='sigmoid traj %d' % b
      if showConfInt:
        pass

      pl.xlim(np.min(minX), np.max(maxX))

      minY = np.min([np.min(ysXB[:, b])] + [np.min(dataCurr) for dataCurr in YScaled[b] if len(dataCurr) > 0])
      maxY = np.max([np.max(ysXB[:, b])] + [np.max(dataCurr) for dataCurr in YScaled[b] if len(dataCurr) > 0])

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

  def plotHierData(self, unitModels, disModels, replaceFig=True):

    nrDis = len(disModels)
    nrFuncUnits = len(unitModels)
    figSizeInch = (self.plotTrajParams['SubfigTrajWinSize'][0] / 100, self.plotTrajParams['SubfigTrajWinSize'][1] / 100)
    fig = pl.figure(1, figsize = figSizeInch)
    pl.clf()

    # nrRows = self.plotTrajParams['nrRows']
    # nrCols = self.plotTrajParams['nrCols']

    # nrRows, nrCols = auxFunc.findOptimalRowsCols(1+nrFuncUnits*2+len(self.plotTrajParams['labels']))
    print(nrDis + nrFuncUnits * 2 + len(self.plotTrajParams['labels']))
    nrRows, nrCols = auxFunc.findOptimalRowsCols(nrDis + nrFuncUnits * 2 + len(self.plotTrajParams['labels']))

    if (nrDis*2 + nrFuncUnits*2) > nrRows * nrCols:
      print('nrRows', nrRows)
      print('nrCols', nrCols)
      print('nrDis', nrDis)
      print('nrFuncUnits', nrFuncUnits)
      print('labels', self.plotTrajParams['labels'])
      raise ValueError('too few nrRows and nrCols')

    # trueParamsDis = self.plotTrajParams['trueParamsDis']
    # trueParamsFuncUnits = self.plotTrajParams['trueParamsFuncUnits']

    ######### compare subject shifts ##########
    nrPlotsSoFar = 0

    for d in range(nrDis):
      plotterDis = disModels[d].plotter
      nrPlotsSoFar = plotterDis.scatterSubjShifts(disModels[d], nrPlotsSoFar, nrRows, nrCols)

    ######### compare dysfunc traj within diseases ##########

    for d in range(nrDis):
      plotterDis = disModels[d].plotter
      trajStruct = plotterDis.getTrajStruct(disModels[d])
      nrPlotsSoFar = plotterDis.subplotAllTrajWithModelData(disModels[d], trajStruct, nrPlotsSoFar, nrRows, nrCols)

    ######### compare biomk traj within functional units ##########

    for f in range(nrFuncUnits):
      plotterFunc = unitModels[f].plotter
      trajStruct = plotterFunc.getTrajStruct(unitModels[f])
      nrPlotsSoFar = plotterFunc.subplotAllTrajWithModelData(unitModels[f], trajStruct, nrPlotsSoFar, nrRows, nrCols)

    pl.tight_layout(pad=1)

    if replaceFig:
      fig.show()
    else:
      pl.show()
    pl.pause(0.05)

    # print(ads)
    return fig

  def plotHierGivenData(self, unitModels, disModel,xsDisBSX,ysDisBSX,xsFuncUBSX,ysFuncUBSX, replaceFig=True):

    nrDis = 1
    nrFuncUnits = len(unitModels)
    figSizeInch = (self.plotTrajParams['SubfigTrajWinSize'][0] / 100, self.plotTrajParams['SubfigTrajWinSize'][1] / 100)
    fig = pl.figure(1, figsize = figSizeInch)
    pl.clf()

    # nrRows = self.plotTrajParams['nrRows']
    # nrCols = self.plotTrajParams['nrCols']

    nrRows, nrCols = auxFunc.findOptimalRowsCols(nrDis * 2 + nrFuncUnits * 2 + 5)

    if (nrDis*2 + nrFuncUnits*2) > nrRows * nrCols:
      print('nrRows', nrRows)
      print('nrCols', nrCols)
      print('nrDis', nrDis)
      print('nrFuncUnits', nrFuncUnits)
      print('labels', self.plotTrajParams['labels'])
      raise ValueError('too few nrRows and nrCols')

    # trueParamsDis = self.plotTrajParams['trueParamsDis']
    # trueParamsFuncUnits = self.plotTrajParams['trueParamsFuncUnits']

    nrPlotsSoFar = 0

    ######### compare dysfunc traj within diseases ##########


    plotterDis = disModel.plotter
    trajStruct = plotterDis.getTrajStruct(disModel)
    nrPlotsSoFar = plotterDis.subplotAllTrajWithGivenData(xsDisBSX, ysDisBSX, trajStruct, nrPlotsSoFar)

    ######### compare biomk traj within functional units ##########

    for f in range(nrFuncUnits):
      plotterFunc = unitModels[f].plotter
      unitModels[f].plotter.plotTrajParams['labels'] = ['unit %d %s' % (f, x)
        for x in unitModels[f].plotter.plotTrajParams['labels']]

      trajStruct = plotterFunc.getTrajStruct(unitModels[f])
      nrPlotsSoFar = plotterFunc.subplotAllTrajWithGivenData(xsFuncUBSX[f], ysFuncUBSX[f], trajStruct, nrPlotsSoFar)

    pl.tight_layout(pad=1)

    if replaceFig:
      fig.show()
    else:
      pl.show()
    pl.pause(0.05)

    # print(ads)
    return fig


  def plotCompWithTrueParams(self, unitModels, disModels, replaceFig=True):

    nrDis = len(disModels)
    nrFuncUnits = len(unitModels)
    figSizeInch = (self.plotTrajParams['SubfigTrajWinSize'][0] / 100, self.plotTrajParams['SubfigTrajWinSize'][1] / 100)
    fig = pl.figure(2, figsize = figSizeInch)
    pl.clf()

    # nrRows = self.plotTrajParams['nrRows']
    # nrCols = self.plotTrajParams['nrCols']

    nrRows, nrCols = auxFunc.findOptimalRowsCols(nrDis * 2 + nrFuncUnits * 2 + 5)

    if (nrDis*2 + nrFuncUnits*2) > nrRows * nrCols:
      print('nrRows', nrRows)
      print('nrCols', nrCols)
      print('nrDis', nrDis)
      print('nrFuncUnits', nrFuncUnits)
      print('labels', self.plotTrajParams['labels'])
      raise ValueError('too few nrRows and nrCols')

    # trueParamsDis = self.plotTrajParams['trueParamsDis']
    # trueParamsFuncUnits = self.plotTrajParams['trueParamsFuncUnits']

    ######### compare subject shifts ##########
    nrPlotsSoFar = 0

    for d in range(nrDis):
      plotterDis = disModels[d].plotter
      nrPlotsSoFar = plotterDis.scatterSubjShifts(disModels[d], nrPlotsSoFar, nrRows, nrCols)

    ######### compare dysfunc traj within diseases ##########

    for d in range(nrDis):
      plotterDis = disModels[d].plotter
      trajStruct = plotterDis.getTrajStructWithTrueParams(disModels[d])
      nrPlotsSoFar = plotterDis.subplotAllTraj(disModels[d], trajStruct, nrPlotsSoFar, nrRows, nrCols)

    ######### compare biomk traj within functional units ##########

    for f in range(nrFuncUnits):
      plotterFunc = unitModels[f].plotter
      trajStruct = plotterFunc.getTrajStructWithTrueParams(unitModels[f])
      nrPlotsSoFar = plotterFunc.subplotAllTraj(unitModels[f], trajStruct, nrPlotsSoFar, nrRows, nrCols)

    pl.tight_layout(pad=1)

    if replaceFig:
      fig.show()
    else:
      pl.show()
    pl.pause(0.05)

    # print(ads)
    return fig

  def plotAllBiomkDisSpace(self, dpmObj, params, disNr):
    # first predict subject DTI measures

    diag = params['diag']
    indxSubjToKeep = np.where(dpmObj.binMaskSubjForEachDisD[disNr])[0]

    nrBiomk = len(params['X'])
    print('nrBiomk', nrBiomk)
    Xfilt = [0 for b in range(nrBiomk)]
    Yfilt = [0 for b in range(nrBiomk)]
    for b in range(nrBiomk):
      Xfilt[b] = [params['X'][b][i] for i in indxSubjToKeep]
      Yfilt[b] = [params['Y'][b][i] for i in indxSubjToKeep]

    diagSubjCurrDis = diag[indxSubjToKeep]
    ridCurrDis = params['RID'][indxSubjToKeep]
    nrSubCurrDis = indxSubjToKeep.shape[0]

    XshiftedDisModelBS = [[] for b in range(nrBiomk)]
    XshiftedDisModelUS, XdisModelUS, YdisModelUS, _ = dpmObj.disModels[disNr].getData()

    for s in range(nrSubCurrDis):
      # bTmp = 0  # some biomarker, doesn't matter which one
      # ysCurrSubXB = dpmObj.predictBiomkSubjGivenXs(XshiftedDisModelUS[bTmp][s], disNr)

      for b in range(nrBiomk):

        if Xfilt[b][s].shape[0] > 0:
          # fix problem when a subject has the same xs twice (bad input dataset with same visit twice)
          while np.unique(Xfilt[b][s]).shape[0] < Xfilt[b][s].shape[0]:
            for x in Xfilt[b][s]:
              if np.sum(Xfilt[b][s] == x) > 1:
                idxToRemove = np.where(Xfilt[b][s] == x)[0][0]
                Yfilt[b][s] = np.concatenate((Yfilt[b][s][:idxToRemove], Yfilt[b][s][idxToRemove + 1:]))
                Xfilt[b][s] = np.concatenate((Xfilt[b][s][:idxToRemove], Xfilt[b][s][idxToRemove + 1:]))

                break

          XshiftedDisModelBS[b] += [XshiftedDisModelUS[0][s]]
        else:
          XshiftedDisModelBS[b] += [np.array([])]

        if Yfilt[b][s].shape[0] < XshiftedDisModelBS[b][s].shape[0]:
          allXsOrig = XdisModelUS[0][s]
          xsCurrSubj = Xfilt[b][s]
          maskMissing = np.in1d(allXsOrig, xsCurrSubj)
          XshiftedDisModelBS[b][s] = XshiftedDisModelBS[b][s][maskMissing]
          # print('xsCurrSubj', xsCurrSubj)
          # print('allXsOrig', allXsOrig)
          # print('maskMissing', maskMissing)
          # print('XshiftedDisModelBS[b][s]', XshiftedDisModelBS[b][s])
          # print(das)

    for b in range(nrBiomk):
      assert len(params['X'][b]) == len(params['Y'][b])
      assert len(XshiftedDisModelBS[b]) == len(Yfilt[b])

    # part 2. plot the inferred dynamics for DRC data:
    # every biomarker against original DPS
    # also plot extra validation data on top
    xsTrajX = dpmObj.disModels[disNr].getXsMinMaxRange()
    predTrajXB = dpmObj.predictBiomkSubjGivenXs(xsTrajX, disNr)
    trajSamplesBXS = dpmObj.sampleBiomkTrajGivenXs(xsTrajX, disNr, nrSamples=100)

    print('predTrajXB', predTrajXB)
    # print('XshiftedDisModelBS', XshiftedDisModelBS[0][0])
    # print('Yfilt', Yfilt[0][0])
    # print(adsa)

    gpPlotter = PlotterDis(self.plotTrajParams)
    if gpPlotter.plotTrajParams['isSynth']:

      trueXsTrajX = dpmObj.params['trueParamsDis'][disNr]['xsX']
      # trueXsScaledZeroOne = (trueXsTrajX - np.min(trueXsTrajX)) / (np.max(trueXsTrajX) - np.min(trueXsTrajX))

      trueYsTrajXB = dpmObj.params['trueParamsDis'][disNr]['ysXB']

      fig = gpPlotter.plotTrajInDisSpaceTrueTraj(xsTrajX, predTrajXB, trajSamplesBXS,
         XshiftedDisModelBS, Yfilt, diagSubjCurrDis, trueXsTrajX, trueYsTrajXB,
         labels=self.plotTrajParams['labels'], ssdDKT=None,
         ssdNoDKT=None,
         replaceFig=True)
    else:
      fig = gpPlotter.plotTrajInDisSpace(xsTrajX, predTrajXB, trajSamplesBXS,
         XshiftedDisModelBS, Yfilt, diagSubjCurrDis, XsubjValidBSX=None, YsubjValidBSX=None,
         diagValidS=None,  labels=self.plotTrajParams['labels'], ssdDKT=None, ssdNoDKT=None,
         replaceFig=True)



    return fig


class PlotterGP(ABC):

  def __init__(self, plotTrajParams):
    self.plotTrajParams = plotTrajParams

  def plotTraj(self, gpModel, replaceFig=True, legendExtraPlot=False):
    nrBiomk = gpModel.nrBiomk

    # Plot method

    font = {'family': 'normal',
      'size': 13}

    import matplotlib
    matplotlib.rc('font', **font)

    figSizeInch = (self.plotTrajParams['SubfigTrajWinSize'][0] / 100, self.plotTrajParams['SubfigTrajWinSize'][1] / 100)
    fig = pl.figure(1, figsize = figSizeInch)
    pl.clf()
    # fig.show()

    print(self.plotTrajParams['labels'])
    print('nrBiomk', nrBiomk)
    # print(asda)

    diagNrs = np.unique(self.plotTrajParams['diag'])
    nrDiags = diagNrs.shape[0]

    min_yB, max_yB = gpModel.getMinMaxY_B(extraDelta=0)
    deltaB = [(max_yB[b] - min_yB[b]) * 0.2 for b in range(nrBiomk)]

    nrRows = self.plotTrajParams['nrRows']
    nrCols = self.plotTrajParams['nrCols']

    # nrRows = 2
    # nrCols = 3

    newX = gpModel.getXsMinMaxRange()
    predBiomksYscaledXB = gpModel.predictBiomk(newX)

    Xshifted, X, Y, _ = gpModel.getData()

    # import pdb
    # pdb.set_trace()

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
      for sub in range(gpModel.nrSubj):
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

    if legendExtraPlot:
      ax = pl.subplot(nrRows, nrCols, nrBiomk+1)
      pl.axis('off')

      # plot subject data
      diagCounters = dict([(k,0) for k in self.plotTrajParams['diagLabels'].keys()])
      for sub in range(gpModel.nrSubj):
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

  def plotCompWithTrueParams(self, gpModel, replaceFig=True):

    nrBiomk = gpModel.nrBiomk
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
    nrPlotsSoFar = 0
    nrPlotsSoFar = self.scatterSubjShifts(gpModel, nrPlotsSoFar, nrRows, nrCols)

    ######### compare all trajectories ##########

    trajStruct = self.getTrajStructWithTrueParams(gpModel)

    nrPlotsSoFar = self.subplotAllTraj(gpModel, trajStruct, nrPlotsSoFar, nrRows, nrCols)

    ######### compare biomarker trajectories one by one ##########

    nrPlotsSoFar = self.subplotTrajOneByOne(gpModel, trajStruct, nrPlotsSoFar, nrRows, nrCols)

    if replaceFig:
      fig.show()
    else:
      pl.show()
    pl.pause(0.05)

    # print(ads)
    return fig

  def plotTrajSameSpace(self, gpModel, replaceFig=True, subjStagesEstim=None):

    # nrBiomk = gpModel.nrBiomk
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
    predTrajXB = gpModel.predictBiomk(newXTraj)

    #rescale all trajectories
    predTrajScaledXB, _, yMinAll, yMaxAll, min_yB, max_yB = \
      rescaleTraj(predTrajXB, predTrajXB, self.plotTrajParams['yNormMode'],
      self.plotTrajParams['diag'], nrBiomk, subjStagesEstim, gpModel)

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
      XsubjBSX, YsubjBSX, diagS, XsubjValidBSX, YsubjValidBSX, diagValidS, labels,
      ssdDKT=None, ssdNoDKT=None, replaceFig=True):
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
    fig = pl.figure(3, figsize = figSizeInch)
    pl.clf()
    fig.show()

    diagNrs = np.unique(diagS)
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

      minY = min_yB[b]-deltaB[b]
      maxY = max_yB[b]+deltaB[b]
      ax.set_ylim([minY, maxY])
      minX, maxX = ax.get_xlim()

      # pl.legend(ncol=1,fontsize=12)

      if ssdDKT is not None:
        ax.text(minX + 0.1*(maxX-minX), minY + 0.9*(maxY-minY), 'SSD DKT=%.3f' % ssdDKT[b])
        ax.text(minX + 0.1 * (maxX - minX), minY + 0.8 * (maxY - minY), 'SSD no-DKT=%.3f' % ssdNoDKT[b])

    # pl.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)


    fig.text(0.5, 0.04, 'Disease Progression (months)', ha='center')
    fig.text(0.08, 0.5, 'Biomarker Value (normalised)', va='center', rotation='vertical')

    h, labels = ax.get_legend_handles_labels()
    print(h, labels)
    # legend =  pl.legend(handles=h, bbox_to_anchor=plotTrajParams['legendPos'], loc='upper center', ncol=plotTrajParams['legendCols'])

    legend = pl.figlegend(h[:5], labels[:5], loc='upper center', ncol=5, labelspacing=0.)
    # set the linewidth of each legend object
    for legobj in legend.legendHandles:
      legobj.set_linewidth(4.0)


    if replaceFig:
      fig.show()
    else:
      pl.show()
    # pl.pause(5)

    return fig

  def plotTrajInDisSpaceTrueTraj(self, xsTrajX, predTrajXB, trajSamplesBXS,
      XsubjBSX, YsubjBSX, diagS, trueXsTrajX, trueYsTrajXB, labels,
      ssdDKT=None, ssdNoDKT=None, replaceFig=True):
    """
    plot biomarker traj and subject data in disease space. function doesn't do any scaling of xs or ys

    :param xsTrajX:
    :param predTrajXB:
    :param trajSamplesBXS:
    :param XsubjBSX:
    :param YsubjBSX:
    :param diagS:
    :param trueXsTrajX:
    :param trueYsTrajXB:
    :param replaceFig:
    :return:
    """

    font = {'family': 'normal',
      'size': 13}

    import matplotlib
    matplotlib.rc('font', **font)

    # Plot method
    figSizeInch = (self.plotTrajParams['SubfigTrajWinSize'][0] / 100, self.plotTrajParams['SubfigTrajWinSize'][1] / 100)
    fig = pl.figure(3, figsize = figSizeInch)
    pl.clf()
    fig.show()

    diagNrs = np.unique(diagS)
    nrDiags = diagNrs.shape[0]

    nrBiomk = len(XsubjBSX)
    # nrBiomk = 18
    nrRows, nrCols = auxFunc.findOptimalRowsCols(nrBiomk)

    nrSamples = trajSamplesBXS.shape[2]

    min_yB = np.zeros(nrBiomk)
    max_yB = np.zeros(nrBiomk)
    for b in range(nrBiomk):
      listsMin = [np.min(yS) for yS in YsubjBSX[b] if len(yS) > 0] + [np.min(predTrajXB[:,b])]
      listsMax = [np.max(yS) for yS in YsubjBSX[b] if len(yS) > 0] + [np.max(predTrajXB[:,b])]

      listsMin += np.min(trueYsTrajXB[b,:])
      listsMax += np.max(trueYsTrajXB[b,:])

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

      ax.plot(xsTrajX,predTrajXB[:,b],
        lw=2, color='black', label='estim. traj.')

      ax.plot(trueXsTrajX, trueYsTrajXB[:,b], '--', lw=2,color='black', label='true traj')

      # ax.plot([np.min(xsTrajX), np.max(xsTrajX)], [min_yB[b], max_yB[b]],
      #   color=(0.5,0.5,0.5), lw=2)

      minY = min_yB[b]-deltaB[b]
      maxY = max_yB[b]+deltaB[b]
      ax.set_ylim([minY, maxY])
      minX, maxX = ax.get_xlim()

      # pl.legend(ncol=1,fontsize=12)

      if ssdDKT is not None:
        ax.text(minX + 0.1*(maxX-minX), minY + 0.9*(maxY-minY), 'SSD DKT=%.3f' % ssdDKT[b])
        ax.text(minX + 0.1 * (maxX - minX), minY + 0.8 * (maxY - minY), 'SSD no-DKT=%.3f' % ssdNoDKT[b])

    # pl.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)


    fig.text(0.5, 0.04, 'Disease Progression (months)', ha='center')
    fig.text(0.08, 0.5, 'Biomarker Value (normalised)', va='center', rotation='vertical')

    h, labels = ax.get_legend_handles_labels()
    print(h, labels)
    # legend =  pl.legend(handles=h, bbox_to_anchor=plotTrajParams['legendPos'], loc='upper center', ncol=plotTrajParams['legendCols'])

    legend = pl.figlegend(h[:5], labels[:5], loc='upper center', ncol=5, labelspacing=0.)
    # set the linewidth of each legend object
    for legobj in legend.legendHandles:
      legobj.set_linewidth(4.0)


    if replaceFig:
      fig.show()
    else:
      pl.show()
    # pl.pause(5)

    return fig


  def plotTrajInBiomkSpace(self, xsTrajXB, predTrajXB, trajSamplesBXS,
      XsubjData1BSX, YsubjData1BSX, diagData1S, XsubjData2BSX, YsubjData2BSX, diagData2S,
      XsubjData3BSX, YsubjData3BSX, diagData3S, labels, ssdDKT=None, ssdNoDKT=None,
      replaceFig=True):
    """
    plot biomarker traj and subject data over the space of other biomarkers.
    assumes xsTrajXB.shape[1] == predTrajXB.shape[1], so each biomarker in predTrajXB
    is plotted against another biomarkers in xsTrajXB

    """

    font = {'family': 'normal',
      'size': 13}

    import matplotlib
    matplotlib.rc('font', **font)

    # Plot method
    figSizeInch = (self.plotTrajParams['SubfigTrajWinSize'][0] / 100, self.plotTrajParams['SubfigTrajWinSize'][1] / 100)
    fig = pl.figure(10, figsize = figSizeInch)
    pl.clf()
    fig.show()

    nrBiomk = len(XsubjData1BSX)
    # nrBiomk = 18
    nrRows, nrCols = auxFunc.findOptimalRowsCols(nrBiomk)

    min_yB = np.zeros(nrBiomk)
    max_yB = np.zeros(nrBiomk)
    for b in range(nrBiomk):
      # print([np.min(yS) for yS in YsubjBSX[b] if len(yS) > 0])
      # print([np.min(predTrajXB[:,b])])
      # print([np.min(yS) for yS in YsubjBSX[b] if len(yS) > 0] + [np.min(predTrajXB[:,b])])
      listsMin = []
      listsMax = []

      print('YsubjData2BSX[b]', YsubjData2BSX[b])
      print([ys.shape for ys in YsubjData2BSX[b]])

      if predTrajXB is not None:
        listsMin = [np.min(predTrajXB[:, b])]
        listsMax = [np.max(predTrajXB[:, b])]
      if YsubjData1BSX is not None:
        listsMin = [np.min(yS) for yS in YsubjData1BSX[b] if yS.shape[0] > 0]
        listsMax = [np.max(yS) for yS in YsubjData1BSX[b] if yS.shape[0] > 0]
      if YsubjData2BSX is not None:
        listsMin += [np.min(yS) for yS in YsubjData2BSX[b] if yS.shape[0] > 0]
        listsMax += [np.max(yS) for yS in YsubjData2BSX[b] if yS.shape[0] > 0]

      min_yB[b] = np.min(listsMin)
      max_yB[b] = np.max(listsMax)

    deltaB = (max_yB - min_yB)/5
    legendEntries = 3

    for b in range(nrBiomk):
      ax = pl.subplot(nrRows, nrCols, b + 1)
      pl.title(labels[b])
      # print('--------------b', b)
      # plot traj samples
      if xsTrajXB is not None:
        nrSamples = trajSamplesBXS.shape[2]
        for i in range(nrSamples):
          ax.plot(xsTrajXB[:,b], trajSamplesBXS[b, :, i], lw = 0.05,
            color = 'red', alpha=1)

      if XsubjData1BSX is not None:
        self.plotSubjData(ax, XsubjData1BSX[b], YsubjData1BSX[b], diagData1S, labelExtra ='')

      # print('-------------- validation data')
      if XsubjData2BSX is not None:
        self.plotSubjData(ax, XsubjData2BSX[b], YsubjData2BSX[b], diagData2S, labelExtra ='')
        legendEntries += np.unique(diagData2S).shape[0]

      if XsubjData3BSX is not None:
        self.plotSubjData(ax, XsubjData3BSX[b], YsubjData3BSX[b], diagData3S, labelExtra ='')
        legendEntries += np.unique(diagData3S).shape[0]

      if xsTrajXB is not None:
        ax.plot(xsTrajXB[:,b],predTrajXB[:, b],
          lw=2, color='black', label='estim. traj.')

      # ax.plot([np.min(xsTrajX), np.max(xsTrajX)], [min_yB[b], max_yB[b]],
      #   color=(0.5,0.5,0.5), lw=2)

      minY = min_yB[b]-deltaB[b]
      maxY = max_yB[b]+deltaB[b]
      ax.set_ylim([minY, maxY])
      minX, maxX = ax.get_xlim()

      # pl.legend(ncol=1,fontsize=12)


      ax.text(minX + 0.1*(maxX-minX), minY + 0.9*(maxY-minY), 'SSD DKT=%.3f' % ssdDKT[b])
      ax.text(minX + 0.1 * (maxX - minX), minY + 0.8 * (maxY - minY), 'SSD no-DKT=%.3f' % ssdNoDKT[b])

    # pl.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)


    fig.text(0.5, 0.04, 'Disease Progression (months)', ha='center')
    fig.text(0.08, 0.5, 'Biomarker Value (normalised)', va='center', rotation='vertical')

    h, labels = ax.get_legend_handles_labels()
    print(h, labels)
    # legend =  pl.legend(handles=h, bbox_to_anchor=plotTrajParams['legendPos'], loc='upper center', ncol=plotTrajParams['legendCols'])


    legend = pl.figlegend(h[:legendEntries], labels[:legendEntries], loc='upper center', ncol=5, labelspacing=0.)
    # set the linewidth of each legend object
    for legobj in legend.legendHandles:
      legobj.set_linewidth(4.0)


    if replaceFig:
      fig.show()
    else:
      pl.show()
    # pl.pause(5)

    return fig


  def plotTrajInDisSpaceOverlap(self, xsTrajX, predTrajXB, trajSamplesBXS, params, replaceFig=True):
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

      if XsubjSX[sub].shape[0] > 0:
        ax.plot(XsubjSX[sub], YsubjSX[sub],
                color=self.plotTrajParams['diagColors'][diagCurrSubj], lw=0.5)
        ax.scatter(XsubjSX[sub], YsubjSX[sub],
                   marker=self.plotTrajParams['diagScatterMarkers'][diagCurrSubj],
                   color=self.plotTrajParams['diagColors'][diagCurrSubj], lw=2.5,
                   label=currLabel)

        diagCounters[diagCurrSubj] += 1


  def Plot_predictions(self, gpModel, final_predSX, Xrange, names=[]):
    scaling = gpModel.mean_std_X[0][1] * gpModel.max_X[0]
    for i in range(len(final_predSX)):
      valid_indices = np.where(np.array(final_predSX[i]) != 0)
      assert final_predSX[i].shape[0] == Xrange.shape[0]
      pl.plot(Xrange[valid_indices] * scaling + gpModel.mean_std_X[0][0], np.array(final_predSX[i])[valid_indices])
      if len(names) > 0:
        max = np.int(np.where(final_predSX[i] == np.max(final_predSX[i]))[0])
        pl.annotate(names[i], xy=(Xrange[max] * scaling + gpModel.mean_std_X[0][0], final_predSX[i][max]))
    pl.show()

  def subplotAllTraj(self, gpModel, trajStruct, nrPlotsSoFar, nrRows, nrCols):

    newXTrajScaledZeroOne = trajStruct['newXTrajScaledZeroOne']
    trueXsScaledZeroOne = trajStruct['trueXsScaledZeroOne']
    predTrajScaledXB = trajStruct['predTrajScaledXB']
    trueTrajScaledXB = trajStruct['trueTrajScaledXB']
    yMinAll = trajStruct['yMinAll']
    yMaxAll = trajStruct['yMaxAll']
    min_yB = trajStruct['min_yB']
    max_yB = trajStruct['max_yB']

    # nrRows = self.plotTrajParams['nrRows']
    # nrCols = self.plotTrajParams['nrCols']

    if self.plotTrajParams['allTrajOverlap']:
      ax2 = pl.subplot(nrRows, nrCols, nrPlotsSoFar+1)
      pl.title('%s all trajectories' % self.plotTrajParams['title'])
      ax2.set_ylim([yMinAll, yMaxAll])
      for b in range(gpModel.nrBiomk):
        ax2.plot(newXTrajScaledZeroOne, predTrajScaledXB[:, b], '-', lw=2
                 , c=self.plotTrajParams['colorsTraj'][b], label=self.plotTrajParams['labels'][b])
        print('trueXsScaledZeroOne trueYsTrajXB', trueXsScaledZeroOne.shape, trueYsTrajXB.shape)
        ax2.plot(trueXsScaledZeroOne, trueTrajScaledXB[:, b], '--', lw=2
                 , c=self.plotTrajParams['colorsTraj'][b], label=self.plotTrajParams['labels'][b])

      ax2.legend(loc='lower right',ncol=1)
      nrPlotsSoFar += 1
    else:
      ax2 = pl.subplot(nrRows, nrCols, nrPlotsSoFar+1)
      pl.title('%s estimated trajectories'  % self.plotTrajParams['title'])
      ax2.set_ylim([yMinAll, yMaxAll])
      for b in range(gpModel.nrBiomk):
        ax2.plot(newXTrajScaledZeroOne, predTrajScaledXB[:, b], '-', lw=2
                 , c=self.plotTrajParams['colorsTraj'][b], label=self.plotTrajParams['labels'][b])

      ax2.legend(loc='lower right',ncol=1)

      ax3 = pl.subplot(nrRows, nrCols, nrPlotsSoFar+2)
      pl.title('%s true trajectories' % self.plotTrajParams['title'])
      ax3.set_ylim([yMinAll, yMaxAll])
      for b in range(gpModel.nrBiomk):
        ax3.plot(trueXsScaledZeroOne, trueTrajScaledXB[:, b], '--', lw=2
                 , c=self.plotTrajParams['colorsTraj'][b], label=self.plotTrajParams['labels'][b])

      ax3.legend(loc='lower right',ncol=1)

      nrPlotsSoFar += 2

    return nrPlotsSoFar

  def subplotAllTrajWithModelData(self, gpModel, trajStruct, nrPlotsSoFar, nrRows, nrCols):

    newXTraj = trajStruct['newXTraj']
    predTrajXB = trajStruct['predTrajXB']
    yMinAll = trajStruct['yMinAll']
    yMaxAll = trajStruct['yMaxAll']
    min_yB = trajStruct['min_yB']
    max_yB = trajStruct['max_yB']

    print('nrPlotsSoFar', nrPlotsSoFar)
    print('nrRows', nrRows)
    print('nrCols', nrCols)

    # nrRows = self.plotTrajParams['nrRows']
    # nrCols = self.plotTrajParams['nrCols']

    xsShiftedBSX, _, ysBSX, _ = gpModel.getData()

    for b in range(gpModel.nrBiomk):
      ax2 = pl.subplot(nrRows, nrCols, nrPlotsSoFar + 1)
      pl.title(self.plotTrajParams['labels'][b])
      ax2.set_ylim([yMinAll, yMaxAll])
      ax2.plot(newXTraj, predTrajXB[:, b], '-', lw=2
               , c=self.plotTrajParams['colorsTraj'][b], label=self.plotTrajParams['labels'][b])

      self.plotSubjData(ax2, xsShiftedBSX[b], ysBSX[b], self.plotTrajParams['diag'], labelExtra='')

      nrPlotsSoFar += 1


    return nrPlotsSoFar

  def subplotAllTrajWithGivenData(self, xsShiftedBSX,ysBSX, trajStruct, nrPlotsSoFar):

    newXTraj = trajStruct['newXTraj']
    predTrajXB = trajStruct['predTrajXB']
    yMinAll = trajStruct['yMinAll']
    yMaxAll = trajStruct['yMaxAll']
    min_yB = trajStruct['min_yB']
    max_yB = trajStruct['max_yB']

    nrRows = self.plotTrajParams['nrRows']
    nrCols = self.plotTrajParams['nrCols']

    nrBiomk = len(xsShiftedBSX)
    for b in range(nrBiomk):
      ax2 = pl.subplot(nrRows, nrCols, nrPlotsSoFar + 1)
      pl.title(self.plotTrajParams['labels'][b])
      ax2.set_ylim([yMinAll, yMaxAll])
      ax2.plot(newXTraj, predTrajXB[:, b], '-', lw=2
               , c=self.plotTrajParams['colorsTraj'][b], label=self.plotTrajParams['labels'][b])

      self.plotSubjData(ax2, xsShiftedBSX[b], ysBSX[b], self.plotTrajParams['diag'], labelExtra='')

      nrPlotsSoFar += 1


    return nrPlotsSoFar

  def subplotTrajOneByOne(self, gpModel, trajStruct, nrPlotsSoFar, nrRows, nrCols):

    newXTrajScaledZeroOne = trajStruct['newXTrajScaledZeroOne']
    trueXsScaledZeroOne = trajStruct['trueXsScaledZeroOne']
    predTrajScaledXB = trajStruct['predTrajScaledXB']
    trueTrajScaledXB = trajStruct['trueTrajScaledXB']
    yMinAll = trajStruct['yMinAll']
    yMaxAll = trajStruct['yMaxAll']
    min_yB = trajStruct['min_yB']
    max_yB = trajStruct['max_yB']

    # nrRows = self.plotTrajParams['nrRows']
    # nrCols = self.plotTrajParams['nrCols']

    for b in range(gpModel.nrBiomk):
      ax4 = pl.subplot(nrRows, nrCols, b+nrPlotsSoFar+1)
      pl.title(self.plotTrajParams['labels'][b])

      ax4.plot(newXTrajScaledZeroOne, predTrajScaledXB[:, b], '-',lw=2,
        c=self.plotTrajParams['colorsTraj'][b], label='estimated')

      ax4.plot(trueXsScaledZeroOne, trueTrajScaledXB[:,b], '--', lw=2,
        c=self.plotTrajParams['colorsTraj'][b], label='true')

      ax4.set_ylim([min_yB[b], max_yB[b]])
      ax4.legend(loc='lower right')

    return nrPlotsSoFar + gpModel.nrBiomk

  def getTrajStruct(self, gpModel):

    newXTraj = gpModel.getXsMinMaxRange()
    predTrajXB = gpModel.predictBiomk(newXTraj)

    nrBiomk = predTrajXB.shape[1]
    subjShiftsEstimS = gpModel.getSubShiftsLong()

    # rescale all trajectories
    _, _, yMinAll, yMaxAll, min_yB, max_yB = \
      rescaleTraj(predTrajXB, predTrajXB, 'unscaled',
                  self.plotTrajParams['diag'], nrBiomk, subjShiftsEstimS, gpModel)

    trajStruct = dict(newXTraj=newXTraj, predTrajXB=predTrajXB, yMinAll=yMinAll, yMaxAll=yMaxAll,
      min_yB=min_yB, max_yB=max_yB)

    return trajStruct

  def getTrajStructWithTrueParams(self, gpModel):

    newXTraj = gpModel.getXsMinMaxRange()
    newXTrajScaledZeroOne = (newXTraj - np.min(newXTraj)) / (np.max(newXTraj) - np.min(newXTraj))
    trueXsTrajX = self.getTrueXs()
    trueXsScaledZeroOne = (trueXsTrajX - np.min(trueXsTrajX)) / (np.max(trueXsTrajX) - np.min(trueXsTrajX))

    predTrajXB = gpModel.predictBiomk(newXTraj)
    trueYsTrajXB = self.getTrueYs()  # either ysXB or ysXU
    trueTrajCopyXB = copy.deepcopy(trueYsTrajXB)

    nrBiomk = predTrajXB.shape[1]
    subjShiftsEstimS = gpModel.getSubShiftsLong()


    # rescale all trajectories
    predTrajScaledXB, trueTrajScaledXB, yMinAll, yMaxAll, min_yB, max_yB = \
      rescaleTraj(predTrajXB, trueTrajCopyXB, self.plotTrajParams['yNormMode'],
                  self.plotTrajParams['diag'], nrBiomk, subjShiftsEstimS, gpModel)

    trajStruct = dict(newXTrajScaledZeroOne=newXTrajScaledZeroOne, trueXsScaledZeroOne=trueXsScaledZeroOne,
                      predTrajScaledXB=predTrajScaledXB, trueTrajScaledXB=trueTrajScaledXB,
                      yMinAll=yMinAll, yMaxAll=yMaxAll, min_yB=min_yB, max_yB=max_yB)

    # print(trajStruct)
    # print(predTrajXB, trueTrajCopyXB, self.plotTrajParams['yNormMode'], self.plotTrajParams['diag'], nrBiomk, subjShiftsEstimS)
    # print(adsa)

    return trajStruct

  def scatterSubjShifts(self, gpModel, nrPlotsSoFar, nrRows, nrCols):

    # nrRows = self.plotTrajParams['nrRows']
    # nrCols = self.plotTrajParams['nrCols']

    subShiftsTrueS = self.getTrueShifts()
    # estimShifts = gpModel.params_time_shift[0,:]

    nrSubjLong = len(gpModel.X[0])

    subjShiftsEstimS = gpModel.getSubShiftsLong()

    ax = pl.subplot(nrRows, nrCols, nrPlotsSoFar+1)


    diagNrs = np.unique(self.plotTrajParams['diag'])
    nrDiags = diagNrs.shape[0]
    # print('diag.shape', self.plotTrajParams['diag'].shape)
    # print('diagNrs', diagNrs)
    # print('subShiftsTrueS', subShiftsTrueS.shape)
    # print('subjShiftsEstimS', subjShiftsEstimS.shape)
    # print(adsa)

    for d in range(nrDiags):
      pl.scatter(subjShiftsEstimS[self.plotTrajParams['diag'] == diagNrs[d]],
                 subShiftsTrueS[self.plotTrajParams['diag'] == diagNrs[d]],
                 c=self.plotTrajParams['diagColors'][diagNrs[d]],
                 label=self.plotTrajParams['diagLabels'][diagNrs[d]])

    pl.title('Subject shifts')
    pl.xlabel('estimated shifts')
    pl.ylabel('true shifts')
    ax.set_ylim([np.min(subShiftsTrueS), np.max(subShiftsTrueS)])
    ax.legend(ncol=1)

    return nrPlotsSoFar+1

  @abstractmethod
  def getTrueYs(self):
    raise NotImplementedError('need to call plotter subclass')

  def getTrueXs(self):
    return self.plotTrajParams['trueParams']['xsX']

  def getTrueShifts(self):
    return self.plotTrajParams['trueParams']['subShiftsS']

class PlotterFuncUnit(PlotterGP):

  def __init__(self, plotTrajParams):
    super(PlotterFuncUnit, self).__init__(plotTrajParams)

  def getTrueYs(self):
    return self.plotTrajParams['trueParams']['ysXB']

class PlotterDis(PlotterGP):

  def __init__(self, plotTrajParams):
    super(PlotterDis, self).__init__(plotTrajParams)

  def getTrueYs(self):
    return self.plotTrajParams['trueParams']['ysXU']



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


def rescaleTraj(predTrajXB, trueTrajXB, yNormMode, diag, nrBiomk, subjStagesEstim, gpModel):
  """
  re-scales the trajectories both along the Y axis (DPS ~ N(0,1) or DPS_ealry ~ N(0,1) )
  :param predTrajXB:
  :param trueTrajXB:
  :param yNormMode:
  :param diag:
  :param nrBiomk:
  :param subjStagesEstim:
  :param gpModel:
  :return:
  """

  if yNormMode == 'zScoreTraj':
    idxZscore = [np.where(np.in1d(diag, [CTL, CTL2]))[0]
                 for b in range(nrBiomk)]
  elif yNormMode == 'zScoreEarlyStageTraj':
    # take only the first 10 percentile at earliest stages and set that
    # as the true control group
    idxZscore = [0 for b in range(nrBiomk)]
    for b in range(nrBiomk):
      sortedIndx = np.argsort(subjStagesEstim)
      sortedIndx = np.array([i for i in sortedIndx if gpModel.Y[b][i].shape[0] > 0])
      percPoint = 0.1
      if sortedIndx.shape[0] < 50:
        # increase percPoint due to not enough samples
        # (with 10% of subj then at most 5 subj will be used to estimate the Z-scores)
        percPoint = 0.2

      idxZscore[b] = sortedIndx[:int(sortedIndx.shape[0] * percPoint)]

      if len(idxZscore) == 0:
        idxZscore[b] = list(range(int(subjStagesEstim.shape[0] / 20)))
  elif yNormMode == 'unscaled':
    pass
  else:
    raise ValueError('plotTrajParams[yNormMode] should be either unscaled, zScoreTraj or zScoreEarlyStageTraj')

  if yNormMode in ['zScoreTraj', 'zScoreEarlyStageTraj']:
    meanCtlB = np.zeros(gpModel.nrBiomk)
    stdCtlB = np.zeros(gpModel.nrBiomk)

    print('predTrajXB', predTrajXB)

    for b in range(gpModel.nrBiomk):
      yValsOfCtl = [gpModel.Y[b][s] for s in idxZscore[b]]
      yValsOfCtl = [l2 for l in yValsOfCtl for l2 in l]
      meanCtlB[b] = np.mean(yValsOfCtl)
      stdCtlB[b] = np.std(yValsOfCtl)

      print('b', b)
      print(meanCtlB[b], stdCtlB[b])
      print('trueTrajXB', trueTrajXB)
      print(trueTrajXB[:, b])
      print(gpModel.applyGivenScalingY(trueTrajXB[:, b], meanCtlB[b], stdCtlB[b]))

      predTrajXB[:, b] = gpModel.applyGivenScalingY(predTrajXB[:, b], meanCtlB[b], stdCtlB[b])
      trueTrajXB[:, b] = gpModel.applyGivenScalingY(trueTrajXB[:, b], meanCtlB[b], stdCtlB[b])

    yMinPred = np.min(predTrajXB, axis=(0, 1))
    yMinTrue = np.min(trueTrajXB, axis=(0, 1))
    yMinAll = np.min([yMinPred, yMinTrue])

    yMaxPred = np.max(predTrajXB, axis=(0, 1))
    yMaxTrue = np.max(trueTrajXB, axis=(0, 1))
    yMaxAll = np.max([yMaxPred, yMaxTrue])

    min_yB = yMinAll * np.ones(nrBiomk)
    max_yB = yMaxAll * np.ones(nrBiomk)

    deltaB = [(max_yB[b] - min_yB[b]) * 0.2 for b in range(nrBiomk)]
    min_yB = min_yB - deltaB
    max_yB = max_yB + deltaB

    # print('predTrajXB', predTrajXB)
    # print('trueTrajXB', trueTrajXB)
    # print('yMinTrue', yMinTrue)
    # print('yMinPred', yMinPred)
    # print('yMinAll', yMinAll)

    # print(adsa)


  elif yNormMode == 'unscaled':
    scaledYarrayB = [gpModel.applyScalingY(gpModel.Y_array[b], b) for b in range(nrBiomk)]
    min_yB, max_yB = gpModel.getMinMaxY_B(extraDelta=0.2)

    yMinAll = np.min(min_yB)
    yMaxAll = np.max(max_yB)
  else:
    raise ValueError('plotTrajParams[yNormMode] should be either unscaled, zScoreTraj or zScoreEarlyStageTraj')

  return predTrajXB, trueTrajXB, yMinAll, yMaxAll, min_yB, max_yB

