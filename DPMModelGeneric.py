import matplotlib.pyplot as plt
import numpy as np
import sys
import scipy.optimize

class DPMModelGeneric(object):
  plt.interactive(False)

  def __init__(self, X, Y, visitIndices, outFolder, plotter, names_biomarkers, group):
    # Initializing variables
    self.plotter = plotter
    self.outFolder = outFolder
    self.expName = plotter.plotTrajParams['expName']
    self.names_biomarkers = names_biomarkers
    self.group = group
    self.nrSubj = len(X[0])
    self.nrBiomk = len(X)
    self.X_array = []
    self.Y_array = []
    self.X = X
    self.Y = Y
    self.N_obs_per_sub = []
    self.params_time_shift = np.ndarray([2, len(X[0])])

    # Time shift initialized to 0
    self.params_time_shift[0, :] = 0

    # Estension of the model will include a time scaling factor (fixed to 1 so far)
    self.params_time_shift[1, :] = 1

    self.visitIndices = visitIndices
    self.X_array, self.N_obs_per_sub, self.indFullToMissingArrayB = self.convertLongToArray(self.X, self.visitIndices)
    self.Y_array, N_obs_per_sub2, _ = self.convertLongToArray(self.Y, self.visitIndices)
    self.checkNobsMatch(N_obs_per_sub2)

    minX = np.float128(np.min([el for sublist in self.X_array for item in sublist for el in item]))
    maxX = np.float128(np.max([el for sublist in self.X_array for item in sublist for el in item]))
    self.minX = minX
    self.maxX = maxX
    self.minScX = self.minX
    self.maxScX = self.maxX

  def updateTimeShifts(self, optimal_params):
    # for l in range(1):
    self.params_time_shift[0,:] = self.params_time_shift[0,:] + optimal_params[0,:]

    for b in range(self.nrBiomk):
      Xdata = np.array([[100]])
      for sub in range(self.nrSubj):
        temp = self.X_array[b][int(np.sum(self.N_obs_per_sub[b][:sub])):np.sum(self.N_obs_per_sub[b][:sub + 1])]
        shifted_temp = (temp + optimal_params[0][sub])
        Xdata = np.hstack([Xdata, shifted_temp.T])

      self.X_array[b] = Xdata[0][1:].reshape([len(Xdata[0][1:]), 1])

    self.xsUpdateSetLimits()

  def updateTimeShiftsAndData(self, optimalShiftsDisModels):
    ysNewBSX = [[np.array([]) for s in range(self.nrSubj)] for b in range(self.nrBiomk)]
    self.updateTimeShifts(optimalShiftsDisModels)
    XshiftedBSX, _, _, _ = self.getData()
    for b in range(self.nrBiomk):
      self.Y_array[b] = self.predictBiomk(self.X_array[b])[b].reshape(-1, 1)
      for s in range(self.nrSubj):
        dysScoresXU = self.predictBiomk(XshiftedBSX[b][s])
        ysNewBSX[b][s] = dysScoresXU[:, b]
        assert ysNewBSX[b][s].shape[0] == self.Y[b][s].shape[0]

    self.Y = ysNewBSX

  def xsUpdateSetLimits(self):
    minX = np.float128(np.min([el for sublist in self.X_array for item in sublist for el in item]))
    maxX = np.float128(np.max([el for sublist in self.X_array for item in sublist for el in item]))
    self.updateMinMax(minX, maxX)

  def updateXvals(self, newXvalsSX, origXvalsSX):
    """ Update the X_array with the given values. Compare origXvalsSX (full) with self.X (containing missing vals)
    to be able to tell where there was missing data originally. """

    print('self.X_array[0][:10]', self.X_array[0][:10])

    newX_BSX = [0 for b in range(self.nrBiomk)]
    for b in range(self.nrBiomk):
      newX_BSX[b] = [0 for b in range(self.nrSubj)]
      for s in range(self.nrSubj):
        # remove the entries that are meant to be missing for this biomarker
        indToIncludeCurr = np.in1d(origXvalsSX[s], self.X[b][s])
        newX_BSX[b][s] = newXvalsSX[s][indToIncludeCurr]

        print('self.X[b][s]', self.X[b][s])
        print('self.Y[b][s]', self.Y[b][s])
        print('newXvalsSX[s]', newXvalsSX[s])
        print('newX_BSX[b][s]', newX_BSX[b][s])
        print('self.N_obs_per_sub[b][s]', self.N_obs_per_sub[b][s])
        assert self.N_obs_per_sub[b][s] == len(newX_BSX[b][s])

      newXarrayCurrBiomk = [np.float128(item) for sublist in newX_BSX[b] for item in sublist]
      assert len(self.X_array[b]) == len(newXarrayCurrBiomk)
      self.X_array[b] = np.array(newXarrayCurrBiomk).reshape([len(newXarrayCurrBiomk), 1])

    # reset time-shifts to 0 (acceleration is left unchanged to 1. not currently used in this model)
    self.params_time_shift[0, :] = 0
    # print('self.X_array[0][:10]', self.X_array[0][:10])
    # print(adsa)

    self.xsUpdateSetLimits()

  def checkNobsMatch(self, N_obs_per_sub2):
    for b in range(self.nrBiomk):
      for s in range(self.nrSubj):
        assert self.N_obs_per_sub[b][s] == N_obs_per_sub2[b][s]

  def convertLongToArray(self, Z, visitIndices):
    """
    takes a list Z of dimension [NR_BIOMK] x [NR_SUBJ] x array(NR_VISITS) and a similar list of
    visit indices where data is available. For instance, if for biomarker 2 subject 3 had data
    only in visits 0 and 2, then visitIndices[2][3] = array([0,2])
    :param Z:
    :param visitIndices:
    :return:
    Z_array - a biomarker-wise serialised version of Z, where len(Z[b]) = all possible elements
              in Z_array[b] dimension is [NR_BIOMK] x array(all_values_linearised)
    N_obs_per_sub - a list of dimensions [NR_BIOMKS] x array(NR_SUBJ), containing the number of
                    observations for each sbuject in each biomarker
    indFullToMissingArrayB - a list of indices which could map a potential array
                             Z_full with no missing entries to Z_array.
                             so Z_array = Z_full[indFullToMissingArrayB]
    """
    nrBiomk = len(Z)
    Z_array = [0 for b in range(nrBiomk)]
    N_obs_per_sub = [0 for b in range(nrBiomk)]

    indFullToMissingArrayB = [0 for b in range(nrBiomk)]


    for b in range(nrBiomk):
      # Creating 1d arrays of individuals' time points and observations
      Z_array[b] = np.array([np.float128(item) for sublist in Z[b] for item in sublist]).reshape(-1,1)
      N_obs_per_sub[b] = [len(Z[b][j]) for j in range(len(Z[b]))]

      visitsSoFar = 0
      indFullToMissingArrayS = [0 for s in range(len(Z[0]))]
      for s in range(len(Z[0])):
        indFullToMissingArrayS[s] = visitsSoFar + visitIndices[b][s]
        visitsSoFar += visitIndices[b][s].shape[0]

      indFullToMissingArrayB[b] = np.array([i for subIdx in indFullToMissingArrayS for i in subIdx])

    return Z_array, N_obs_per_sub, indFullToMissingArrayB

  def filterLongArray(self, Z_array, N_obs_per_sub, indSubj, visitIndices):
    """
    in longitudinal array (Z_array), filter ALL visits from SOME (indSubj) subjects.
    generally used to filter subjects with different disease

    :param Z_array: [NR_BIOMK] x array(NR_ELEM, 1)
    :param N_obs_per_sub:
    :param indSubj: boolean mask of indices to filter
    :return:
    filtZ: filtered Z_array
    indFullToMissingFilteredSubj: indices that map from Full Array to filtered array
    """
    nrBiomk = len(Z_array)
    filtZ = [0 for b in range(nrBiomk)]
    indFiltToMissingArrayB  = [0 for b in range(nrBiomk)]

    # print('visitIndices', visitIndices)
    # print(ads)

    # check that it's a boolean mask
    assert indSubj.shape[0] == len(N_obs_per_sub[0])

    for b in range(nrBiomk):
      idxFiltArray = []
      visitsSoFar = 0
      intIndicesSubj = np.where(indSubj)[0]
      indFiltToMissingArrayS = [0 for s in range(intIndicesSubj.shape[0])]

      for s in range(intIndicesSubj.shape[0]):
        # find array that can go from full to filtered Array
        idxFiltArray += list(range(int(np.sum(N_obs_per_sub[b][:intIndicesSubj[s]])),
          np.sum(N_obs_per_sub[b][:intIndicesSubj[s] + 1])))

        # find indices that can go from filtered- to filtered+missing_data Array
        indFiltToMissingArrayS[s] = visitsSoFar + visitIndices[b][intIndicesSubj[s]]
        visitsSoFar += visitIndices[b][intIndicesSubj[s]].shape[0]

      # remember Z_Array[b] has shape array(NR_ELEM, 1). Call special filter function
      print('b', b)
      print('indSubj', indSubj)
      print('np.array(idxFiltArray)', np.array(idxFiltArray))
      print('Z_array[b]', Z_array[b].shape)
      filtZ[b] = self.filterZarray(Z_array[b], np.array(idxFiltArray))

      # print('indFiltToMissingArrayS', indFiltToMissingArrayS)

      indFiltToMissingArrayB[b] = np.array([i for subIdx in
        indFiltToMissingArrayS for i in subIdx])

      # print('Z_array[b]', Z_array[b].shape)
      # print('filtZ[b]', filtZ[b].shape)
      # print('indFiltToMissingArrayB[b]', indFiltToMissingArrayB[b].shape)
      # print('indSubj.shape', indSubj.shape)
      # # print('indSubj', indSubj)
      # assert filtZ[b].shape[0] == indFiltToMissingArrayB[b].shape[0]
      # if b == 1:
      #   # print('N_obs_per_sub[0], ', N_obs_per_sub[0])
      #   # print('N_obs_per_sub[1], ', N_obs_per_sub[1])
      #
      #   print(ads)


    return filtZ, indFiltToMissingArrayB

  def filterZarray(self, Z_array, filterInd):
    # remember Z_Array[b] has shape array(NR_ELEM, 1). Hence define special filter
    if filterInd.shape[0] > 0:
      return Z_array[filterInd, :]
    else:
      return np.array([])

  def getXsMinMaxRange(self, nrPoints=50):
    return np.linspace(self.minScX, self.maxScX, nrPoints).reshape([-1, 1])

  def updateMinMax(self, minX, maxX):
    self.minX = minX
    self.maxX = maxX
    self.minScX = self.applyScalingX(self.minX)
    self.maxScX = self.applyScalingX(self.maxX)

  def applyScalingXzeroOneFwd(self, xs):
    return (xs - self.minScX) / \
           (self.maxScX - self.minScX)

  def applyScalingXzeroOneInv(self, xs):
    return xs * (self.maxScX - self.minScX) + self.minScX

  def getData(self):
    nrBiomk = len(self.X)
    nrSubj = len(self.X[0])
    XshiftedScaled = [[] for b in range(nrBiomk)]
    X_arrayScaled = [0 for b in range(nrBiomk)]

    for b in range(nrBiomk):
      for s in range(nrSubj):
        XshiftedCurrSubj = np.array([self.X_array[b][k][0] for k in range(int(np.sum(
          self.N_obs_per_sub[b][:s])), np.sum(self.N_obs_per_sub[b][:s + 1]))])

        XshiftedScaled[b] += [self.applyScalingX(XshiftedCurrSubj)]

        assert XshiftedScaled[b][s].shape[0] == self.X[b][s].shape[0]
        assert XshiftedScaled[b][s].shape[0] == self.Y[b][s].shape[0]

      X_arrayScaled[b] = self.applyScalingX(self.X_array[b])

    return XshiftedScaled, self.X, self.Y, X_arrayScaled


  def getSubShiftsLong(self):
    return self.applyScalingX(self.params_time_shift[0])

  def getMinMaxY_B(self, extraDelta=0):
    ''' get minimum and maximum of Ys per biomarker'''
    deltaB = (self.max_yB - self.min_yB) * extraDelta

    return self.min_yB - deltaB, self.max_yB + deltaB


