import matplotlib.pyplot as plt
import numpy as np
import sys
import scipy.optimize
import DPMModelGeneric
import scipy.stats


class SigmoidModel(DPMModelGeneric.DPMModelGeneric):
  plt.interactive(False)

  def __init__(self, X, Y, visitIndices, outFolder, plotter, names_biomarkers, params):
    super().__init__(X, Y, visitIndices, outFolder, plotter, names_biomarkers, params)

    minX = np.float(np.min([el for sublist in self.X_array for item in sublist for el in item]))
    maxX = np.float(np.max([el for sublist in self.X_array for item in sublist for el in item]))
    self.updateMinMax(minX, maxX)

    # self.parameters = [0 for b in range(self.nrBiomk)] # parameters of sigmoid trajectories
    # avgStdX = np.nanmean([np.std(self.X_array[b]) for b in range(self.nrBiomk) if np.std(self.X_array[b] != 0)])
    # print('avgStdX', avgStdX)
    # for b in range(self.nrBiomk):
    #
    #   minY = np.min(self.Y_array[b])
    #   maxY = np.max(self.Y_array[b])
    #   transitionTime = 20 * np.std(self.X_array[b])
    #   if transitionTime == 0:
    #     transitionTime = 20 * avgStdX # in months
    #   center = np.mean(self.X_array[b])
    #   trajParams = self.transfTrajParams(minY, transitionTime, center, maxY)
    #   variance = np.var(self.Y_array[b])
    #   self.parameters[b] = [trajParams, variance]

    # print('Y_array', self.Y_array)
    # print(ads)

    self.parameters = [0 for b in range(self.nrBiomk)]
    self.initialiseParams()

    scaledYarrayB = [self.applyScalingY(self.Y_array[b], b) for b in range(self.nrBiomk)]
    self.min_yB = np.array([np.min(scaledYarrayB[b].reshape(-1)) for b in range(self.nrBiomk)])
    self.max_yB = np.array([np.max(scaledYarrayB[b].reshape(-1)) for b in range(self.nrBiomk)])
    self.priors = params['priors']

  def initialiseParams(self):

    avgStdX = np.nanmean([np.std(self.X_array[b]) for b in range(self.nrBiomk) if np.std(self.X_array[b] != 0)])
    print('stdsB-nonZero', [np.std(self.X_array[b]) for b in range(self.nrBiomk) if np.std(self.X_array[b] != 0)])
    print('avgStdX', avgStdX)
    for b in range(self.nrBiomk):

      print('self.Y_array[b]', self.Y_array[b])
      minY = np.min(self.Y_array[b])
      maxY = np.max(self.Y_array[b])
      transitionTime = 20 * np.std(self.X_array[b])
      if transitionTime == 0:
        transitionTime = 20 * avgStdX # in months
      center = np.mean(self.X_array[b])
      trajParams = self.transfTrajParams(minY, transitionTime, center, maxY)
      variance = np.var(self.Y_array[b])
      self.parameters[b] = [trajParams, variance]


  def transfTrajParams(self, minY, transitionTime, center, maxY):
    """
    :param params: [minY, transitionTime, center, maxY]
    :return: [a,b,c,d] form where a is the min-max range, b = -16/(a*tt) c = center d = minX
    """

    a = maxY - minY
    b = 16 / (a * transitionTime)
    c = center
    d = minY
    return [a,b,c,d]

  def applyScalingX(self, x_data, biomk=0):
    return x_data

  def applyScalingY(self, y_data, biomk):
    return y_data

  def applyScalingYInv(self, y_data, biomk):
    return y_data

  def applyScalingYAllBiomk(self, biomksXB):
    return biomksXB

  def applyScalingXForward(self, x_data, biomk=0):
    return x_data

  def applyScalingXInv(self, x_data, biomk=0):
    return x_data


  def sigFunc(self, xs, theta):
    # print('theta', theta)
    return theta[0] * np.power((1 + np.exp(-theta[1] * (xs - theta[2]))), -1) + theta[3]

  def computePriorTrajOneBiomk(self, params):
    return (params[0][0] - self.priors['meanA']) ** 2 / self.priors['stdA'] + \
      (params[0][3] - self.priors['meanD']) ** 2 / self.priors['stdD'] - \
      scipy.stats.gamma.logpdf(params[0][1], a=self.priors['shapeB'],
        scale=1.0/self.priors['rateB'])

  def ssdTrajOneBiomk(self, params, X_arrayX, Y_arrayX, biomkIndex):
    """
    computes the log posterior for the current biomarker
    :param X_arrayX: linearised array of DPS scores for current biomarker
    :param Y_arrayX: linearised array of measurements for current biomarker
    :param params: [trajParams, variance] where params[0] - (4,) array of parameters for the sigmoid func
    :param biomkIndex:
    :return:
    """

    prior_traj = 0
    if self.priors is not None:
      '''
      a = maxY - minY
      b = 16 / (a * transitionTime)
      c = center
      d = minY
      '''
      prior_traj = self.computePriorTrajOneBiomk(params)



    # actually it's SSD + prior
    ssd =  np.sum((Y_arrayX - self.sigFunc(X_arrayX, params[0])) ** 2) + prior_traj
    grad = 0

    return ssd, grad

  def unpack_parameters(self, params):
    return params[0], params[1]

  def computeTrajParamsForTimeShifts(self):
    return self.parameters

  def estimVariance(self, params, X_arrayX, Y_arrayX):
    """
    computes the log posterior for the current biomarker
    :param X_arrayX: linearised array of DPS scores for current biomarker
    :param Y_arrayX: linearised array of measurements for current biomarker
    :param params: (4,) array of parameters for the sigmoid func
    :return:
    """

    # SSD/nrMeasurements
    variance = np.sum((Y_arrayX - self.sigFunc(X_arrayX, params)) ** 2)/Y_arrayX.shape[0]
    return variance

  def estimTrajParams(self):
    # Method for optimization of GP parameters (weights, length scale, amplitude and noise term)

    nrPerturbMax = 5

    for b in range(self.nrBiomk):
      objectiveFun = lambda params: self.ssdTrajOneBiomk([params, None], self.X_array[b],
                                                         self.Y_array[b], b)[0]
      initParams, initVariance = self.unpack_parameters(self.parameters[b])
      print('initParams', initParams)

      resStructMin = scipy.optimize.minimize(objectiveFun, initParams, method='Nelder-Mead',
                                          options={'disp': True, 'maxiter': len(initParams) * 500})



      # keep trying perturbed initial points until fitting is successful, up to a max nr of perturbations
      # p = 0
      # perturbParams = copy.deepcopy(initParams)
      stdPerturb = [0, 0.3, 10, 0.3]
      successList = np.zeros(nrPerturbMax, bool)
      # resStructMin = resStruct
      for p in range(nrPerturbMax):
        print('trying perturbation %d' % p)
        perturbParams = [np.random.normal(initParams[i], stdPerturb[i])
          for i in range(len(initParams))]
        resStruct = scipy.optimize.minimize(objectiveFun, perturbParams, method='Nelder-Mead',
                                              options={'disp': True, 'maxiter':len(initParams)*500})

        # p += 1
        successList += [resStruct.success]
        if resStruct.success and resStruct.fun < resStructMin.fun:
          resStructMin = resStruct

      # print(resStruct)
      # print(resStructMin)
      # print(adsa)

      if not resStructMin.success:
        print(resStructMin)
        print('self.Y_array[b]', self.Y_array[b])
        print('self.X_array[b]', self.X_array[b])

        print(das)

      variance = self.estimVariance(resStructMin.x, self.X_array[b], self.Y_array[b])
      # variance = self.varianceCTL

      # print('resStruct', resStruct)

      # if p >= 1:
      #   import pdb
      #   pdb.set_trace()

      self.parameters[b] = [resStructMin.x, variance]

  def shiftObjFunc(self, params, time_shift_one_sub, sub):
    # Input: X, Y and a biomarker's parameters, current time-shift estimates
    # Output: log-posterior and time-shift gradient

    # Shifting data according to current time-shift estimate
    loglik = 0
    timeShiftPriorMean = 0
    if self.priors is not None:
      timeShiftPriorStd = self.priors['timeShiftStd']
    else:
      timeShiftPriorStd = 20

    # print('timeShiftPriorStd', timeShiftPriorStd)
    # print(adas)

    for b in range(self.nrBiomk):
      trajParams, variance = self.unpack_parameters(params[b])

      Xdata = time_shift_one_sub + self.X_array[b][int(np.sum(self.N_obs_per_sub[b][:sub])): \
                                                   np.sum(self.N_obs_per_sub[b][:sub + 1])]
      Ydata = self.Y_array[b][int(np.sum(self.N_obs_per_sub[b][:sub])):np.sum(self.N_obs_per_sub[b][:sub + 1])]
      Ypred = self.sigFunc(Xdata, trajParams)

      prior_time_shift = (time_shift_one_sub - timeShiftPriorMean) ** 2 / timeShiftPriorStd
      loglik += - 0.5 * (np.sum((Ydata - Ypred) ** 2) / variance) - prior_time_shift

    return loglik

  def estimSubjShifts(self):
    init_params = self.params_time_shift.copy()
    init_params[0] = np.zeros(len(init_params[0]))

    init_params_time_only = init_params[0]

    ######## calculate subject-nonspecific terms
    optimal_params_time_only = np.zeros(init_params_time_only.shape)

    nrSubj = self.nrSubj
    for s in range(nrSubj):
      objectiveFun = lambda time_shift_one_sub: -self.shiftObjFunc(self.parameters,
                       time_shift_one_sub, s)[0]

      options = {'disp': True, 'xtol': 1e-3}
      resStruct = scipy.optimize.minimize(objectiveFun, init_params_time_only[s], method='Nelder-Mead',
                                          options={'disp': True})

      optimal_params_time_only[s] = resStruct.x

    convTimeOnlyToTimePlusAcc = lambda params_time_shift_only_shift: \
      np.concatenate((params_time_shift_only_shift.reshape(1, -1),
                      np.ones((1, params_time_shift_only_shift.shape[0]))), axis=0)
    optimal_params = convTimeOnlyToTimePlusAcc(optimal_params_time_only)

    # print('optimal_params', optimal_params)
    # print(adsa)

    for l in range(1):
      self.params_time_shift[l] = self.params_time_shift[l] + optimal_params[l]

    for i in range(self.nrBiomk):
      Xdata = np.array([[100]])
      for sub in range(self.nrSubj):
        temp = self.X_array[i][int(np.sum(self.N_obs_per_sub[i][:sub])):np.sum(self.N_obs_per_sub[i][:sub + 1])]
        shifted_temp = (temp + optimal_params[0][sub])
        Xdata = np.hstack([Xdata, shifted_temp.T])

      self.X_array[i] = Xdata[0][1:].reshape([len(Xdata[0][1:]), 1])

    minX = np.float128(np.min([el for sublist in self.X_array for item in sublist for el in item]))
    maxX = np.float128(np.max([el for sublist in self.X_array for item in sublist for el in item]))
    self.updateMinMax(minX, maxX)

  def log_posterior_time_shift_onebiomk_given_arrays(self, Xdata, Ydata, trajParams):
    # Input: X, Y and a biomarker's parameters, current time-shift estimates
    # Output: log-posterior and time-shift gradient

    trajParams, variance = self.unpack_parameters(trajParams)
    Ypred = self.sigFunc(Xdata, trajParams)
    loglik = 0.5 * (np.sum((Ydata - Ypred) ** 2) / variance)


    return loglik


  def Optimize(self, N_global_iterations, Plot=True):
    # Global optimizer (Sigmoid parameters + time shift)
    # fig = self.plotter.plotTraj(self)
    # fig.savefig('%s/allTraj%d0_%s.png' % (self.outFolder, 0, self.expName))

    # if self.plotter.plotTrajParams['isSynth']:
    #   fig2 = self.plotter.plotCompWithTrueParams(self, replaceFig=True)
    #   fig2.savefig('%s/compTrueParams%d0_%s.png' % (self.outFolder, 0, self.expName))

    for i in range(N_global_iterations):

      print("Optimizing time shift")
      self.estimSubjShifts()

      if Plot:
        fig = self.plotter.plotTraj(self)
        fig.savefig('%s/allTraj%d0_%s.png' % (self.outFolder, i + 1, self.expName))
        if self.plotter.plotTrajParams['isSynth']:
          fig2 = self.plotter.plotCompWithTrueParams(self)
          fig2.savefig('%s/compTrueParams%d0_%s.png' % (self.outFolder, i + 1, self.expName))

      # print(self.X_array)
      # print(adsa)

      self.estimTrajParams()
      print("Current penalty parameters: ")

      if Plot:
        fig = self.plotter.plotTraj(self)
        fig.savefig('%s/allTraj%d1_%s.png' % (self.outFolder, i + 1, self.expName))
        fig2 = self.plotter.plotCompWithTrueParams(self)
        fig2.savefig('%s/compTrueParams%d1_%s.png' % (self.outFolder, i + 1, self.expName))

  def predictBiomkWithParams(self, newX, params):

    deltaX = 5 * (self.maxScX - self.minScX)
    if not (self.minScX - deltaX <= np.min(newX) <= self.maxScX + deltaX):
      print('newX', newX)
      print('self.minScX', self.minScX)
      print('self.maxScX', self.maxScX)
      raise ValueError('newX not in the correct range. check the scaling')

    xsScaled = self.applyScalingXForward(newX.reshape(-1), biomk=0)  # arbitrary space ->[0,1]

    predictedBiomksXB = np.zeros((xsScaled.shape[0], self.nrBiomk))
    for b in range(self.nrBiomk):
      trajParams, variance = self.unpack_parameters(params[b])
      predictedBiomksXB[:, b] = self.sigFunc(xsScaled, trajParams)


    return self.applyScalingYAllBiomk(predictedBiomksXB)

  def predictBiomk(self, newX):
    return self.predictBiomkWithParams(newX, self.parameters)

  def sampleTrajPost(self, newX, biomarker, nrSamples):
    '''
    sample trajectory posterior

    :param newX:
    :param biomarker:
    :param nrSamples:
    :return:
    '''

    xsScaled = self.applyScalingXForward(newX.reshape(-1))  # arbitrary space ->[0,1]

    trajSamplesXS = np.zeros((xsScaled.shape[0], nrSamples))
    trajParams, variance = self.unpack_parameters(self.parameters[biomarker])

    for i in range(nrSamples):
      trajSamplesXS[:, i] = self.sigFunc(xsScaled, trajParams)

    return self.applyScalingY(trajSamplesXS, biomarker)
