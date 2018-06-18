### Gaussian process-based disease progression modeling and time-shift estimation.
#
#  - The software iteratively estimates monotonic progressions for each biomarker and realigns the individual observations in time
#   Basic usage:
#
#       model = GP_progression_model.GP_progression_model(input_X,input_N,N_random_features)
#
#   X and Y should be A list of biomarkers arrays. Each entry "i" of the list is a list of individuals' observations for the biomarker i
#   The monotonicity is enforced by the parameter self.penalty (higher -> more monotonic)
#
# - The class comes with an external method for transforming a given .csv file in the required input X and Y:
#
#       X,Y,list_biomarker = GP_progression_model.convert_csv(file_path)
#
# - The method Save(folder_path) saves the model parameters to an external folder, that can be subsequently read with the
# method Load(folder_path)
#
# - Optimization can be done with the method Optimize:
#
#       model.Optimize()
#
# This software is based on the publication:
#
# Disease Progression Modeling and Prediction through Random Effect Gaussian Processes and Time Transformation
# Marco Lorenzi, Maurizio Filippone, Daniel C. Alexander, Sebastien Ourselin
# arXiv:1701.01668
#
# Gaussian process regression based on random features approximations is based on the paper:
#
# Random Feature Expansions for Deep Gaussian Processes (ICML 2017, Sydney)
# K. Cutajar, E. V. Bonilla, P. Michiardi, M. Filippone
# arXiv:1610.04386


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sys
import scipy.optimize
import DPMModelGeneric

class GP_progression_model(DPMModelGeneric.DPMModelGeneric):
    plt.interactive(False)
    def __init__(self, X,Y, visitIndices, outFolder, plotter, names_biomarkers, params):
      super().__init__(X,Y, visitIndices, outFolder, plotter, names_biomarkers, params)

      self.mean_std_X = []
      self.mean_std_Y = []
      self.max_X = []
      self.max_Y = []

      N_rnd_features = int(3) # Number of random features for kernel approximation
      self.priors = params['priors']
      self.group = []
      self.N_rnd_features = int(N_rnd_features)

      self.rescale()


      self.minX = np.float128(np.min([el for sublist in self.X_array for item in sublist for el in item]))
      self.maxX = np.float128(np.max([el for sublist in self.X_array for item in sublist for el in item]))

      self.minScX = self.applyScalingX(self.minX)
      self.maxScX = self.applyScalingX(self.maxX)

      nrBiomk = len(self.X)
      scaledYarrayB = [self.applyScalingY(self.Y_array[b], b) for b in range(nrBiomk)]
      self.min_yB = np.array([np.min(scaledYarrayB[b].reshape(-1)) for b in range(nrBiomk)])
      self.max_yB = np.array([np.max(scaledYarrayB[b].reshape(-1)) for b in range(nrBiomk)])

      # Number of derivative points uniformely distributed on the X axis
      self.N_Dpoints = 10
      self.DX = np.linspace(self.minX,self.maxX,self.N_Dpoints).reshape([self.N_Dpoints,1])

      # Initializing random features for kernel approximation
      self.perturbation_Omega = np.random.randn(self.N_rnd_features)

      self.init_params_var = []
      self.init_params_full = []

      # Monotonicity constraint (higher -> more monotonic)
      self.penalty = []

      # Initializing fixed effect parameters per biomarkers to default values
      for l in range(self.nrBiomk):
          self.init_params_var.append(np.concatenate([ np.zeros([ self.N_rnd_features]) - 1, np.zeros([ self.N_rnd_features]) - 1, np.zeros([ 2 * self.N_rnd_features]) , np.zeros([ 2 * self.N_rnd_features])]))
          sigma = 0
          length_scale = -3
          eps = -4
          self.init_params_full.append(np.concatenate([self.init_params_var[l], np.array([sigma]), np.array([length_scale]), np.array([eps])]))
          self.penalty.append(1)

      self.parameters = []
      for l in range(self.nrBiomk):
          self.parameters.append(self.init_params_full[l])

      # Initializing individuals random effects
      self.rand_parameters = []
      self.rand_parameter_type = []

      for biom in range(self.nrBiomk):
          self.rand_parameter_type.append([])
          self.rand_parameters.append([])
          for sub in range(self.nrSubj):
              if self.N_obs_per_sub[biom][sub]==0:
                  self.rand_parameter_type[biom].append(0)
                  self.rand_parameters[biom].append(0)
              elif self.N_obs_per_sub[biom][sub] < 3:
                  self.rand_parameter_type[biom].append(1)
                  self.rand_parameters[biom].append(0)
              else:
                  self.rand_parameter_type[biom].append(2)
                  self.rand_parameters[biom].append([0,0])

              obs = np.array([self.X_array[biom][k][0] for k in range(int(np.sum(self.N_obs_per_sub[biom][:sub])),
                                                             np.sum(self.N_obs_per_sub[biom][:sub + 1]))])



    def applyScalingX(self, x_data, biomk=0):
      scaleX = self.max_X[biomk] * self.mean_std_X[biomk][1]
      return scaleX * x_data + self.mean_std_X[biomk][0]

    def applyScalingY(self, y_data, biomk):
      scaleY = self.max_Y[biomk] * self.mean_std_Y[biomk][1]
      return scaleY * y_data + self.mean_std_Y[biomk][0]

    def applyScalingYInv(self, y_data, biomk):
      scaleY = self.max_Y[biomk] * self.mean_std_Y[biomk][1]
      return (y_data - self.mean_std_Y[biomk][0])/scaleY

    def applyScalingYAllBiomk(self, biomksXB):
      biomksNewXB = np.zeros(biomksXB.shape)
      for b in range(self.nrBiomk):
        biomksNewXB[:, b] = self.applyScalingY(biomksXB[:, b], b)

      return biomksNewXB

    def applyScalingYAllBiomkInv(self, biomksXB):
      biomksNewXB = np.zeros(biomksXB.shape)
      for b in range(self.nrBiomk):
        biomksNewXB[:, b] = self.applyScalingYInv(biomksXB[:, b], b)

      return biomksNewXB

    def updateMinMax(self, minX, maxX):
      self.minX = minX
      self.maxX = maxX
      self.minScX = self.applyScalingX(self.minX)
      self.maxScX = self.applyScalingX(self.maxX)

    def applyScalingXForward(self, x_data, biomk=0):
      scaleX = self.max_X[biomk] * self.mean_std_X[biomk][1]
      return (x_data - self.mean_std_X[biomk][0])/scaleX

    def applyGivenScalingY(self, y_data, meanY, stdY):
      return (y_data - meanY)/stdY

    def applyScalingXzeroOneFwd(self, xs):
      return (xs - self.minScX) / \
        (self.maxScX - self.minScX)

    def applyScalingXzeroOneInv(self, xs):
      return xs * (self.maxScX - self.minScX) + self.minScX

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
          # print('self.X[b][s]', self.X[b][s])
          # print('origXvalsSX[s]', origXvalsSX[s])
          # print('indToIncludeCurr', indToIncludeCurr)
          # print('newXvalsSX[s]', newXvalsSX[s])
          # print(adsa)
          newX_BSX[b][s] = newXvalsSX[s][indToIncludeCurr]

          assert self.N_obs_per_sub[b][s] == len(newX_BSX[b][s])


        newXarrayCurrBiomk = [np.float128(item) for sublist in newX_BSX[b] for item in sublist]
        assert len(self.X_array[b]) == len(newXarrayCurrBiomk)
        self.X_array[b] = np.array(newXarrayCurrBiomk).reshape([len(newXarrayCurrBiomk),1])


      # reset time-shifts to 0 (acceleration is left unchanged to 1. not currently used in this model)
      self.params_time_shift[0, :] = 0
      # print('self.X_array[0][:10]', self.X_array[0][:10])
      # print(adsa)

      # also remove the transformation of X. try to keep it standard
      self.mean_std_X = []
      self.max_X = []
      for b in range(self.nrBiomk):
        self.mean_std_X.append([0, 1])
        self.max_X.append(1)

      minX = np.float128(np.min([el for sublist in self.X_array for item in sublist for el in item]))
      maxX = np.float128(np.max([el for sublist in self.X_array for item in sublist for el in item]))
      self.updateMinMax(minX, maxX)
      self.DX = np.linspace(self.minX, self.maxX, self.N_Dpoints).reshape([self.N_Dpoints, 1])


    def rescale(self):
      # Standardizes X and Y axes and saves the rescaling parameters for future output
      # Raz modification - made the scaling of every X[l] the same for every biomarker l

      # for l in range(self.nrBiomk):
      #   self.X_array[l] = np.array(self.X_array[l]).reshape([len(self.X_array[l]),1])
      #   self.Y_array[l] = np.array(self.Y_array[l]).reshape([len(self.Y_array[l]), 1])


      XarrayAllBiomk = np.array([x2 for l in self.X_array for x2 in list(l)])
      meanAll = np.mean(XarrayAllBiomk)
      stdAll = np.std(XarrayAllBiomk)

      print('self.Y_array', self.Y_array)
      print('self.Y_array', self.Y_array)

      # print('meanAll', meanAll)
      # print('stdAll', stdAll)
      # print('maxXAll', maxXAll)
      # print(adsa)

      for l in range(self.nrBiomk):
        # sd = np.std(self.X_array[l])
        if (stdAll > 0):
          self.mean_std_X.append([meanAll, stdAll])
        else:
          self.mean_std_X.append([meanAll, 1])

        self.mean_std_Y.append([np.mean(self.Y_array[l]), np.std(self.Y_array[l])])
        self.X_array[l] = (self.X_array[l] - self.mean_std_X[l][0])/ self.mean_std_X[l][1]
        self.Y_array[l] = (self.Y_array[l] - self.mean_std_Y[l][0]) / self.mean_std_Y[l][1]
        self.max_Y.append(np.max(self.Y_array[l]))
        self.Y_array[l] = self.Y_array[l] / self.max_Y[l]

      XarrayAllBiomk = np.array([x2 for l in self.X_array for x2 in list(l)])
      maxXAll = np.max(XarrayAllBiomk)

      for l in range(self.nrBiomk):
        if maxXAll > 0:
          self.max_X.append(maxXAll) # wrong, should compute maxXAll after the first normalisation
          # self.max_X.append(1)
        else:
          self.max_X.append(1)

        self.X_array[l] = self.X_array[l]/self.max_X[l]



    def Set_penalty(self, penalty):
        for l in range(self.nrBiomk):
            self.penalty[l] = penalty

    def Reset_parameters(self):
        # Reset paramters to standard values
        self.init_params_var = []
        self.init_params_full = []
        for l in range(self.nrBiomk):
            self.init_params_var.append(np.concatenate([ np.zeros([ self.N_rnd_features]) - 1, np.zeros([ self.N_rnd_features]) - 1, np.zeros([ 2 * self.N_rnd_features]) , np.zeros([ 2 * self.N_rnd_features])]))
            sigma = -1

            if (self.maxX==self.minX):
                length_scale = 0
            else:
                length_scale = np.log((self.maxX - self.minX)/8)

            eps = -3
            self.init_params_full.append(np.concatenate([self.init_params_var[l], np.array([sigma]), np.array([length_scale]), np.array([eps])]))

        self.parameters = []
        for l in range(self.nrBiomk):
            self.parameters.append(self.init_params_full[l])

    def phi(self, X, omega, sigma):
        # Random feature expansion in cosine a sine basis
        # print('sigma', sigma)
        # print('omega', omega)
        # print('X', X)
        # print(adsa)
        return np.sqrt(sigma) / np.sqrt(len(omega)) * np.concatenate([np.cos(omega * X), np.sin(omega * X)], axis=1)

    def Dphi(self, X, omega, sigma):
        # Derivative of the random feature expansion with respect to X
        return np.sqrt(sigma) / np.sqrt(len(omega)) * np.concatenate([- np.sin(omega * X) * omega, np.cos(omega * X) * omega], axis=1)

    def DDphi_omega(self, X, omega, sigma):
        # Double derivative of the random feature expansion with respect to X and omega
        return np.sqrt(sigma) / np.sqrt(len(omega)) * \
               np.concatenate([- np.cos(omega * X) * omega * X - np.sin(omega * X)  ,\
                               - np.sin(omega * X) * omega * X + np.cos(omega * X) ], axis=1)

    def Dphi_omega(self, X, omega, sigma):
        # Derivative of the random feature expansion with respect to omega
        return np.sqrt(sigma) / np.sqrt(len(omega)) * np.concatenate([- np.sin(omega * X) * X , np.cos(omega * X) * X ], axis=1)

    def Dphi_time_shift(self, X, omega, sigma):
        # Derivative of the random feature expansion with respect to time-shift parameters
        return np.sqrt(sigma) / np.sqrt(len(omega)) * np.concatenate([- np.sin(omega * X) , np.cos(omega * X) ], axis=1)

    def basis(self, X, sigma, random_weights):
        if len(X.shape) == 1:
          raise ValueError('shape of X needs to be array(NR_ELEM, 1). '
                               'Make sure you apply X.reshape(-1,1)')
        return self.phi(X, random_weights, sigma)

    def Dbasis_omega(self, X, sigma, random_weights):
        return self.Dphi_omega(X, random_weights, sigma)

    def Dbasis(self, X, sigma, random_weights):
        return self.Dphi(X, random_weights, sigma)

    def DDbasis_omega(self, X, sigma, random_weights):
        return self.DDphi_omega(X, random_weights, sigma)

    def Dbasis_time_shift(self, X, sigma, random_weights):
        return self.Dphi_time_shift(X, random_weights, sigma)

    def KL(self, s_omega, m_omega,  s, m, l):
        # Kullback Leibler divergence for random features and weights distributions
        termOmega = np.sum(0.5 * (np.log(1 / (l * s_omega)) - 1 + (l * s_omega) + l * m_omega ** 2))
        termW = np.sum(0.5 * (np.log(1 / s) - 1 + s + m ** 2))
        return termW + termOmega

    def unpack_parameters(self, params):
        s_omega = params[:self.N_rnd_features].reshape([self.N_rnd_features,1])
        m_omega = params[self.N_rnd_features:2*self.N_rnd_features].reshape([self.N_rnd_features,1])
        s_w = params[2*self.N_rnd_features:4*self.N_rnd_features].reshape([2*self.N_rnd_features,1])
        m_w = params[4*self.N_rnd_features:6*self.N_rnd_features].reshape([2*self.N_rnd_features,1])
        sigma = params[6*self.N_rnd_features]
        length_scale = params[6*self.N_rnd_features+1]
        eps = params[6*self.N_rnd_features+2]
        return s_omega, m_omega, s_w, m_w, sigma, length_scale, eps

    def log_posterior(self, X,Y, N, perturbationW, params, penalty):
        # NOT USED
        # Returns log-posterior for a given set of biomarker's paramters and a random perturbation of the weights W
        s_omega, m_omega, s_w, m_w, sigma, l, eps = self.unpack_parameters(params)
        s_omega = np.exp(s_omega)
        s_w = np.exp(s_w)
        l = np.exp(l)
        sigma = np.exp(sigma)
        eps= np.exp(eps)

        W = np.multiply(perturbationW, np.sqrt(s_w)) + m_w
        Omega = 1/np.sqrt(l) * self.perturbation_Omega
        output = self.basis(X, sigma, Omega)
        Doutput = self.Dbasis(self.DX, sigma, Omega)
        Kullback_Leibler = self.KL( s_omega, m_omega, s_w, m_w, l)

        Dterm = np.sum(penalty * np.dot(Doutput, W) - np.log(1 + np.exp(penalty * np.dot(Doutput, W))))

        prior = (eps - 0.3) ** 2 / 1e-2 + (sigma - 0.5) ** 2 / 1e-2

        return -0.5 *  ( np.log(2 * np.pi * eps) + np.sum((Y - np.dot(output,W))**2)/eps) - Kullback_Leibler  - prior + Dterm

    def log_posterior_grad(self, X,Y, N, perturbationW, params, penalty):
        # Input: X, Y and a biomarker's parameters, random perturbation of the weights W
        # Output: log-posterior and parameters gradient
        s_omega, m_omega, s_w, m_w, sigma, l, eps = self.unpack_parameters(params)
        s_omega = np.exp(s_omega)
        s_w = np.exp(s_w)
        l = np.exp(l)
        sigma = np.exp(sigma)
        eps= np.exp(eps)

        W = np.multiply(perturbationW, np.sqrt(s_w)) + m_w
        Omega = 1/np.sqrt(l) * self.perturbation_Omega
        output = self.basis(X, sigma, Omega)
        Doutput = self.Dbasis(self.DX, sigma, Omega)
        Kullback_Leibler = self.KL( s_omega, m_omega, s_w, m_w, l)

        # Modify the prior length scale according to current X range
        prior_length_scale_mean = (self.maxX-self.minX)*self.priors['prior_length_scale_mean_ratio']
        prior_length_scale_std = self.priors['prior_length_scale_std']

        prior_sigma_mean = self.priors['prior_sigma_mean']
        prior_sigma_std = self.priors['prior_sigma_std']

        prior_eps_mean = self.priors['prior_eps_mean']
        prior_eps_std = self.priors['prior_eps_std']

        Dterm = np.sum(penalty * np.dot(Doutput, W) - np.log(1 + np.exp(penalty * np.dot(Doutput, W))))
        prior = (eps - prior_eps_mean) ** 2 / prior_eps_std + (sigma - prior_sigma_mean) ** 2 / prior_sigma_std + (l - prior_length_scale_mean)**2/prior_length_scale_std

        if ~np.isfinite(Dterm):
          # import pdb
          # pdb.set_trace()
          return np.inf, np.repeat(0, len(params)).flatten(), 0


        posterior = -0.5 *  ( np.log(2 * np.pi * eps) + np.sum((Y - np.dot(output,W))**2)/eps) - \
          Kullback_Leibler  - prior + Dterm

        # Derivative of weights mean ad sd
        d_m_w = np.dot(((Y - np.dot(output,W))).T,output)/eps + penalty * np.sum(Doutput,0) \
                - np.multiply(1/(1 + np.exp(penalty * np.dot(Doutput, W))), np.exp(penalty * np.dot(Doutput, W))*penalty * np.dot(Doutput, W)).T.dot(Doutput)\
                -  m_w.T

        d_s_w = np.multiply(np.dot(((Y - np.dot(output,W))).T,output),0.5*np.multiply(perturbationW, np.sqrt(s_w)).T)/eps \
                + np.multiply(penalty * np.sum(Doutput,0), 0.5*np.multiply(perturbationW, np.sqrt(s_w)).T)\
                - np.multiply(0.5*np.multiply(perturbationW, np.sqrt(s_w)).T, \
                np.multiply(1/(1 + np.exp(penalty * np.dot(Doutput, W))), np.exp(penalty * np.dot(Doutput, W))*penalty * np.dot(Doutput, W)).T.dot(Doutput))\
                + 0.5 * (-1 + s_w).T


        Doutput_omega = self.Dbasis_omega(X, sigma, Omega)
        DDoutput_omega = self.DDbasis_omega(self.DX, sigma, Omega)

        grad_prod = - 0.5 * np.dot(np.multiply(Doutput_omega, np.tile(self.perturbation_Omega, 2) * 1 / np.sqrt(l)), W)
        grad2_prod = - 0.5 * np.dot(np.multiply(DDoutput_omega, np.tile(self.perturbation_Omega, 2) * 1 / np.sqrt(l)), W)

        # Derivative of length scale
        d_l = - 2* np.sum( np.multiply(((Y - np.dot(output,W)))/eps, grad_prod)) \
              + penalty  *  np.sum( grad2_prod ) \
              - np.sum(np.multiply(np.multiply(np.exp(penalty * np.dot(Doutput, W)), 1 / (1 + np.exp(penalty * np.dot(Doutput, W)))), \
            penalty * grad2_prod))\
            -  2* (l - prior_length_scale_mean)/prior_length_scale_std * l

        # Derivative of amplitude
        d_sigma = + np.sum(np.multiply(((Y - np.dot(output,W))).T/eps,np.dot(output,W).T/np.sqrt(sigma))) * np.sqrt(sigma) \
                  -  0.5* penalty  *  np.sum(np.dot(Doutput, W)) \
                  + np.sum(np.multiply( np.multiply(np.exp(penalty * np.dot(Doutput, W)), 1 / (1 + np.exp(penalty * np.dot(Doutput, W)))), \
                                        0.5 * penalty * (np.dot(Doutput, W))))\
                  - 2* (sigma - prior_sigma_mean)/prior_sigma_std * sigma

        # Derivative of noise term
        d_eps = + 0.5 *  ( 1 + np.sum((Y - np.dot(output,W))**2)/eps) - 2* (eps - prior_eps_mean) / prior_eps_std * eps

        # # Derivative of penalization parameter
        # d_penalty = np.sum(np.dot(Doutput, W)) \
        #             - np.sum( np.multiply( np.multiply(np.dot(Doutput, W),np.exp(penalty * np.dot(Doutput, W))), \
        #                         1/(1 + np.exp(penalty * np.dot(Doutput, W)))))

        d_penalty = 0

        return posterior, np.hstack([np.repeat(0,len(s_omega)).flatten(), np.repeat(0,len(m_omega)).flatten(), d_s_w.flatten(), d_m_w.flatten(),  np.array([d_sigma]), np.array(d_l), np.array([d_eps])]), d_penalty

    def stochastic_grad_manual(self, paramsB, X_arrayB, Y_arrayB, fixSeed=False):
        # Stochastic gradient of log-posterior with respect ot given parameters
        # Default number of MC samples is 100

        if fixSeed:
          np.random.seed(1)
          # print(ads)

        loglik_list = []
        MC_grad_list = []
        grad_penalty_list = []
        for b in range(self.nrBiomk):
            output_loglik, output_MC_grad, output_grad_penalty = \
                self.stochastic_grad_manual_onebiomk(paramsB[b], X_arrayB[b], Y_arrayB[b],
                self.penalty[b], fixSeed=False)
            # current_params = paramsB[b]
            # current_X = X_arrayB[b]
            # current_Y = Y_arrayB[b]
            # MC_grad = np.zeros(len(paramsB[b]))
            # output_grad_penalty.append(0)
            # loglik = 0
            # for j in range(100):
            #     perturbation_W = np.random.randn( 2 * self.N_rnd_features).reshape(
            #       [ 2*self.N_rnd_features,1])
            #     objective_cost_function = lambda params: self.log_posterior_grad(
            #       current_X, current_Y,self.N_rnd_features, perturbation_W, params,
            #       self.penalty[b])
            #     value, grad, grad_penalty = objective_cost_function(current_params)
            #     MC_grad = MC_grad - grad
            #     loglik = loglik - value
            #     output_grad_penalty[b] = output_grad_penalty[b] - grad_penalty

            loglik_list.append(output_loglik)
            MC_grad_list.append(output_MC_grad)
            grad_penalty_list.append(output_grad_penalty)


        return loglik_list, MC_grad_list, grad_penalty_list

    def stochastic_grad_manual_onebiomk(self, params, X_array, Y_array, penalty, fixSeed=False):
        # Stochastic gradient of log-posterior with respect ot given parameters
        # Default number of MC samples is 100

        if fixSeed:
          np.random.seed(1)
          # print(ads)

        current_params = params
        current_X = X_array
        current_Y = Y_array
        MC_grad = np.zeros(len(params))
        output_grad_penalty = 0
        loglik = 0
        for j in range(100):
            perturbation_W = np.random.randn( 2 * self.N_rnd_features).reshape(
              [ 2*self.N_rnd_features,1])
            objective_cost_function = lambda params: self.log_posterior_grad(
              current_X, current_Y,self.N_rnd_features, perturbation_W, params,
              penalty)
            value, grad, grad_penalty = objective_cost_function(current_params)
            MC_grad = MC_grad - grad
            loglik = loglik - value
            output_grad_penalty = output_grad_penalty - grad_penalty
        output_MC_grad = MC_grad/100
        output_loglik = loglik/100
        output_grad_penalty = output_grad_penalty/100


        return output_loglik, output_MC_grad, output_grad_penalty



    def Adadelta(self, Niterat, objective_grad, learning_rate, init_params, output_grad_penalty = False):
        # Adadelta optimizer
        params = []
        diag = []

        if output_grad_penalty:
            param_penalty = []
            diag_penalty = []

        for l in range(self.nrBiomk):
            params.append(init_params[l].copy())
            diag.append(np.zeros(len(params[l])))
            if output_grad_penalty:
                param_penalty.append(0)
                diag_penalty.append(0)

        epsilon = 1e-8

        for i in range(Niterat):
          fun_value, fun_grad, fun_grad_penalty = objective_grad(params)
          print('fun_value', fun_value)
          # print('fun_grad', len(fun_grad), fun_grad[0].shape fun_grad)
          # print(ads)

          for l in range(self.nrBiomk):
            # diag[l] = 0.9 * diag[l] + 0.1 * fun_grad[l] ** 2
            diag[l] = 0.9 * diag[l] + 0.1 * fun_grad[l] ** 2
            params[l] = params[l] - np.multiply(learning_rate * fun_grad[l], 1 / np.sqrt(diag[l] + epsilon))

            if output_grad_penalty:
              diag_penalty[l] = 0.9 * diag_penalty[l] + 0.1 * fun_grad_penalty[l] ** 2
              param_penalty[l] = param_penalty[l] - learning_rate * fun_grad_penalty[l]/ np.sqrt(diag_penalty[l] + epsilon)

          print(i, end=' ')
          sys.stdout.flush()

          for l in range(self.nrBiomk):
            self.parameters[l] = params[l]

            if output_grad_penalty:
              self.penalty [l]= param_penalty[l]

        print('final func value', fun_value)
        for b in range(self.nrBiomk):
          s_omega, m_omega, s_w, m_w, sigma, l, eps = self.unpack_parameters(np.array(self.parameters[b]))
          s_omega = np.exp(s_omega)
          s_w = np.exp(s_w)
          l = np.exp(l)
          sigma = np.exp(sigma)
          eps= np.exp(eps)
          print('b%d l sigma eps' % b, [l, sigma, eps])

        # import pdb
        # pdb.set_trace()

    def Optimize_GP_parameters(self, optimize_penalty = False, Niterat = 10):
        # Method for optimization of GP parameters (weights, length scale, amplitude and noise term)
        self.DX = np.linspace(self.minX, self.maxX, self.N_Dpoints).reshape([self.N_Dpoints, 1])
        self.Reset_parameters()
        objective_grad = lambda params: self.stochastic_grad_manual(params, self.X_array, self.Y_array)
        self.Adadelta(Niterat, objective_grad, 0.05, self.parameters,
          output_grad_penalty = optimize_penalty)

    def Optimize_GP_parameters_Raz(self, optimize_penalty = False, Niterat = 10):
      # Method for optimization of GP parameters (weights, length scale, amplitude and noise term)
      self.DX = np.linspace(self.minX, self.maxX, self.N_Dpoints).reshape([self.N_Dpoints, 1])
      self.Reset_parameters()

      encapsParams = lambda par: np.concatenate((np.zeros(2 * self.N_rnd_features) - 1, par))
      decapsParams = lambda par: par[2 * self.N_rnd_features:]

      for l in range(self.nrBiomk):
        objFuncCurrBiomk = lambda params: self.stochasticObjFuncOneBiomkRaz(params,
          self.X_array[l], self.Y_array[l], self.penalty[l])

        print('objFuncCurrBiomk', objFuncCurrBiomk(decapsParams(self.parameters[l])))
        # resStruct = scipy.optimize.minimize(objFuncCurrBiomk, decapsParams(self.parameters[l]), method='Powell',
        #   jac=True, options={'disp': True, 'xatol':1e+0, 'adaptive':True, 'maxiter':100, 'maxfev':300, 'eps':1e+1})

        resStruct = scipy.optimize.minimize(objFuncCurrBiomk, decapsParams(self.parameters[l]), method='CG',
          jac=True, options={'disp': True, 'maxiter':100})

        self.parameters[l] = encapsParams(resStruct.x)

    def stochasticObjFuncOneBiomkRaz(self, current_params, current_X, current_Y, current_penalty):
      # Stochastic gradient of log-posterior with respect ot given parameters
      # Default number of MC samples is 100
      output_MC_grad = []
      output_loglik = []
      output_grad_penalty = 0

      encapsParams = lambda par: np.concatenate((np.zeros(2 * self.N_rnd_features) - 1, par))
      decapsParams = lambda par: par[2 * self.N_rnd_features:]

      MC_grad = np.zeros(len(current_params))
      loglik = 0
      nrPerturb = 100
      for j in range(nrPerturb):
          perturbation_W = np.random.randn( 2 * self.N_rnd_features).reshape(
            [ 2*self.N_rnd_features,1])
          objective_cost_function = lambda params: self.log_posterior_grad(
            current_X, current_Y,self.N_rnd_features, perturbation_W, params,
            current_penalty)

          value, grad, grad_penalty = objective_cost_function(encapsParams(current_params))

          # print('value, current_params, grad', value, current_params, grad)

          if ~np.isfinite(grad).all():
            import pdb
            pdb.set_trace()

          loglik = loglik - value
          MC_grad = MC_grad - decapsParams(grad)
          output_grad_penalty = output_grad_penalty - grad_penalty

      return loglik/nrPerturb, MC_grad/(nrPerturb)


    # def log_posterior_time_shift(self, params, params_time_shift):
    #     # Input: X, Y and a biomarker's parameters, current time-shift estimates
    #     # Output: log-posterior and time-shift gradient
    #     loglik =  0
    #     Gradient = []
    #     for l2 in range(2):
    #         Gradient.append(np.zeros(self.nrSubj, np.float128))
    #
    #     # Shifting data according to current time-shift estimate
    #     for i in range(self.nrBiomk):
    #         Xdata = np.array([[1e10]])
    #         Ydata = np.array([[1e10]])
    #         for sub in range(self.nrSubj):
    #             temp = self.X_array[i][int(np.sum(self.N_obs_per_sub[i][:sub])):np.sum(self.N_obs_per_sub[i][:sub+1])]
    #             shifted_temp = (temp * params_time_shift[1][sub] + params_time_shift[0][sub])
    #             Xdata = np.hstack([Xdata,shifted_temp.T])
    #             tempY = self.Y_array[i][int(np.sum(self.N_obs_per_sub[i][:sub])):np.sum(self.N_obs_per_sub[i][:sub + 1])]
    #             Ydata = np.hstack([Ydata, tempY.T])
    #
    #         Xdata = Xdata[0][1:].reshape([len(Xdata[0][1:]),1])
    #         Ydata = Ydata[0][1:].reshape([len(Ydata[0][1:]), 1])
    #
    #         s_omega, m_omega, s_w, m_w, sigma, l, eps = self.unpack_parameters(params[i])
    #         s_omega = np.exp(s_omega)
    #         s_w = np.exp(s_w)
    #         l = np.exp(l)
    #         sigma = np.exp(sigma)
    #         eps = np.exp(eps)
    #
    #         perturbation_zero_W = np.zeros(int(2 * self.N_rnd_features)).reshape([2 * self.N_rnd_features, 1])
    #         W = np.multiply(perturbation_zero_W, np.sqrt(np.exp(s_w))) + m_w
    #         Omega = 1 / np.sqrt(l) * self.perturbation_Omega
    #
    #         output = self.basis(Xdata, sigma, Omega)
    #         Doutput_time_shift = self.Dbasis_time_shift(Xdata, sigma, Omega)
    #
    #         Doutput = self.Dbasis(self.DX, sigma, Omega)
    #         Kullback_Leibler = self.KL(s_omega, m_omega, s_w, m_w, l)
    #         Dterm = np.sum(
    #             np.log(2) - self.penalty[i] * np.dot(Doutput, W) / 2 + (self.penalty[i] * np.dot(Doutput, W)) ** 2 / 8)
    #         prior = (eps - 0.3) ** 2 / 1e-2 + (sigma - 0.5) ** 2 / 1e-2   # + (l - np.log(0.2))**2/1e-0
    #         prior_time_shift = np.sum((params_time_shift[0] - 0)**2/1e-0)
    #
    #         loglik = loglik - 0.5 * (
    #         np.log(2 * np.pi * eps) + np.sum((self.Y_array[i] - np.dot(output, W)) ** 2) / eps) - prior - Dterm - Kullback_Leibler - prior_time_shift
    #
    #         temp = np.multiply(Doutput_time_shift, np.concatenate([Omega , Omega ]))
    #         grad0 = (((Ydata - np.dot(output, W))) / eps * np.dot(temp, W)).flatten()
    #         temp = np.multiply(Doutput_time_shift, np.concatenate([Omega * Xdata,Omega * Xdata],1))
    #         grad1 = (((Ydata - np.dot(output, W))) / eps *  np.dot(temp, W)).flatten()
    #
    #         for sub in range(self.nrSubj):
    #             temp0 = np.sum([grad0[k] for k in range(int(np.sum(self.N_obs_per_sub[i][:sub])),np.sum(self.N_obs_per_sub[i][:sub+1]))]) - 2 * ((params_time_shift[0] - 0) / 1e-0)[sub]
    #             temp1 = np.sum([grad1[k] for k in range(int(np.sum(self.N_obs_per_sub[i][:sub])),np.sum(self.N_obs_per_sub[i][:sub+1]))])
    #             Gradient[0][sub] = Gradient[0][sub] + temp0
    #             Gradient[1][sub] = Gradient[1][sub] + 0 #temp1
    #
    #     return loglik, Gradient



    def log_posterior_time_shift_Raz(self, time_shift_one_sub, sub, sigmas, Omegas,
      epss, Ws):
      # Input: X, Y and a biomarker's parameters, current time-shift estimates
      # Output: log-posterior and time-shift gradient
      loglik =  0
      # grad = 0

      timeShiftPriorSpread = 6
      prior_time_shift = (time_shift_one_sub - 0) ** 2 / timeShiftPriorSpread

      # Shifting data according to current time-shift estimate
      for i in range(self.nrBiomk):

        sigma = sigmas[i]
        Omega = Omegas[i]
        eps = epss[i]
        W = Ws[i]

        Xdata = time_shift_one_sub + self.X_array[i][int(np.sum(self.N_obs_per_sub[i][:sub])): \
          np.sum(self.N_obs_per_sub[i][:sub+1])]
        Ydata = self.Y_array[i][int(np.sum(self.N_obs_per_sub[i][:sub])):np.sum(self.N_obs_per_sub[i][:sub + 1])]

        loglikCurr = self.log_posterior_time_shift_onebiomk_given_arrays(Xdata, Ydata,
          sigma, Omega, eps, W, prior_time_shift)

        loglik += loglikCurr
        # grad += gradCurr

      return loglik

    def log_posterior_time_shift_onebiomk_given_arrays(self, Xdata, Ydata, sigma, Omega,
                                                       eps, W, prior_time_shift):
      # Input: X, Y and a biomarker's parameters, current time-shift estimates
      # Output: log-posterior and time-shift gradient

      output = self.basis(Xdata, sigma, Omega)
      Doutput_time_shift = self.Dbasis_time_shift(Xdata, sigma, Omega)

      Ypred = np.dot(output, W)
      # print('inside sigma, Omega, eps, W', sigma, Omega, eps, W)
      # print('Ypred inside', Ypred)
      # print('Ydata', Ydata)

      # print('eps', eps)
      loglik = 0.5 * (np.sum((Ydata - Ypred) ** 2) / eps) + prior_time_shift

      # temp = np.multiply(Doutput_time_shift, np.concatenate([Omega , Omega]))
      # grad0 = ((Ydata - np.dot(output, W)) / eps * np.dot(temp, W)).flatten()
      # Gradient = np.sum(grad0) - 2 * ((time_shift_one_sub - 0) / timeShiftPriorSpread)

      # print('prior_time_shift', prior_time_shift)
      # print('loglik', loglik)
      # print(adsa)

      return loglik

    def grad_time_shift(self, params_time_shift):
        output_loglik = []
        objective_cost_function = lambda params_time_shift: \
                    self.log_posterior_time_shift(self.parameters, params_time_shift)

        loglik, MC_grad = objective_cost_function(params_time_shift)

        return loglik, MC_grad

    def addInitTimeShifts(self):

        convTimeOnlyToTimePlusAcc = lambda params_time_shift_only_shift: \
          np.concatenate((params_time_shift_only_shift.reshape(1,-1),
          np.ones((1, params_time_shift_only_shift.shape[0]))),axis=0)

        initTimeShifts = []
        for s in range(len(self.Y[0])):
          initTimeShifts += [[self.Y[b][s][0] for b in range(len(self.Y)) if self.Y[b][s].shape[0] > 0][0] * 50] # 50 months


        initTimeShifts = np.array(initTimeShifts)
        initTimeShifts = initTimeShifts - np.mean(initTimeShifts)
        initTimeShifts = convTimeOnlyToTimePlusAcc(np.array(initTimeShifts))
        # print(ads)

        # import pdb
        # pdb.set_trace()

        for l in range(1):
            self.params_time_shift[l] = self.params_time_shift[l] + initTimeShifts[l]

        for i in range(self.nrBiomk):
            Xdata = np.array([[100]])
            for sub in range(self.nrSubj):
                temp = self.X_array[i][int(np.sum(self.N_obs_per_sub[i][:sub])):np.sum(self.N_obs_per_sub[i][:sub+1])]
                shifted_temp = (temp + initTimeShifts[0][sub])
                Xdata = np.hstack([Xdata,shifted_temp.T])

            self.X_array[i] = Xdata[0][1:].reshape([len(Xdata[0][1:]),1])


        minX = np.float128(np.min([el for sublist in self.X_array for item in sublist for el in item]))
        maxX = np.float128(np.max([el for sublist in self.X_array for item in sublist for el in item]))
        self.updateMinMax(minX, maxX)
        self.DX = np.linspace(self.minX, self.maxX, self.N_Dpoints).reshape([self.N_Dpoints, 1])

    def computeTrajParamsForTimeShifts(self):
      ######## calculate subject-nonspecific terms
      sigmas = []
      Ws = []
      Omegas = []
      epss = []
      for i in range(self.nrBiomk):
        s_omega, m_omega, s_w, m_w, sigma, l, eps = self.unpack_parameters(self.parameters[i])
        s_omega = np.exp(s_omega)
        s_w = np.exp(s_w)
        l = np.exp(l)
        sigma = np.exp(sigma)
        eps = np.exp(eps)

        perturbation_zero_W = np.zeros(int(2 * self.N_rnd_features)).reshape([2 * self.N_rnd_features, 1])
        W = np.multiply(perturbation_zero_W, np.sqrt(np.exp(s_w))) + m_w
        Omega = 1 / np.sqrt(l) * self.perturbation_Omega

        sigmas += [sigma]
        Ws += [W]
        Omegas += [Omega]
        epss += [eps]

      #### end of subject non-specific part

      return sigmas, Ws, Omegas, epss

    def Optimize_time_shift_Raz_indiv(self):
      # Adadelta for optimization of time shift parameters
      init_params = self.params_time_shift.copy()
      init_params[0] = np.zeros(len(init_params[0]))

      init_params_time_only = init_params[0]

      sigmas, Ws, Omegas, epss = self.computeTrajParamsForTimeShifts()

      optimal_params_time_only = np.zeros(init_params_time_only.shape)

      idxOfDRCSubj = 15

      nrSubj = self.nrSubj
      for s in range(nrSubj):
        objectiveFun = lambda time_shift_one_sub: self.log_posterior_time_shift_Raz(
          time_shift_one_sub, s, sigmas, Omegas, epss, Ws)
        # objectiveGrad = lambda time_shift_one_sub: -self.log_posterior_time_shift_Raz(
        #   time_shift_one_sub, s, sigmas, Omegas, epss, Ws)[1]

        options = {'disp': True, 'gtol':1e-8}
        # resStruct = scipy.optimize.minimize(objectiveFun, init_params_time_only[s], method='BFGS', jac=objectiveGrad, options=options)
        resStruct = scipy.optimize.minimize(objectiveFun, init_params_time_only[s], method='Nelder-Mead', options={'disp': True})

        optimal_params_time_only[s] = resStruct.x

      convTimeOnlyToTimePlusAcc = lambda params_time_shift_only_shift: \
        np.concatenate((params_time_shift_only_shift.reshape(1,-1),
        np.ones((1, params_time_shift_only_shift.shape[0]))),axis=0)
      optimal_params = convTimeOnlyToTimePlusAcc(optimal_params_time_only)

      self.updateTimeShifts(optimal_params)

    def updateTimeShifts(self, optimal_params):
      self.super().updateTimeShifts(optimal_params)
      self.DX = np.linspace(minX, maxX, self.N_Dpoints).reshape([self.N_Dpoints, 1])

    def Optimize(self, N_global_iterations, iterGP, Plot = True):
      # Global optimizer (GP parameters + time shift)
      # fig = self.plotter.plotTraj(self)
      # fig.savefig('%s/allTraj%d0_%s.png' % (self.outFolder, 0, self.expName))
      fig2 = self.plotter.plotCompWithTrueParams(self)
      fig2.savefig('%s/compTrueParams%d0_%s.png' % (self.outFolder, 0, self.expName))

      for i in range(N_global_iterations):
        print("iteration ", i, "of ", N_global_iterations)
        print("Optimizing GP parameters")
        if i>float(N_global_iterations)-2:
          self.Optimize_GP_parameters(Niterat = iterGP)
        else:
          # self.N_Dpoints = 10
          self.Optimize_GP_parameters(Niterat=iterGP, optimize_penalty = False)
          print("Current penalty parameters: ")
          print(self.penalty)

        if Plot:
          fig = self.plotter.plotTraj(self)
          fig.savefig('%s/allTraj%d0_%s.png' % (self.outFolder, i + 1, self.expName))
          fig2 = self.plotter.plotCompWithTrueParams(self)
          fig2.savefig('%s/compTrueParams%d0_%s.png' % (self.outFolder, i + 1, self.expName))

        # import pdb
        # pdb.set_trace()

        if i<(N_global_iterations -1):
          print("Optimizing time shift")
          self.Optimize_time_shift_Raz_indiv()

        if Plot:
          fig = self.plotter.plotTraj(self)
          fig.savefig('%s/allTraj%d1_%s.png' % (self.outFolder, i + 1, self.expName))
          fig2 = self.plotter.plotCompWithTrueParams(self)
          fig2.savefig('%s/compTrueParams%d1_%s.png' % (self.outFolder, i + 1, self.expName))


    def Return_time_shift(self):
        individual_time = []
        for sub in range(self.nrSubj):
            individual_time.append(np.array([self.X_array[0][k][0] for k in
                               range(int(np.sum(self.N_obs_per_sub[0][:sub])),
                                     np.sum(self.N_obs_per_sub[0][:sub + 1]))])[0])

        scaleX = self.max_X[0] * self.mean_std_X[0][1]
        return np.array(individual_time) *  scaleX + self.mean_std_X[0][0]


    def StageSubjects(self,X_test, Y_test, Xrange):
      """predicts the posterior distribution of the subject time shifts. Doesn't predict biomarker values"""

      # subject prediction
      pred_sub = []
      expectation_sub = []

      # distribution of trajectory samples
      sampling_dist = []

      for biomarker in range(self.nrBiomk):
          sampling_dist.append([])
          for i in range(500):
              s_omega, m_omega, s, m, sigma, l, eps = self.unpack_parameters(self.parameters[biomarker])
              perturbation_zero_W = np.random.randn(int(2 * self.N_rnd_features)).reshape(
                  [2 * self.N_rnd_features, 1])
              perturbation_zero_Omega = np.random.randn(int(self.N_rnd_features))
              Omega = 1 / np.sqrt(np.exp(l)) * self.perturbation_Omega
              W = np.multiply(perturbation_zero_W, np.sqrt(np.exp(s))) + m
              output = self.basis(Xrange, np.exp(sigma), Omega)
              sampling_dist[biomarker].append(np.dot(output, W))

      for sub in range(len(X_test[0])):
          print("predicting sub: ", sub, "out of ", len(X_test[0]))
          pred_sub.append([])
          expectation_sub.append([])
          for pos_index,position in enumerate(Xrange):
              pred_sub[sub].append(0)
              for biomarker in range(self.nrBiomk):
                  Y_test_biom = np.array(Y_test[biomarker][sub]).reshape([len(Y_test[biomarker][sub]),1])
                  X_test_biom = np.array(X_test[biomarker][sub]).reshape([len(X_test[biomarker][sub]),1])

                  X_test_biom = (X_test_biom - self.mean_std_X[biomarker][0]) / self.mean_std_X[biomarker][1]
                  X_test_biom = X_test_biom / self.max_X[biomarker]

                  Y_test_biom = (Y_test_biom - self.mean_std_Y[biomarker][0]) / self.mean_std_Y[biomarker][1]
                  Y_test_biom = Y_test_biom / self.max_Y[biomarker]

                  if len(X_test_biom > 0):
                      X_to_test = position + X_test_biom
                      for i in range(500):
                          current_sample = sampling_dist[biomarker][i][pos_index:(pos_index + len(Y_test_biom))]
                          if (len(Y_test_biom) == len(current_sample)):
                              pred_sub[sub][pos_index] = pred_sub[sub][pos_index] \
                                                  + np.sum((Y_test_biom - current_sample) ** 2)
                          # - 0.5 * (np.log(2 * np.pi * np.exp(eps)) \

      final_pred = []
      for sub in range(len(pred_sub)):
          invalid_indices = np.where(np.array(pred_sub[sub])==0)[0]
          # pred_sub[sub][pred_sub[sub] == 0] = 1e10
          # print('valid_indices', valid_indices, np.array(pred_sub[sub]).shape)
          # invalid_indices = np.logical_not(np.in1d(np.array(range(Xrange.shape[0])), valid_indices))
          # print(asds)
          # predictions = np.array(pred_sub[sub])[valid_indices]
          predictions = np.array(pred_sub[sub])
          final_pred.append([])
          final_pred[sub] = np.exp(-predictions/500)/ np.sum(np.exp(-predictions/500))
          final_pred[sub][invalid_indices] = 0
          final_pred[sub] /= np.sum(final_pred[sub])
          scaling = self.mean_std_X[biomarker][1]*self.max_X[biomarker]
          #expectation_sub[sub] = np.sum(final_pred[sub] * Xrange.flatten()[valid_indices]) * scaling + self.mean_std_X[biomarker][0]
          # expectation_sub[sub] = np.sum(final_pred[sub] * (Xrange.flatten()[valid_indices] * scaling + self.mean_std_X[biomarker][0]))
          expectation_sub[sub] = np.sum(final_pred[sub] * (Xrange.flatten() * scaling + self.mean_std_X[biomarker][0]))
      return final_pred, expectation_sub

    def predictBiomkWithParams(self, newX, params):

      deltaX = 5 * (self.maxScX - self.minScX)
      if not (self.minScX - deltaX <= np.min(newX) <= self.maxScX + deltaX):
        print('newX', newX)
        print('self.minScX', self.minScX)
        print('self.maxScX', self.maxScX)
        raise ValueError('newX not in the correct renge. check the scaling')

      xsScaled = self.applyScalingXForward(newX.reshape(-1, 1), biomk=0) # arbitrary space ->[0,1]

      predictedBiomksXB = np.zeros((xsScaled.shape[0], self.nrBiomk))
      for bio_pos, biomarker in enumerate(range(self.nrBiomk)):
        s_omega, m_omega, s, m, sigma, l, eps = self.unpack_parameters(params[biomarker])

        sigma = np.exp(sigma)
        eps = np.exp(eps)
        # scaleX = self.max_X[biomarker] * self.mean_std_X[biomarker][1]
        # scaleY = self.max_Y[biomarker] * self.mean_std_Y[biomarker][1]
        perturbation_zero_W = np.zeros(int(2 * self.N_rnd_features)).reshape([2 * self.N_rnd_features, 1])
        Omega = 1 / np.sqrt(np.exp(l)) * self.perturbation_Omega
        sys.stdout.flush()
        W = np.multiply(perturbation_zero_W, np.sqrt(np.exp(s))) + m
        output = self.basis(xsScaled, sigma, Omega)
        sys.stdout.flush()
        predictedBiomksXB[:,biomarker] = np.dot(output, W).reshape(-1)

        # if bio_pos == 0:
        #   print('outside sigma, Omega, eps, W', sigma, Omega, eps, W)
        #   print('outside predictedBiomksXB[:,biomarker]', predictedBiomksXB[:,biomarker])


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

      xsScaled = self.applyScalingXForward(newX.reshape(-1, 1), biomk=0)# arbitrary space ->[0,1]

      trajSamplesXS = np.zeros((xsScaled.shape[0], nrSamples))
      s_omega, m_omega, s, m, sigma, l, eps = self.unpack_parameters(self.parameters[biomarker])

      for i in range(nrSamples):
        perturbation_zero_W = np.random.randn(int(2 * self.N_rnd_features)).reshape([2 * self.N_rnd_features, 1])
        perturbation_zero_Omega = np.random.randn(int(self.N_rnd_features))
        Omega = 1 / np.sqrt(np.exp(l)) * self.perturbation_Omega
        W = np.multiply(perturbation_zero_W, np.sqrt(np.exp(s))) + m
        output = self.basis(xsScaled, np.exp(sigma), Omega)
        trajSamplesXS[:,i] = np.dot(output, W).reshape(-1)


      return self.applyScalingY(trajSamplesXS, biomarker)

    def Save(self, path):
        np.save(path + "/names_biomarkers", self.names_biomarkers)
        np.save(path + "/N_rnd_features", self.N_rnd_features)
        np.save(path + "/N_biom", self.nrBiomk)
        np.save(path + "/X_array", self.X_array)
        np.save(path + "/Y_array", self.Y_array)
        np.save(path + "/DX", self.DX)
        np.save(path + "/group", self.group)
        np.save(path + "/init_params_full", self.init_params_full)
        np.save(path + "/init_params_var", self.init_params_var)
        np.save(path + "/max_X", self.max_X)
        np.save(path + "/max_Y", self.max_Y)
        np.save(path + "/mean_std_X", self.mean_std_X)
        np.save(path + "/mean_std_Y", self.mean_std_Y)
        np.save(path + "/N_biom2", self.nrBiomk)
        np.save(path + "/N_Dpoints", self.N_Dpoints)
        np.save(path + "/N_rnd_features2", self.N_rnd_features)
        np.save(path + "/N_samples", self.nrSubj)
        np.save(path + "/parameters", self.parameters)
        np.save(path + "/params_time_shift", self.params_time_shift)
        np.save(path + "/penalty", self.penalty)
        np.save(path + "/perturbation_Omega", self.perturbation_Omega)
        np.save(path + "/Y_array2", self.Y_array)
        np.save(path + "/maxX", self.maxX)
        np.save(path + "/minX", self.minX)
    
    def Load(self, path):
        self.names_biomarkers = np.load(path + "/names_biomarkers.npy")
        self.N_rnd_features = np.load(path + "/N_rnd_features.npy")
        self.nrBiomk = np.load(path + "/N_biom.npy")
        self.X_array = np.load(path + "/X_array.npy")
        self.Y_array = np.load(path + "/Y_array.npy")
        self.DX = np.load(path + "/DX.npy")
        self.group = np.load(path + "/group.npy")
        self.init_params_full = np.load(path + "/init_params_full.npy")
        self.init_params_var = np.load(path + "/init_params_var.npy")
        self.max_X = np.load(path + "/max_X.npy")
        self.max_Y = np.load(path + "/max_Y.npy")
        self.mean_std_X = np.load(path + "/mean_std_X.npy")
        self.mean_std_Y = np.load(path + "/mean_std_Y.npy")
        self.nrBiomk = np.load(path + "/N_biom2.npy")
        self.N_Dpoints = np.load(path + "/N_Dpoints.npy")
        self.N_rnd_features = np.load(path + "/N_rnd_features2.npy")
        self.nrSubj = np.load(path + "/N_samples.npy")
        self.parameters = np.load(path + "/parameters.npy")
        self.params_time_shift = np.load(path + "/params_time_shift.npy")
        self.penalty = np.load(path + "/penalty.npy")
        self.perturbation_Omega = np.load(path + "/perturbation_Omega.npy")
        self.Y_array = np.load(path + "/Y_array2.npy")
        self.maxX = np.load(path + "/maxX.npy")
        self.minX = np.load(path + "/minX.npy")

    def printParams(self):
      print('names_biomarkers', self.names_biomarkers)
      print('N_rnd_features', self.N_rnd_features)
      print('N_biom', self.nrBiomk)
      print('X_array', self.X_array) # series of flat arrays
      print('Y_array', self.Y_array) # series of flat arrays
      print('DX', self.DX) # derivative points?
      print('group', self.group) # empty
      print('init_params_full', self.init_params_full)
      print('init_params_var', self.init_params_var)
      print('max_X', self.max_X) # maximum values for each biomarker
      print('max_Y', self.max_Y)
      print('mean_std_X', self.mean_std_X) # mean and standard deviation
      print('mean_std_Y', self.mean_std_Y)
      print('N_biom', self.nrBiomk) #
      print('N_Dpoints', self.N_Dpoints) # number of derivative points
      print('N_rnd_features', self.N_rnd_features)
      print('N_samples', self.nrSubj)
      print('parameters', self.parameters)
      print('params_time_shift', self.params_time_shift)
      print('penalty', self.penalty) # penalty flags for each biomarker
      print('perturbation_Omega', self.perturbation_Omega) # something for every derivative points
      print('maxX', self.maxX)
      print('minX', self.minX)


