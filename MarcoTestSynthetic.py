#!/usr/bin/env python

import numpy as np
import sys
sys.path.append('../')
import MarcoModel
import MarcoDataGenerator
import matplotlib.pyplot as plt
import Plotter
import os
import pickle
import colorsys

import numpy as np

np.random.seed(1)

# Creating synthetic multivariate progressions data with monotonic behaviour
L = 1
k = 0.3
interval = [-15,15]

# Number of biomarkers
nrBiomk = 9
# Number of individuals
nrSubj = 300
# Gaussian observational noise
noise = 0.05

flag = 0
while (flag!=1):
    CurveParam = []
    for i in range(nrBiomk):
        CurveParam.append([L,0.8*np.random.rand(),noise])
        if CurveParam[i][1] > 0.0:
            flag = 1

dg = MarcoDataGenerator.DataGenerator(nrBiomk, interval, CurveParam, nrSubj)

# Number of random features for kernel approximation
N=int(10)

X = []
Y = []

for biom in range(len(dg.ZeroXData)):
    X.append([])
    Y.append([])
    for sub in range(len(dg.ZeroXData[biom])):
        # X[biom].append([0])
        # Y[biom].append([dg.YData[biom][sub][0]])
        X[biom].append(dg.XData[biom][sub] - dg.XData[biom][sub][0])
        Y[biom].append(dg.YData[biom][sub])



#gp  = GP_progression_modelSYN.GP_progression_model(dg.ZeroXData,dg.YData,N)
labels = ['b%d' % b for b in range(nrBiomk)]
plotTrajParams = dict(nrRows=3,nrCols=4, SubfigTrajWinSize=(1600,900), isSynth=True, labels=labels)
plotTrajParams['colorsTraj'] = [colorsys.hsv_to_rgb(hue, 1, 1) for hue in np.linspace(0, 1, num=nrBiomk, endpoint=False)]
plotTrajParams['allTrajOverlap'] = False
gpPlotter = Plotter.PlotterGP(plotTrajParams)
outFolder = 'resfiles/marcoSynth'
os.system('mkdir -p %s' % outFolder)

subShiftsTrueMarcoFormat = dg.OutputTimeShift()
trueLineSpacedDPSs = np.array(range(int(np.min(subShiftsTrueMarcoFormat)), int(np.max(subShiftsTrueMarcoFormat))))
trueTrajPredXB = np.zeros((trueLineSpacedDPSs.shape[0], nrBiomk))
for b in range(nrBiomk):
  trueTrajPredXB[:, b] = dg.predPop(trueLineSpacedDPSs, b)

plotTrajParams['trueParams'] = dict(trueLineSpacedDPSs=trueLineSpacedDPSs, trueTrajPredXB=trueTrajPredXB,
  subShiftsTrueMarcoFormat=subShiftsTrueMarcoFormat)


for b in range(nrBiomk):
  Yiflat = [x2 for x in Y[b] for x2 in x]
  # zippedList = list(zip(dg.XData[b], [dg.model[b][x[0]] for x in dg.XData[b]], Y[b]))
  # print('data', np.mean(Yiflat),'+/-', np.std(Yiflat),
  #   zippedList)
  # print('noise std', np.mean([np.abs(x[2] - x[1]) for x in zippedList]))
  print('trueTraj', np.mean(trueTrajPredXB[:,b]),'+/-', np.std(trueTrajPredXB[:,b]), trueTrajPredXB[:,b])
  print('trueLineSpacedDPSs', trueLineSpacedDPSs)
  print('trueTrajPredXB[:, b]', trueTrajPredXB[:, b])
  print('---------')

print('dg.XData', dg.XData)
# print('dg.ZeroXData', dg.ZeroXData)
print('dg.shiftData', dg.shiftData)
print('dg.model', dg.model)
print('dg.YDataNoNoise', dg.YDataNoNoise)
print('dg.YData', dg.YData)
print('dg.OutputTimeShift()', dg.OutputTimeShift())
print('X',X)
print('Y', Y)
# print(asd)

gp  = MarcoModel.GP_progression_model(X,Y,N, outFolder, gpPlotter)
N_global_iterations = 50
iterParams = 50
iterShifts = 30

runPart = ['R', 'R']
gpFile = '%s/fittedGPModel.npz' % outFolder
if runPart[0] == 'R':
  gp.Optimize(N_global_iterations, [iterParams, iterShifts], Plot = True)

  pickle.dump(gp, open(gpFile, 'wb'), protocol = pickle.HIGHEST_PROTOCOL)
else:
  gp = pickle.load(open(gpFile, 'rb'))
  gp.plotter.plotTraj(gp, replaceFig=False)


### prediction
gpPredFile = '%s/preds.npz' % outFolder
Xrange = np.linspace(gp.minX, gp.maxX, 30).reshape([30, 1])
if runPart[1] == 'R':
  pred_prob, pred_exp = gp.StageSubjects(X,Y,Xrange)
  datastruct = dict(pred_prob=pred_prob, pred_exp=pred_exp)
  pickle.dump(datastruct, open(gpPredFile, 'wb'), protocol = pickle.HIGHEST_PROTOCOL)
else:
  dataStruct = pickle.load(open(gpPredFile, 'rb'))
  pred_prob = dataStruct['pred_prob']
  pred_exp = dataStruct['pred_exp']


gpPlotter.Plot_predictions(gp, pred_prob, Xrange, [str(i) for i in range(len(X[0]))])

plt.scatter(pred_exp,dg.OutputTimeShift())
plt.xlabel('Predicted time shift')
plt.ylabel('Ground truth')
plt.title('estimated time-shift vs ground truth')
plt.show()

# Plotting estimated time-shift vs ground truth
plt.scatter(gp.Return_time_shift(), dg.OutputTimeShift())
plt.xlabel('GP time shift')
plt.ylabel('Ground truth')
plt.title('estimated time-shift vs ground truth')
plt.show()


#
#l = 0
#
# Xdata = gp.X_array[0]
# Ydata = gp.Y_array[0]
#
# def compute_params(gp):
#     current_rand_effects = np.zeros(len(Xdata))
#     position = 0
#     for idx_sub, sub in enumerate(range(gp.N_samples)):
#         obs_sub = Xdata[position:  position + gp.N_obs_per_sub[l][idx_sub]]
#         if gp.rand_parameter_type[l][idx_sub] == 2:
#             obs_sub = obs_sub - np.mean(obs_sub)
#             current_rand_effects[position: position +  gp.N_obs_per_sub[l][idx_sub]] = \
#                 gp.rand_parameters[l][idx_sub][0] + obs_sub.flatten() * gp.rand_parameters[l][idx_sub][1]
#         elif gp.rand_parameter_type[l][idx_sub] == 1:
#             current_rand_effects[position: position +  gp.N_obs_per_sub[l][idx_sub]] = \
#                 gp.rand_parameters[l][idx_sub]
#         position = position + gp.N_obs_per_sub[l][idx_sub]
#     return current_rand_effects
#
# res_final = []
# grad_final = []
#
# for i in np.arange(-5,5,0.1):
#     if np.isscalar(gp.rand_parameters[0][10]):
#         gp.rand_parameters[0][10] = i
#     else:
#         gp.rand_parameters[0][10][1] = i
#
#     current_rand_effects = compute_params(gp)
#
#     res,grad = gp.log_posterior_grad_random_effects(gp.X_array[0], gp.Y_array[0], gp.N_rnd_features, gp.parameters[0], gp.penalty[0], current_rand_effects, 0)
#
#     res_final.append(res)
#     if np.isscalar(gp.rand_parameters[0][10]):
#         grad_final.append(grad[10])
#     else:
#         grad_final.append(grad[10][1])
#
#
# print np.diff(res_final), grad_final[1:]
# plt.scatter(np.diff(res_final),grad_final[1:])
# plt.show()
#
# print "a"

######################

#gp  = GP_progression_model.GP_progression_model(X,Y,N)

#gp.Plot()

# print 'Optimizing Model'
# gp.Optimize(Plot=False)

# pred_prob, pred_exp = gp.Predict([dg.ZeroXData,dg.YData])
#
# gp.Plot_predictions(pred_prob,[str(i) for i in range(len(X[0]))])
#
# plt.scatter(pred_exp,dg.OutputTimeShift())
# plt.show()
#
# # Plotting fitted progression models
# gp.Plot()
#
# # Plotting estimated time-shift vs ground truth
# plt.scatter(gp.Return_time_shift(), dg.OutputTimeShift())
# plt.xlabel('GP time shift')
# plt.ylabel('Ground truth')
# plt.title('estimated time-shift vs ground truth')
# plt.show()










