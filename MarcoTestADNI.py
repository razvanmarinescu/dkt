#!/usr/bin/env python

import numpy as np
import sys
sys.path.append('../')
import MarcoModel
import matplotlib.pyplot as plt
# import MarcoDataGenerator
import Plotter
import os
import pickle
import colorsys
import auxFunc

# remember to change test_synthetic/test_ADNI/test_TADPOLE in GP_progression_model

### load the biomarkers
X,Y,RID,list_biomarker, visitIndices = auxFunc.convert_csv("./table_APOEpos.csv")

N=int(10)   # Number of random features for kernel approximation
nrBiomk = len(list_biomarker)

### optimise the model
plotTrajParams = dict(nrRows=2,nrCols=3, SubfigTrajWinSize=(1600,900), isSynth=False, labels=list_biomarker)
plotTrajParams['colorsTraj'] = [colorsys.hsv_to_rgb(hue, 1, 1) for hue in np.linspace(0, 1, num=nrBiomk, endpoint=False)]
plotTrajParams['allTrajOverlap'] = False
gpPlotter = Plotter.PlotterGP(plotTrajParams)
outFolder = 'resfiles/marcoADNI'
os.system('mkdir -p %s' % outFolder)
gp  = MarcoModel.GP_progression_model(X,Y,N, outFolder, gpPlotter, list_biomarker)

N_global_iterations = 100
iterParams = 50
iterShifts = 30
runPart = ['R', 'R']
gpFile = '%s/fittedGPModel.npz' % outFolder
if runPart[0] == 'R':
  gp.Optimize(N_global_iterations, iterParams, Plot = True)

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

print('pred_exp', pred_exp)

# gpPlotter.Plot_predictions(gp, pred_prob,Xrange, [str(i) for i in range(len(X[0]))])
gpPlotter.Plot_predictions(gp, pred_prob,Xrange, [])

# print(gp)

# os.system('mkdir -p %s/vars/' % outFolder)
# gp.Save('%s/vars/' % outFolder)
# gp.printParams()
