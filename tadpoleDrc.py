import sys
import numpy
import numpy as np
import colorsys
from socket import gethostname
import time
import argparse
import os
import colorsys
import copy
import pandas as pd
import pickle
import random
import auxFunc
import scipy
import scipy.io as sio
import scipy.stats

# sys.path.append(os.path.abspath("../diffEqModel/"))


parser = argparse.ArgumentParser(description='Launches voxel-wise/point-wise DPM on ADNI'
                                             'using cortical thickness maps derived from MRI')

parser.add_argument('--agg', dest='agg', type=int, default=0,
  help='agg=1 => plot figures without using Xwindows, for use on cluster where the plots cannot be displayed '
       ' agg=0 => plot with Xwindows (for use on personal machine)')

parser.add_argument('--runIndex', dest='runIndex', type=int,
  default=1, help='index of run instance/process .. for cross-validation')

parser.add_argument('--nrProc', dest='nrProc', type=int,
  default=1, help='# of processes')

parser.add_argument('--modelToRun', dest='modelToRun', type=int,
  help='index of model to run')

parser.add_argument('--cluster', action="store_true",
  help='need to include this flag if runnin on cluster')

parser.add_argument('--nrRows', dest='nrRows', type=int,
  help='nr of subfigure rows to plot at every iteration')

parser.add_argument('--nrCols', dest='nrCols', type=int,
  help='nr of subfigure columns to plot at every iteration')

parser.add_argument('--penalty', dest='penalty', type=float,
  help='penalty value for non-monotonic trajectories. between 0 (no effect) and 10 (strong effect). ')

parser.add_argument('--regData', action="store_true", default=False,
  help=' add this flag to regenerate the data')

parser.add_argument('--runPartStd', dest='runPartStd', default='RR',
  help=' choose whether to (R) run or (L) load from the checkpoints: '
  'either LL, RR, LR or RL. ')

parser.add_argument('--tinyData', action="store_true", default=False,
  help=' only run on a tiny subset of the data: around 200/1980 subjects')


args = parser.parse_args()

if args.agg:
  # print(matplotlib.__version__)
  import matplotlib
  matplotlib.use('Agg')
  # print(asds)

import genSynthData
import GPModel
import ParHierModel
import Plotter
from auxFunc import *
import evaluationFramework
from matplotlib import pyplot as pl

hostName = gethostname()
if hostName == 'razvan-Inspiron-5547':
  freesurfPath = '/usr/local/freesurfer-5.3.0'
  homeDir = '/home/razvan'
  blenderPath = 'blender'
elif hostName == 'razvan-Precision-T1700':
  freesurfPath = '/usr/local/freesurfer-5.3.0'
  homeDir = '/home/razvan'
  blenderPath = 'blender'
elif args.cluster:
  freesurfPath = '/share/apps/freesurfer-5.3.0'
  homeDir = '/home/rmarines'
  blenderPath = '/share/apps/blender-2.75/blender'
elif hostName == 'planell-VirtualBox':
  homeDir = '/home/planell'
  freesurfPath = ""
  blenderPath = ""
else:
  raise ValueError('Wrong hostname. If running on new machine, add '
                   'application paths in python code above')


plotTrajParams = {}
plotTrajParams['SubfigTrajWinSize'] = (1600,900)
plotTrajParams['nrRows'] = args.nrRows
plotTrajParams['nrCols'] = args.nrCols
plotTrajParams['diagColors'] = {CTL:'g', MCI:'y', AD:'r',
  CTL2:'g', PCA:'y', AD2:'r'}
plotTrajParams['diagScatterMarkers'] = {CTL:'o', MCI:'o', AD:'o',
  CTL2:'x', PCA:'x', AD2:'x'}
plotTrajParams['legendCols'] = 4
plotTrajParams['diagLabels'] = {CTL:'CTL ADNI', MCI:'MCI ADNI', AD:'tAD ADNI',
  CTL2:'CTL LOCAL', PCA:'PCA LOCAL', AD2:'tAD LOCAL'}

plotTrajParams['freesurfPath'] = freesurfPath
# plotTrajParams['ylimitsRandPoints'] = (-3,2)
plotTrajParams['blenderPath'] = blenderPath
# plotTrajParams['isSynth'] = False


if args.agg:
  plotTrajParams['agg'] = True
else:
  plotTrajParams['agg'] = False

hostName = gethostname()
if hostName == 'razvan-Inspiron-5547':
  height = 350
else: #if hostName == 'razvan-Precision-T1700':
  height = 450

if hostName == 'razvan-Inspiron-5547':
  homeDir = '/home/razvan'
  freesurfPath = '/usr/local/freesurfer-6.0.0'
elif hostName == 'razvan-Precision-T1700':
  homeDir = '/home/razvan'
  freesurfPath = '/usr/local/freesurfer-6.0.0'
elif args.cluster:
  homeDir = '/home/rmarines'
  freesurfPath = '/home/rmarines/src/freesurfer-6.0.0'
elif hostName == 'planell-VirtualBox':
  homeDir = '/home/planell'
  freesurfPath = ""
  blenderPath = ""
else:
  raise ValueError('wrong hostname or cluster flag')

def loadTadpole(tadpoleFile):
  df = pd.read_csv(tadpoleFile)

  cols = list(df.loc[:, 'CDRSB':'FAQ']) \
     + list(df.loc[:, 'ST101SV_UCSFFSX_11_02_15_UCSFFSX51_08_01_16':'ST9SV_UCSFFSX_11_02_15_UCSFFSX51_08_01_16']) \
     + list(df.loc[:, 'HIPPL01_BAIPETNMRC_09_12_16':'MCSUVRCERE_BAIPETNMRC_09_12_16']) \
     + list(df.loc[:, 'CEREBELLUMGREYMATTER_UCBERKELEYAV45_10_17_16':'WM_HYPOINTENSITIES_SIZE_UCBERKELEYAV45_10_17_16']) \
     + list(df.loc[:, 'CEREBELLUMGREYMATTER_UCBERKELEYAV1451_10_17_16':'WM_HYPOINTENSITIES_SIZE_UCBERKELEYAV1451_10_17_16']) \
     + list(df.loc[:, 'FA_CST_L_DTIROI_04_30_14':'AD_SUMFX_DTIROI_04_30_14'])

  # filter out the FS cols with Standard deviation of volumes, cort thickness, etc ... Only keep average
  colsFilt = []
  for col in cols:
    if col[:2] == 'ST' and (col[5] == 'S' or col[6] == 'S'):
      continue

    colsFilt += [col]


  # convert diagnoses such as 'MCI to Dementia' to 'Dementia', etc ...
  # ctlDxchange = [1, 7, 9] mciDxchange = [2, 4, 8] adDxChange = [3, 5, 6]
  mapping = {1: CTL, 7: CTL, 9: CTL, 2: MCI, 4: MCI, 8: MCI, 3: AD, 5: AD, 6: AD}
  # df.replace({'DXCHANGE': mapping}, inplace=True)
  df['DXCHANGE'] = df['DXCHANGE'].map(mapping)

  cols = list(df.loc[:, ['RID', 'AGE', 'Month_bl', 'DXCHANGE']]) + cols

  # df[cols] = df[cols].apply(pd.to_numeric, errors='coerce', axis=1)
  # pickle.dump(df, open('tadpoleCleanDf.npz', 'wb'), protocol=pickle.HIGHEST_PROTOCOL)
  df = pickle.load(open('tadpoleCleanDf.npz', 'rb'))

  dataDf = df[['RID', 'Month_bl']]
  dataDf.loc[:,'scanID'] = np.nan
  print('dataDf', dataDf.loc[:20,:])

  mapGender = {'Female' : 0, 'Male' : 1}
  df.loc[:, 'PTGENDER'] = df['PTGENDER'].map(mapGender)

  # print(ads)

  dataDf.loc[:, 'gender-0f1m'] = df['PTGENDER']
  dataDf.loc[:, 'age'] = df['AGE'] + (df['Month_bl'] / 12)
  dataDf.loc[:, 'dataset'] = 1
  ssTagMRI = 'UCSFFSX_11_02_15_UCSFFSX51_08_01_16'
  ICV = df['ST10CV_%s' % ssTagMRI]
  dataDf.loc[:,'ICV'] = ICV
  dataDf.loc[:,'diag'] = df['DXCHANGE']
  cogTestsLabels = list(df.loc[:, 'CDRSB':'FAQ'])
  dataDf[cogTestsLabels] = df.loc[:, cogTestsLabels]

  # colsList = dataDf.columns.tolist()

  ######## MRI biomk selection ################

  volBiomkStruct = {
    'Volume Frontal' : ['ST102CV', 'ST104CV', 'ST105CV', 'ST106CV', 'ST110CV',
      'ST111CV', 'ST114CV', 'ST115CV', 'ST15CV', 'ST25CV', 'ST36CV', 'ST39CV',
      'ST43CV', 'ST45CV', 'ST46CV', 'ST47CV', 'ST51CV', 'ST55CV', 'ST56CV',
      'ST74CV', 'ST84CV', 'ST95CV', 'ST98CV'],
    'Volume Parietal' : ['ST108CV', 'ST116CV', 'ST118CV', 'ST31CV', 'ST49CV',
      'ST52CV', 'ST57CV', 'ST59CV', 'ST90CV'],
    'Volume Temporal' : ['ST103CV', 'ST117CV', 'ST119CV', 'ST121CV',
      'ST24CV', 'ST26CV', 'ST32CV', 'ST40CV', 'ST44CV', 'ST58CV', 'ST60CV',
      'ST62CV', 'ST85CV', 'ST91CV', 'ST99CV'],
    'Volume Occipital' : ['ST107CV', 'ST23CV', 'ST35CV', 'ST38CV', 'ST48CV',
      'ST82CV', 'ST83CV', 'ST94CV', 'ST97CV'],
    'Volume Cingulate' : ['ST113CV', 'ST109CV', 'ST14CV', 'ST34CV', 'ST50CV',
      'ST54CV', 'ST73CV', 'ST93CV'],
    'Volume Hippocampus': ['ST29SV', 'ST88SV']
  }

  for k in volBiomkStruct.keys():
    volBiomkStruct[k] = ['%s_%s' % (x, ssTagMRI)
      for x in volBiomkStruct[k]]

  dataDf = addBiomks(volBiomkStruct, df, dataDf, collapseFunc=np.sum)



  ######## DTI biomk selection ################

  # didn't include due to being in multiple regions at the same time:
  # Uncinate fasciculus, fronto-occipital fasciculus, longitudinal fasciculus

  dtiBiomkStructTemplate = {
    'Frontal' : ['CST', 'ACR', 'SCR'],
    'Parietal' : ['PCR', 'PTR'],
    'Temporal' : ['SS'],
    'Occipital' : ['SLF'], # not only occipital, but also frontal & temporal
    'Cingulate' : ['CGC', 'GCC', 'BCC', 'SCC'],
    'Hippocampus': ['CGH', 'FX']
  }

  ssTagDTI = 'DTIROI_04_30_14'
  dtiTypes = ['FA']
  sides = ['L', 'R']
  dtiBiomkStruct = {}
  for k in list(dtiBiomkStructTemplate.keys()):
    roiList = dtiBiomkStructTemplate[k]
    for t in dtiTypes:
      currKey = 'DTI %s %s' % (t, k)
      dtiBiomkStruct[currKey] = []
      for s in sides:
        dtiBiomkStruct[currKey] += ['%s_%s_%s' % (t, x, s)
            for x in roiList]

      # append spreadsheet tag
      dtiBiomkStruct[currKey] = ['%s_%s' % (x, ssTagDTI) for x in dtiBiomkStruct[currKey]]


  dataDf = addBiomks(dtiBiomkStruct, df, dataDf, collapseFunc=np.nanmean)

  # print(dataDf)
  #
  # print(dataDf)
  # print(adsa)

  ######## FDG PET biomk selection ################

  fdgBiomkStruct = {
    'FDG Frontal': ['FRTSUPL01', 'FRTMIDL01', 'FRTINFL01'],
    'FDG Parietal': ['ANGULL01', 'PARIINFL01', 'SUPMRGL01',
      'PRECUNL01', 'PARISUPL01'],
    'FDG Temporal': ['PARAHIPL01', 'FUSFRML01', 'TMPMIDL01',
      'TMPSUPL01', 'TMPINFL01'],
    'FDG Occipital': ['OCCMIDL01', 'LINGUALL01'],
    'FDG Cingulate': ['CINGPSTL01', 'CINGANTL01', 'CINGMIDL01'],
    'FDG Hippocampus': ['HIPPL01']
  }

  ssTagFDG = 'BAIPETNMRC_09_12_16'
  fdgBiomkLabels = list(fdgBiomkStruct.keys())
  for k in range(len(fdgBiomkLabels)):
    fdgBiomkStruct[fdgBiomkLabels[k]] += [(x[:-3] + 'R01')
      for x in fdgBiomkStruct[fdgBiomkLabels[k]] if x not in ['PRECUNL01']]
    print(fdgBiomkStruct[fdgBiomkLabels[k]])
    fdgBiomkStruct[fdgBiomkLabels[k]] = ['%s_%s' % (x, ssTagFDG) for x in
      fdgBiomkStruct[fdgBiomkLabels[k]]]


  fdgCols = [x2 for x in fdgBiomkStruct.values() for x2 in x]

  for c in fdgCols:
    df.loc[df[c] == -4, c] = np.nan


  dataDf = addBiomks(fdgBiomkStruct, df, dataDf, collapseFunc=np.mean)

  print('FDG Hippocampus sum', np.sum(np.isnan(dataDf.loc[:, 'FDG Hippocampus'])))

  fdgColsInDataDf = fdgBiomkStruct.keys()
  nanMaskSB = np.isnan(dataDf.loc[:, fdgColsInDataDf])
  dataDf.loc[np.sum(nanMaskSB,axis=1) > 0, fdgColsInDataDf] = np.nan

  # print(np.sum(nanMaskSB,axis=1))
  # print(np.sum(np.sum(nanMaskSB,axis=1) == 3))
  print('FDG Hippocampus sum', np.sum(np.isnan(dataDf.loc[:, 'FDG Hippocampus'])))
  # print(adsa)

  # if np.any(np.isnan(dataDf.loc[s,fdgColsInDataDf])):
  #   # if any number is nan then make them all NaNs
  #   dataDf.loc[s, fdgColsInDataDf] = np.nan


  ######## AV45 PET biomk selection ################

  av45BiomkStruct = {
    'AV45 Frontal': ['FRONTAL'],
    'AV45 Parietal': ['PARIETAL'],
    'AV45 Temporal': ['TEMPORAL'],
    'AV45 Occipital': ['CTX_LH_LATERALOCCIPITAL'],
    'AV45 Cingulate': ['CINGULATE'],
    'AV45 Hippocampus': ['LEFT_HIPPOCAMPUS', 'RIGHT_HIPPOCAMPUS']
  }

  ssTagAV45 = 'UCBERKELEYAV45_10_17_16'
  av45BiomkLabels = list(av45BiomkStruct.keys())
  for k in av45BiomkStruct.keys():
    av45BiomkStruct[k] = ['%s_%s' % (x, ssTagAV45) for x in av45BiomkStruct[k]]

  dataDf = addBiomks(av45BiomkStruct, df, dataDf, collapseFunc=np.nanmean)


  ######## AV1451 PET biomk selection ################

  av1451BiomkStructTemplate = {
    'AV1451 Frontal': ['CAUDALMIDDLEFRONTAL', 'FRONTALPOLE', 'LATERALORBITOFRONTAL',
      'MEDIALORBITOFRONTAL', 'PARACENTRAL', 'PARSOPERCULARIS', 'PARSORBITALIS',
      'PARSTRIANGULARIS', 'PRECENTRAL', 'ROSTRALMIDDLEFRONTAL', 'SUPERIORFRONTAL'],
    'AV1451 Parietal': ['INFERIORPARIETAL', 'POSTCENTRAL', 'PRECUNEUS', 'SUPERIORPARIETAL',
      'SUPRAMARGINAL'],
    'AV1451 Temporal': ['ENTORHINAL', 'FUSIFORM', 'INFERIORTEMPORAL', 'MIDDLETEMPORAL',
      'PARAHIPPOCAMPAL', 'SUPERIORTEMPORAL', 'TEMPORALPOLE', 'TRANSVERSETEMPORAL'],
    'AV1451 Occipital': ['CUNEUS', 'LATERALOCCIPITAL', 'LINGUAL', 'PERICALCARINE'],
    'AV1451 Cingulate': ['CAUDALANTERIORCINGULATE', 'ISTHMUSCINGULATE', 'POSTERIORCINGULATE',
      'ROSTRALANTERIORCINGULATE'],
    'AV1451 Hippocampus': ['LEFT_HIPPOCAMPUS', 'LEFT_HIPPOCAMPUS']
  }

  ssTagAV1451 = 'UCBERKELEYAV1451_10_17_16'
  av1451BiomkLabels = list(av1451BiomkStructTemplate.keys())
  av1451BiomkStruct = {}
  av1451BiomkStruct['AV1451 Hippocampus'] = ['LEFT_HIPPOCAMPUS', 'LEFT_HIPPOCAMPUS']
  for k in av1451BiomkStructTemplate.keys():
    if k != 'AV1451 Hippocampus':
      av1451BiomkStruct[k] = []
      for h in ['LH', 'RH']: # add left and right hemispheres
        av1451BiomkStruct[k] += ['CTX_%s_%s' % (h, x) for x in av1451BiomkStructTemplate[k]]

    av1451BiomkStruct[k] = ['%s_%s' % (x, ssTagAV1451) for x in av1451BiomkStruct[k]]

    # print(av1451BiomkStruct[k])

  # print(asda)
  dataDf = addBiomks(av1451BiomkStruct, df, dataDf, collapseFunc=np.nanmean)

  # print(dataDf.columns.tolist())
  # print(ads)

  # remove subjects that don't have any data
  noMRI = np.isnan(dataDf.loc[:, 'Volume Frontal'])
  noFDG = np.isnan(dataDf.loc[:, 'FDG Frontal'])
  noDTI = np.isnan(dataDf.loc[:, 'DTI FA Frontal'])
  noAV45 = np.isnan(dataDf.loc[:, 'AV45 Frontal'])
  noAV1451 = np.isnan(dataDf.loc[:, 'AV1451 Frontal'])
  idxToKeep = np.logical_not(noMRI & noFDG & noDTI & noAV45 & noAV1451)

  # print('idxToKeep', np.sum(idxToKeep), idxToKeep.shape)
  # print(ads)

  dataDf = dataDf.loc[idxToKeep,:]


  return dataDf


def agregateFSDataDRC():


  drcFSSubjDir = '/home/razvan/seagate/DRC_data/fs_subjects'
  exportSubjCmd = 'export SUBJECTS_DIR=%s' % drcFSSubjDir
  exportFreeSurfCmd = 'export FREESURFER_HOME=%s; source %s/SetUpFreeSurfer.sh' \
    % (freesurfPath, freesurfPath)

  from glob import glob
  subjFlds = [x.split('/')[-2] for x in glob("%s/*/" % drcFSSubjDir)]


  # subjFlds =   [x[0] for x in os.walk(drcFSSubjDir)]

  # print('subjFlds', subjFlds)
  # print(asd)

  subjListStr = ' '.join(subjFlds)
  print(subjListStr)
  runAggreg = False

  subcortProg = 'asegstats2table'
  subcortOutFile = 'drcFSXSubcort.csv'
  subcortFsAggCmd = '%s ; %s; %s/bin/%s --subjects %s --meas volume --skip ' \
    '--tablefile %s --delimiter=comma ' % (exportSubjCmd, exportFreeSurfCmd,
    freesurfPath, subcortProg, subjListStr, subcortOutFile)
  print(subcortFsAggCmd)
  if runAggreg:
    os.system(subcortFsAggCmd)
  dfSubcort = pd.read_csv(subcortOutFile)

  cortProg = 'aparcstats2table'
  cortLhOutFile = 'drcFSXCortLh.csv'
  cortLhFsAggCmd = '%s ; %s; %s/bin/%s --subjects %s --meas volume --hemi lh --skip ' \
                    '--tablefile %s --delimiter=comma ' % (exportSubjCmd, exportFreeSurfCmd,
  freesurfPath, cortProg, subjListStr, cortLhOutFile)
  print(cortLhFsAggCmd)
  if runAggreg:
    os.system(cortLhFsAggCmd)
  dfCortLh = pd.read_csv(cortLhOutFile)

  cortRhOutFile = 'drcFSXCortRh.csv'
  cortRhFsAggCmd = '%s ; %s; %s/bin/%s --subjects %s --meas volume --hemi rh --skip ' \
                   '--tablefile %s --delimiter=comma ' % (exportSubjCmd, exportFreeSurfCmd,
  freesurfPath, cortProg, subjListStr, cortRhOutFile)
  print(cortRhFsAggCmd)
  if runAggreg:
    os.system(cortRhFsAggCmd)
  dfCortRh = pd.read_csv(cortRhOutFile)

  assert dfSubcort.shape[0] == dfCortLh.shape[0] == dfCortRh.shape[0]

  dfAll = dfSubcort
  dfAll[dfCortLh.columns] = dfCortLh
  dfAll[dfCortRh.columns] = dfCortRh

  # print(adss)
  dfAll['partCode'] = [int(x.split('-')[0][4:]) for x in dfAll['Measure:volume']]
  dfAll['studyID'] = [int(x.split('-')[2]) for x in dfAll['Measure:volume']]

  # print(np.where(dfAll['studyID'] == 0))
  # print(asd)
  # there's two rows with the same scanID 3560. Idential FS volumes
  # suggest this scan is duplicated. Drop both of them.
  print(np.where(dfAll['studyID'] == 3560))
  idxToDrop = np.where(np.in1d(dfAll['studyID'], [3560, 0]))[0]
  # idxToDrop = np.logical_or(dfAll['studyID'] == 3560, np.in1d(dfAll.partCode == 1222))
  dfAll.drop(dfAll.index[idxToDrop], inplace=True)
  dfAll.reset_index(drop=True, inplace=True)

  print(np.where(dfAll['studyID'] == 3560))
  # print(ads)

  return dfAll

def addMetadataDrc(dfAll):
  ############ load medatada - age, gender, ... ###########

  # load metadata in order to get the timepoint information for each subj id
  metaData = sio.loadmat('pcaData.mat')

  print(metaData.keys())
  partCode = metaData['participantCode']
  diag = metaData['pcaDiag']
  ageAtScan = metaData['ageAtScan']
  studyID = metaData['studyID']
  gender = metaData['gender']
  # subgroupPCA = metaData['subgroupPCA']
  # subgroupPCA = ageAtScan

  print(dfAll['studyID'].shape)
  # print(ads)
  nrSubjCross = dfAll.shape[0]
  unqPart = np.unique(dfAll['partCode'])
  nrUnqPart = unqPart.shape[0]

  ageAtScanDf = np.nan * np.ones(nrSubjCross, float)
  genderDf = np.nan * np.ones(nrSubjCross, float)
  diagDf = np.nan * np.ones(nrSubjCross, float)
  Month_blDf = np.nan * np.ones(nrSubjCross, float)

  # print('studyID.shape', studyID.shape, np.unique(studyID).shape)
  # print()
  # print(asd)
  # print('age 3560 ', ageAtScan[studyID == 3560])
  # print('gender 3560 ', gender[studyID == 3560])
  # print('diag 3560 ', diag[studyID == 3560])
  # print('partCode 3560 ', partCode[studyID == 3560])

  for s in range(nrSubjCross):
    # print(dfAll['studyID'][s])
    # print(dfAll['studyID'][s] == studyID)
    idx = (dfAll['studyID'][s] == studyID).reshape(-1)
    if np.sum(idx) > 1:
      print(dfAll['studyID'][s])

    if np.sum(idx) == 1:
      # print(ageAtScan[idx])
      ageAtScanDf[s] = ageAtScan[idx]
      genderDf[s] = gender[idx]
      diagDf[s] = diag[idx]

  # print(adsa)

  for p in range(nrUnqPart):
    currPartIndDf = dfAll['partCode'] == unqPart[p]
    ageCurrPartDf = ageAtScanDf[currPartIndDf]
    Month_blDf[currPartIndDf] = (ageCurrPartDf - np.min(ageCurrPartDf))*12.0


  dfAll['ageAtScan'] = ageAtScanDf
  dfAll['gender'] = genderDf
  dfAll['diag'] = diagDf
  dfAll['Month_bl'] = Month_blDf

  return dfAll

  ##############

def loadDRC(drcFile, columnsFormat):
  # first use Freesurfer to aggregate all the volume information into one csv file

  df = agregateFSDataDRC()

  df = addMetadataDrc(df)
  # print(adssa)
  columnsFormat = columnsFormat[1:]
  # print(columnsFormat)
  # print(asd)

  dataDf = pd.DataFrame(np.nan * np.ones((df.shape[0], len(columnsFormat))),
    columns=columnsFormat)
  print(dataDf.columns)
  dataDf.reindex(range(df.shape[0]))
  dataDf[['RID', 'Month_bl', 'scanID']] = df[['partCode', 'Month_bl', 'studyID']]

  # dataDf.rename(index=str, columns={"partCode": "RID"}, inplace=True)
  # print(dataDf.loc[:10,:])

  # print(list(df.columns))
  # print(ads)

  # add ageDrc, genderDrc, ICVTadpoleDrc, dataset
  dataDf['age'] = df['ageAtScan']
  dataDf['gender-0f1m'] = df['gender']
  dataDf['dataset'] = 2 # number identifying current dataset
  dataDf['diag'] = df['diag']

  print(dataDf['diag'])
  mapDiagDRC = {1: CTL2, 2: PCA, 3: AD2}
  dataDf['diag'] = dataDf['diag'].map(mapDiagDRC)
  # print(asd)

  ######## MRI biomk selection ################




  '''
  Frontal

      Superior Frontal
      Rostral and Caudal Middle Frontal
      Pars Opercularis, Pars Triangularis, and Pars Orbitalis
      Lateral and Medial Orbitofrontal
      Precentral
      Paracentral
      Frontal Pole

  Parietal

      Superior Parietal
      Inferior Parietal
      Supramarginal
      Postcentral
      Precuneus

  Temporal

      Superior, Middle, and Inferior Temporal
      Banks of the Superior Temporal Sulcus
      Fusiform
      Transverse Temporal
      Entorhinal
      Temporal Pole
      Parahippocampal

  Occipital

      Lateral Occipital
      Lingual
      Cuneus
      Pericalcarine

  Cingulate (if you want to include in a lobe)

      Rostral Anterior (Frontal)
      Caudal Anterior (Frontal)
      Posterior (Parietal)
      Isthmus (Parietal)

  '''

  volBiomkStruct = {
    'ICV' : ['EstimatedTotalIntraCranialVol'],
    'Volume Frontal' : ['lh_caudalmiddlefrontal_volume',
      'lh_lateralorbitofrontal_volume',  'lh_medialorbitofrontal_volume',
      'lh_rostralmiddlefrontal_volume', 'lh_superiorfrontal_volume',
      'lh_frontalpole_volume', 'lh_paracentral_volume',
      'lh_parsopercularis_volume', 'lh_parsorbitalis_volume',
      'lh_parstriangularis_volume', 'lh_precentral_volume',
      'lh_rostralmiddlefrontal_volume'],
    'Volume Parietal' : ['lh_inferiorparietal_volume',
      'lh_postcentral_volume',  'lh_precuneus_volume',
      'lh_superiorparietal_volume'],
    'Volume Temporal' : ['lh_entorhinal_volume', 'lh_fusiform_volume',
      'lh_inferiortemporal_volume', 'lh_middletemporal_volume',
      'lh_parahippocampal_volume', 'lh_superiortemporal_volume',
      'lh_supramarginal_volume', 'lh_temporalpole_volume',
      'lh_transversetemporal_volume'],
    'Volume Occipital' : ['lh_cuneus_volume', 'lh_lateraloccipital_volume',
      'lh_lingual_volume', 'lh_pericalcarine_volume', ],
    'Volume Cingulate' : ['lh_caudalanteriorcingulate_volume',
      'lh_isthmuscingulate_volume', 'lh_posteriorcingulate_volume',
      'lh_rostralanteriorcingulate_volume', ],
    'Volume Hippocampus': ['Left-Hippocampus', 'Right-Hippocampus']
  }

  for k in volBiomkStruct.keys():
    volBiomkStruct[k] += ['rh%s' % x[2:] for x in volBiomkStruct[k]
      if x[:2] == 'lh']

  dataDf = addBiomks(volBiomkStruct, df, dataDf, collapseFunc=np.sum)

  cogTestsDf = None
  # print(dataDf.loc[:10,:])
  # print(ads)

  return dataDf

def addBiomks(biomkStruct, sourceDf, targetDf, collapseFunc):

  biomkLabels = np.sort(list(biomkStruct.keys()))
  nrBiomk = len(biomkStruct.keys())
  for b in range(nrBiomk):
    # print(biomkLabels[b], biomkStruct[biomkLabels[b]])
    # fullBiomkNames = ['%s_%s' % (x, spreadsheetTag)
    #   for x in biomkStruct[biomkLabels[b]]]
    sys.stdout.flush()
    # print(np.where(np.in1d(sourceDf.columns.tolist(), biomkStruct[biomkLabels[b]])))
    # print(sourceDf.columns.tolist()[1200:])

    dataFrameCurrSubset = sourceDf[biomkStruct[biomkLabels[b]]]
    cols = list(dataFrameCurrSubset.columns)
    # print(list(dataFrameCurrSubset.columns))
    # print(dict(zip(cols, [x[:6] for x in cols])))
    # dataFrameCurrSubset.rename(index=str, columns=dict(zip(cols, [x[:6] for x in cols])))
    dataFrameCurrSubset.columns = [x[:6] for x in cols]
    # print('dataFrameCurrSubset', dataFrameCurrSubset.loc[:5,:])
    # print('np.sum', np.sum(dataFrameCurrSubset,axis=1))
    targetDf.loc[:, biomkLabels[b]] = pd.Series(np.sum(dataFrameCurrSubset,axis=1),
      index=targetDf.index)
    # targetDf = targetDf.assign(asd=np.sum(dataFrameCurrSubset,axis=1))
    # print(targetDf.loc[:5, biomkLabels[b]])
    # print(ads)



  return targetDf

def regressCov(data, regressorVector, diag, diagsCTL = (CTL, CTL2), printFigs=False):
  oldData = copy.deepcopy(data)
  M = np.zeros((data.shape[1], 2))
  desiredMean = np.zeros(data.shape[1])
  for i in range(data.shape[1]):
      currCol = data.columns[i]
      notNanIndices = np.logical_not(np.isnan(regressorVector))
      notNanIndices = np.logical_and(notNanIndices,
        np.logical_not(np.isnan(data.loc[:,currCol])))
      regressorVectorNN = regressorVector[notNanIndices]
      diagNN = diag[notNanIndices]
      dataNNcurrCol = data.loc[notNanIndices,currCol]
      indicesCtl = np.in1d(diagNN, diagsCTL)
      regressorCTL = regressorVectorNN[indicesCtl]

      # Solve the GLM: Y = [X 1] * M
      X = np.concatenate((regressorCTL.reshape(-1,1),
      np.ones((regressorCTL.shape[0],1))),axis=1)
      XXX = np.dot(np.linalg.pinv(np.dot(X.T,X)), X.T)

      M[i,:] = np.dot(XXX, dataNNcurrCol.loc[indicesCtl]) # params of linear fit
      assert(~any(np.isnan(M[i,:])));
      Xfull = np.concatenate((regressorVector.reshape(-1,1),
      np.ones((regressorVector.shape[0],1))),axis=1)

      Yhat = np.dot(Xfull, M[i,:]) # estimated Ys
      desiredMean[i] = np.nanmean(dataNNcurrCol.loc[indicesCtl])
      data.loc[:, currCol] = data.loc[:,currCol] - (Yhat - desiredMean[i])

      if printFigs:
        h = pl.figure(1, figsize=(15,10))
        pl.scatter(regressorVector, oldData.loc[:,currCol], c='r',label='before', s=5)
        pl.scatter(regressorVector, data[currCol], c='b',label='after', s=5)
        pl.plot(regressorVectorNN, Yhat[notNanIndices], c='r')
        correctedPred = np.nanmean(dataNNcurrCol.loc[indicesCtl]) * \
                        np.ones(dataNNcurrCol.loc[indicesCtl].shape[0])
        pl.plot(regressorVectorNN[indicesCtl],correctedPred  , c='b')
        pl.title('%s' % data.columns[i])
        pl.legend()

        pl.show()

  params = dict(M=M, labels=data.columns, desiredMean=desiredMean)

  return data, params


def applyRegFromParams(data, regressorVector, diag, params, diagsCTL = (CTL, CTL2),
  printFigs=False):

  oldData = copy.deepcopy(data)
  M = params['M']
  desiredMean = params['desiredMean']
  for i in range(data.shape[1]):
      currCol = data.columns[i]
      # notNanIndices = np.logical_not(np.isnan(regressorVector))
      # notNanIndices = np.logical_and(notNanIndices,
      #   np.logical_not(np.isnan(data.loc[:,currCol])))
      # regressorVectorNN = regressorVector[notNanIndices]
      # diagNN = diag[notNanIndices]
      # dataNNcurrCol = data.loc[notNanIndices,currCol]
      # indicesCtl = np.in1d(diagNN, diagsCTL)
      # regressorCTL = regressorVectorNN[indicesCtl]
      #
      # # Solve the GLM: Y = [X 1] * M
      # X = np.concatenate((regressorCTL.reshape(-1,1),
      # np.ones((regressorCTL.shape[0],1))),axis=1)
      # XXX = np.dot(np.linalg.pinv(np.dot(X.T,X)), X.T)

      # M[i,:] = np.dot(XXX, dataNNcurrCol.loc[indicesCtl]) # params of linear fit
      assert(~any(np.isnan(M[i,:])));

      Xfull = np.concatenate((regressorVector.reshape(-1,1),
      np.ones((regressorVector.shape[0],1))),axis=1)

      Yhat = np.dot(Xfull, M[i,:]) # estimated Ys
      data.loc[:, currCol] = data.loc[:,currCol] - (Yhat - desiredMean[i])

      if printFigs:
        h = pl.figure(1, figsize=(15,10))
        pl.scatter(regressorVector, oldData.loc[:,currCol], c='r',label='before', s=5)
        pl.scatter(regressorVector, data[currCol], c='b',label='after', s=5)
        # pl.plot(regressorVectorNN, Yhat[notNanIndices], c='r')
        # correctedPred = np.nanmean(dataNNcurrCol.loc[indicesCtl]) * \
        #                 np.ones(dataNNcurrCol.loc[indicesCtl].shape[0])
        # pl.plot(regressorVectorNN[indicesCtl],correctedPred  , c='b')
        pl.title('%s' % data.columns[i])
        pl.legend()

        pl.show()

  return data



def prepareData(finalDataFile, tinyData):

  tadpoleFile = 'TADPOLE_D1_D2.csv'
  # dataDfTadpole = loadTadpole(tadpoleFile)
  # dataDfTadpole.to_csv('tadpoleCleanDf.csv', sep=',', quotechar='"')
  dataDfTadpole = pd.read_csv('tadpoleCleanDf.csv')

  # print(dsa)

  drcFile = 'drcVolsFSX.csv'

  # dataDfDrc = loadDRC(drcFile,columnsFormat=dataDfTadpole.columns)
  # dataDfDrc.to_csv('drcCleanDf.csv')
  dataDfDrc = pd.read_csv('drcCleanDf.csv')

  dataDfAll = pd.concat([dataDfTadpole, dataDfDrc], ignore_index=True)
  dataDfAll = dataDfAll[[x for x in dataDfAll.columns if x != 'Unnamed: 0']]

  # add extra number to RID to ensure no collisions occur with RIDs of other datasets
  dataDfAll['RID'] = dataDfAll['RID']*10 + dataDfAll['dataset']

  dataDfAll.to_csv('tadpoleDrcAll.csv')

  # exact same format as dataDfAll. make deep copy of the DRC data only

  validDf = dataDfAll[dataDfAll.dataset == 2]
  # validDf.drop(validDf.index[idxToDrop], inplace = True)
  validDf = validDf.copy(deep=True)
  validDf.reset_index(drop = True, inplace = True)
  # validDf.sort_index(inplace=True)
  # print('validDf', validDf)
  #validDf = addDRCValidDataMock(validDf) # generate random numbers for now
  validDf = addDRCValidData(validDf) # change to this real dataset one when ready


  visValidDf(validDf)
  print(asdas)

  validDf.to_csv('validDf.csv')

  print('validDf', validDf)

  testValidDfConsist(validDf, dataDfAll)

  print(dataDfTadpole.columns.tolist())
  print(dataDfDrc.columns.tolist())
  assert all([x == y for x,y in zip(dataDfTadpole.columns.tolist(), dataDfDrc.columns.tolist())])



  # regress out covariates: age, gender, ICV and dataset
  colsList = dataDfAll.columns.tolist()
  mriCols = [x for x in colsList if x.startswith('Volume')]
  allBiomkCols = dataDfAll.loc[:, 'CDRSB' : ].columns.tolist()

  # do for both the data and the validation set
  dataDfAll[mriCols], regParamsICV = regressCov(dataDfAll[mriCols],
    dataDfAll['ICV'], dataDfAll['diag'])

  validDf[mriCols] = applyRegFromParams(validDf[mriCols],
    validDf['ICV'], validDf['diag'], regParamsICV)

  testValidDfConsist(validDf, dataDfAll)

  dataDfAll[allBiomkCols], regParamsAge = regressCov(dataDfAll[allBiomkCols],
    dataDfAll['age'], dataDfAll['diag'])

  validDf[allBiomkCols] = applyRegFromParams(validDf[allBiomkCols],
    validDf['age'], validDf['diag'], regParamsAge)

  dataDfAll[allBiomkCols], regParamsGender = regressCov(dataDfAll[allBiomkCols],
    dataDfAll['gender-0f1m'], dataDfAll['diag'], printFigs=False)

  validDf[allBiomkCols] = applyRegFromParams(validDf[allBiomkCols],
    validDf['gender-0f1m'], validDf['diag'], regParamsGender)

  dataDfAll[allBiomkCols], regParamsDataset = regressCov(dataDfAll[allBiomkCols],
    dataDfAll['dataset'], dataDfAll['diag'], printFigs=False)

  validDf[allBiomkCols] = applyRegFromParams(validDf[allBiomkCols],
    validDf['dataset'], validDf['diag'], regParamsDataset)

  # change directionality of decreasing markers: volume, DTI-FA and FDG
  # This is because the model assumes all biomarkers are increasing
  dataDfAll[mriCols] *= -1
  validDf[mriCols] *= -1

  dtiFaCols = [x for x in colsList if x.startswith('DTI FA')]
  # print(dataDfAll[dtiFaCols])
  dataDfAll[dtiFaCols] *= -1
  validDf[dtiFaCols] *= -1

  fdgCols = [x for x in colsList if x.startswith('FDG')]
  dataDfAll[fdgCols] *= -1
  validDf[fdgCols] *= -1

  dataDfAll[['MMSE', 'RAVLT_immediate']] *= -1
  validDf[['MMSE', 'RAVLT_immediate']] *= -1

  # convert biomarkers to Z-scores
  # meanCtl = np.nanmean(dataDfAll[allBiomkCols][np.in1d(dataDfAll['diag'], [CTL, CTL2])],axis=0)
  # stdCtl = np.nanstd(dataDfAll[allBiomkCols][np.in1d(dataDfAll['diag'], [CTL, CTL2])], axis=0)
  # dataDfAll[allBiomkCols] = (np.array(dataDfAll[allBiomkCols]) - meanCtl[None, :]) / (stdCtl[None, :])

  # convert biomarkers to [0,1] interval
  minB = np.nanmin(dataDfAll[allBiomkCols], axis=0)
  maxB = np.nanmax(dataDfAll[allBiomkCols], axis=0)
  dataDfAll[allBiomkCols] = (np.array(dataDfAll[allBiomkCols]) - minB[None, :]) / (maxB - minB)[None, :]
  validDf[allBiomkCols] = (np.array(validDf[allBiomkCols]) - minB[None, :]) / (maxB - minB)[None, :]

  dtiCols = [c for c in validDf.columns if c.startswith('DTI')]

  # also normalise the validation set to be in the same space as ADNI.
  # Note that the previous dataset normalisation doesn't work, because in the training
  # set there were no DTI biomarkers in dataset 2.
  for c in range(len(dtiCols)):

    stdADNI =np.nanstd(dataDfAll.loc[:, dtiCols[c]])
    stdDRC = np.nanstd(validDf.loc[:, dtiCols[c]])
    stdRatio = stdDRC / stdADNI
    validDf.loc[:, dtiCols[c]] = validDf.loc[:, dtiCols[c]] / stdRatio

    meanADNI = np.nanmean(dataDfAll.loc[:, dtiCols[c]])
    meanDRC = np.nanmean(validDf.loc[:, dtiCols[c]])
    meanDiff = (meanDRC - meanADNI)
    validDf.loc[:, dtiCols[c]] = validDf.loc[:, dtiCols[c]] - meanDiff

    meanDRC = np.nanmean(validDf.loc[:, dtiCols[c]])
    stdDRC = np.nanstd(validDf.loc[:, dtiCols[c]])

    # print('ADNI mean', meanADNI)
    # print('ADNI std', stdADNI)
    # print('DRC mean', meanDRC)
    # print('DRC std', stdDRC)
    #
    # print(asda)

  print(dataDfAll.shape)
  if tinyData:
    # or try to balance the modalities, currently MRI seems to dominate the fitting.

    hasNonMriImgInd = ~np.isnan(dataDfAll['FDG Temporal']) | (~np.isnan(dataDfAll['DTI FA Temporal'])) \
      | (~np.isnan(dataDfAll['AV45 Temporal'])) | (~np.isnan(dataDfAll['AV1451 Temporal']))
    # print('hasNonMriImgInd', np.sum(hasNonMriImgInd))
    drcDatasetInd = dataDfAll.dataset == 2
    # print('drcDatasetInd', np.sum(drcDatasetInd))
    idxToDrop = np.logical_not(hasNonMriImgInd | drcDatasetInd)
    # print('idxToDrop', np.sum(idxToDrop))
    dataDfAll.drop(dataDfAll.index[idxToDrop], inplace=True)
    dataDfAll.reset_index(drop=True, inplace=True)

    unqRID = np.unique(dataDfAll.RID)
    adniUnqRID = np.unique(dataDfAll.RID[dataDfAll.dataset == 1])
    pcaUnqRID = np.unique(dataDfAll.RID[dataDfAll.dataset == 2])
    print('unqRID', unqRID.shape)
    print('adniUnqRID', adniUnqRID.shape)
    print('pcaUnqRID', pcaUnqRID.shape)
    # print(adas)
    ridToKeep = np.random.choice(adniUnqRID, 230, replace=False)
    ridToKeep = np.concatenate((ridToKeep, pcaUnqRID), axis=0)
    idxToDrop = np.logical_not(np.in1d(dataDfAll.RID, ridToKeep))
    dataDfAll.drop(dataDfAll.index[idxToDrop], inplace=True)
    dataDfAll.reset_index(drop=True, inplace=True)


  # fill in the missing diagnoses
  # print(np.sum(np.isnan(dataDfAll.diag)))
  unqRID = np.unique(dataDfAll.RID)
  for s in unqRID:
    idxCurrSubj = np.where(dataDfAll.RID == s)[0]

    monthCurrSubj = dataDfAll.Month_bl[idxCurrSubj]
    diagCurrSubj = dataDfAll.diag[idxCurrSubj]

    idxCurrSubjDiagExists = ~np.isnan(diagCurrSubj)

    if np.sum(idxCurrSubjDiagExists) > 0:
      for v in range(monthCurrSubj.shape[0]):
        if np.isnan(dataDfAll.diag[idxCurrSubj[v]]):
          timeDiffs = monthCurrSubj[idxCurrSubjDiagExists] - monthCurrSubj[idxCurrSubjDiagExists]
          dataDfAll.loc[idxCurrSubj[v], 'diag'] = diagCurrSubj[idxCurrSubjDiagExists][np.argmin(timeDiffs)]
    else:
      dataDfAll.loc[idxCurrSubj, 'diag'] = MCI # only one subj has absolutely no diag. assign MCI

  # print(np.sum(np.isnan(dataDfAll.diag)))
  # print(ads)

  if tinyData:
    dataDfAll.to_csv('tadpoleDrcRegDataTiny.csv')
  else:
    dataDfAll.to_csv('tadpoleDrcRegData.csv')

  validDf.to_csv('validDfReg.csv')

  # print(dataDfAll.shape)
  # print(ads)

  cogTests = dataDfAll.loc[:,'CDRSB' : 'FAQ' ].columns.tolist()

  for c in cogTests:
    for m in mriCols:
      nnInd = ~(np.isnan(dataDfAll.loc[:, c]) | np.isnan(dataDfAll.loc[:, m]))
      (r, p) = scipy.stats.pearsonr(dataDfAll.loc[nnInd, c], dataDfAll.loc[nnInd, m])

      print('%s - %s: r %f   pval %e' % (c, m, r, p))

  # print(asda)

  selectedBiomk = dataDfAll.loc[:, 'Volume Cingulate' : ].columns.tolist()
  selectedBiomk += ['ADAS13', 'CDRSB', 'RAVLT_immediate']

  # print(dataDfAll.dtypes)
  for c in selectedBiomk:
    dataDfAll[c] = dataDfAll[c].astype(np.float128) # increase precision of floats to 128
    validDf[c] = validDf[c].astype(np.float128)


  # print(dataDfAll.dtypes)
  # print(adsa)

  testValidDfConsist(validDf, dataDfAll)

  X, Y, RID, list_biomarkers, diag = \
    auxFunc.convert_table_marco(dataDfAll, list_biomarkers=selectedBiomk)



  # now drop all the mri values, which were used for testing consistency
  # and only keep the DTI. Don't remove the MRI cols though, needs to be in
  # same format as dataDfAll
  print('validDf', validDf.loc[:, mriCols])
  validDf.loc[:,mriCols] = np.nan
  print('validDf', validDf.loc[:,mriCols])


  # visValidDf(validDf)


  Xvalid, Yvalid, RIDvalid, _, diagValid = \
    auxFunc.convert_table_marco(validDf, list_biomarkers = selectedBiomk)

  print('validDf.RID', validDf.RID)
  print('RIDvalid', len(RIDvalid))
  # print(ads)

  ds = dict(X=X, Y=Y, RID=RID, list_biomarkers=list_biomarkers,
    dataDfAll=dataDfAll, regParamsICV=regParamsICV,
    regParamsAge=regParamsAge, regParamsGender=regParamsGender,
    regParamsDataset=regParamsDataset, diag=diag, Xvalid=Xvalid, Yvalid=Yvalid,
    RIDvalid=RIDvalid, diagValid=diagValid)
  pickle.dump(ds, open(finalDataFile, 'wb'), protocol=pickle.HIGHEST_PROTOCOL)

  # print('RID', RID)
  # print('X', len(X), len(X[0]))
  # print('Y', len(Y), len(Y[0]))
  # print(adsa)

def testValidDfConsist(validDf, dataDfAll):
  sID = np.unique(dataDfAll.scanID)[3]
  assert not np.isnan(sID)
  idxValid = np.where(validDf.scanID == sID)[0][0]
  idxData = np.where(dataDfAll.scanID == sID)[0][0]
  print('validDf.loc[3, Volume Parietal]', validDf.loc[idxValid, 'Volume Parietal'])
  print('dataDfAll.loc[3, Volume Parietal]', dataDfAll.loc[idxData, 'Volume Parietal'])
  print('validDf.loc[3, DTI FA Parietal]', validDf.loc[idxValid, 'DTI FA Parietal'])
  print('dataDfAll.loc[3, DTI FA Parietal]', dataDfAll.loc[idxData, 'DTI FA Parietal'])
  print('idxValid', idxValid)
  print('idxData', idxData)
  assert validDf.at[idxValid, 'Volume Parietal'] == \
    dataDfAll.loc[idxData, 'Volume Parietal']
  assert validDf.at[idxValid, 'DTI FA Parietal'] != \
    dataDfAll.at[idxData, 'DTI FA Parietal']

def visDataHist(dataDfAll):

  unqDiags = np.unique(dataDfAll.diag)
  biomks = dataDfAll.loc[:, 'CDRSB':].columns.tolist()
  for b in range(len(biomks)):

    fig = pl.figure(1)
    fig.clf()
    for d in unqDiags:
      pl.hist(dataDfAll.loc[dataDfAll.diag == d, biomks[b]].dropna(), bins=15,
        color=plotTrajParams['diagColors'][d], label=plotTrajParams['diagLabels'][d], alpha=0.5)

    pl.legend(loc='west')
    pl.title(biomks[b])

    fig.show()
    os.system('mkdir -p resfiles/tad-drc')
    fig.savefig('resfiles/tad-drc/%d_%s.png' % (b, biomks[b]))

# test why there are some subj with bad FDG in temporal
#   smallFdgInd = dataDfAll['FDG Temporal'] < 0.8
#   largeFdgInd = dataDfAll['FDG Temporal'] > 0.8
#
#   print('smallFdgInd', dataDfAll.loc[np.where(smallFdgInd)[0],:])
#   print('largeFdgInd', dataDfAll.loc[np.where(largeFdgInd)[0], :])
#
#   dfSmallFdg = dataDfAll.loc[smallFdgInd,:].copy()
#   dfLargeFdg = dataDfAll.loc[largeFdgInd,:].copy()
#
#   dfSmallFdg.to_csv('resfiles/tad-drc/smallFdg.csv')
#   dfLargeFdg.to_csv('resfiles/tad-drc/largeFdg.csv')

  # print(adsas)


def main():
  np.random.seed(1)
  random.seed(1)
  pd.set_option('display.max_columns', 50)
  tinyData = args.tinyData
  regenerateData = args.regData
  if tinyData:
    finalDataFile = 'tadpoleDrcTiny.npz'
    expName = 'tad-drcTiny'
  else:
    finalDataFile = 'tadpoleDrcFinalDataWithRegParams.npz'
    expName = 'tad-drc'

  if regenerateData:
    prepareData(finalDataFile, tinyData)

  ds = pickle.load(open(finalDataFile, 'rb'))
  dataDfAll = ds['dataDfAll']
  regParamsICV = ds['regParamsICV']
  regParamsAge = ds['regParamsAge']
  regParamsGender = ds['regParamsGender']
  regParamsDataset = ds['regParamsDataset']
  X = ds['X']
  Y = ds['Y']
  RID = np.array(ds['RID'])
  labels = ds['list_biomarkers']
  diag = ds['diag']

  # visDataHist(dataDfAll)
  nrUnqDiags = np.unique(dataDfAll.diag)
  print(dataDfAll)
  for d in nrUnqDiags:
    idxCurrDiag = ds['diag'] == d
    print('nr subj %s %d' % (plotTrajParams['diagLabels'][d], np.sum(idxCurrDiag)))
    # avgScans = []
    # print('avg scans %s %d' % plotTrajParams['diagLabels'][d])

  meanVols = np.array([np.mean(Y[0][s]) for s in range(RID.shape[0])])
  meanVols[diag != CTL2] = np.inf
  idxOfDRCSubjWithLowVol = np.argmin(meanVols)
  print('idxOfDRCSubjWithLowVol', idxOfDRCSubjWithLowVol)
  print(diag[idxOfDRCSubjWithLowVol])

  outFolder = 'resfiles/'

  params = {}

  nrFuncUnits = 6
  nrBiomkInFuncUnits = 5

  # nrBiomk = nrBiomkInFuncUnits * nrFuncUnits
  # print(labels)
  # print(adss)
  # mapBiomkToFuncUnits = np.array(list(range(nrFuncUnits)) * nrBiomkInFuncUnits)
  # should give smth like [0,1,2,3,0,1,2,3,0,1,2,3]

  # change the order of the functional units so that the hippocampus and occipital are fitted first
  unitPermutation = [5,3,2,1,4,0]
  mapBiomkToFuncUnits = np.array((unitPermutation * nrBiomkInFuncUnits) + [-1,-1,-1])
  unitNames = [l.split(' ')[-1] for l in labels]
  unitNames = [unitNames[i] for i in unitPermutation]
  nrBiomk = mapBiomkToFuncUnits.shape[0]

  biomkInFuncUnit = [0 for u in range(nrFuncUnits+1)]
  for u in range(nrFuncUnits):
    biomkInFuncUnit[u] = np.where(mapBiomkToFuncUnits == u)[0]
    # biomkInFuncUnit[u] += [nrBiomk-3, nrBiomk-2] # also add CDRSOB and ADAS in order hlp disentangle the trajectories and get better staging

  # add extra entry with other biomks to be added in the disease models
  biomkInFuncUnit[nrFuncUnits] = np.array([nrBiomk-3, nrBiomk-2, nrBiomk-1])


  plotTrajParams['biomkInFuncUnit'] = biomkInFuncUnit
  plotTrajParams['labels'] = labels
  plotTrajParams['nrRowsFuncUnit'] = 3
  plotTrajParams['nrColsFuncUnit'] = 4
  plotTrajParams['colorsTraj'] = [colorsys.hsv_to_rgb(hue, 1, 1) for hue in np.linspace(0, 1, num=nrBiomk, endpoint=False)]

  plotTrajParams['yNormMode'] = 'zScoreTraj'
  # plotTrajParams['yNormMode'] = 'zScoreEarlyStageTraj'
  # plotTrajParams['yNormMode'] = 'unscaled'

  # if False, plot estimated traj. in separate plot from true traj.
  plotTrajParams['allTrajOverlap'] = False

  params['runIndex'] = args.runIndex
  params['nrProc'] = args.nrProc
  params['cluster'] = args.cluster
  params['plotTrajParams'] = plotTrajParams
  params['penaltyUnits'] = args.penalty
  params['penaltyDis'] = args.penalty
  params['nrFuncUnits'] = nrFuncUnits
  params['biomkInFuncUnit'] = biomkInFuncUnit
  params['labels'] = labels

  params['X'] = X
  params['Y'] = Y
  params['RID'] = RID
  params['diag'] = diag
  params['plotTrajParams']['diag'] = params['diag']
  params['Xvalid'] = ds['Xvalid']
  params['Yvalid'] = ds['Yvalid']
  params['RIDvalid'] = ds['RIDvalid']
  params['diagValid'] = ds['diagValid']

  params['nrGlobIterUnit'] = 10 # these parameters are specific for the Joint Model of Disease (JMD)
  params['iterParamsUnit'] = 60
  params['nrGlobIterDis'] = 10
  params['iterParamsDis'] = 60

  nrSubj = len(ds['Yvalid'][0])
  for b in range(6,12):
    print(b, labels[b], [ds['Yvalid'][b][s][0] for s in range(nrSubj) if (ds['Yvalid'][b][s] and ds['diag'][s]
    == CTL)])
    print('mean CTL', np.mean([ds['Yvalid'][b][s][0] for s in range(nrSubj) if (ds['Yvalid'][b][s] and ds['diag'][s]
    == CTL)]))
    # print('mean AD', np.mean(ds['Yvalid'][b][ds['diag'] == AD]))

  print(ads)


  nrBiomkDisModel = nrFuncUnits + 3
  params['nrBiomkDisModel'] = nrBiomkDisModel

  nrXPoints = 50
  nrDis = 2 # nr of diseases
  params['trueParams'] = {}
  params['trueParams']['subShiftsTrueMarcoFormatS'] = np.zeros(RID.shape[0])
  params['trueParams']['trueSubjDysfuncScoresSU'] = np.zeros((RID.shape[0],nrFuncUnits))
  params['trueParams']['trueDysfuncXsX'] = np.linspace(0,1, nrXPoints)
  params['trueParams']['trueTrajXB'] = np.zeros((nrXPoints, nrBiomk))
  params['trueParams']['trueTrajFromDysXB'] = np.zeros((nrXPoints, nrBiomk))

  params['trueParams']['trueLineSpacedDPSsX'] = np.linspace(-10,10, nrXPoints)
  # params['trueParams']['trueTrajPredXB'] = np.zeros((nrXPoints,nrBiomk))
  params['trueParams']['trueDysTrajFromDpsXU'] = [np.zeros((nrXPoints,nrBiomkDisModel)) for d in range(nrDis)]

  params['plotTrajParams']['trueParams'] = params['trueParams']
  params['plotTrajParams']['unitNames'] = unitNames + labels[-3:]

  # map which diagnoses belong to which disease
  # first disease has CTL+AD, second disease has CTL2+PCA
  params['diagsSetInDis'] = [np.array([CTL, MCI, AD]), np.array([CTL2, PCA])]
  params['disLabels'] = ['tAD', 'PCA']
  params['otherBiomkPerDisease'] = [[nrBiomk-3,nrBiomk-2, nrBiomk-1], []]

  print('diag', params['diag'].shape[0])
  print('X[0]',len(params['X'][0]))
  assert params['diag'].shape[0] == len(params['X'][0])
  # assert params['diag'].shape[0] == len(params['trueParams']['subShiftsTrueMarcoFormatS'])
  # assert params['diag'].shape[0] == len(params['trueParams']['trueSubjDysfuncScoresSU'])

  if np.abs(args.penalty - int(args.penalty) < 0.00001):
    expName = '%sPen%d' % (expName, args.penalty)
  else:
    expName = '%sPen%.1f' % (expName, args.penalty)

  # params['runPartStd'] = ['L', 'L']
  params['runPartStd'] = args.runPartStd
  params['runPartMain'] = ['R', 'I', 'I'] # [mainPart, plot, stage]
  params['masterProcess'] = args.runIndex == 0

  expNameDisOne = '%s' % expName
  modelNames, res = evaluationFramework.runModels(params, expName,
   args.modelToRun, runAllExpTadpoleDrc)


# def addDRCValidDataMock(validDf):
#   state = np.random.get_state()
#   # print(np.random.rand())
#
#   nrDRCentries = np.sum(validDf.dataset == 2)
#   colsList = validDf.loc[:,'DTI FA Cingulate' : 'DTI FA Temporal'].columns.tolist()
#   # print('colsList', colsList)
#   validDf[colsList] = \
#     np.random.rand(nrDRCentries, len(colsList))
#
#   np.random.set_state(state)
#   # print(np.random.rand())
#   # print(asda)
#
#   return validDf

def addDRCValidData(validDf):
  '''perform validation on DTI data from the DRC '''

  #dtiSS = pd.read_csv('../data/DRC/DTI/DTI_summary_forRaz.xlsx')
  dtiSS = pd.read_csv('DTI_summary_forRaz.csv')
  mappingIDtoRegion = {0 : "Unclassified" ,
    1: ["Middle cerebellar peduncle", "ICP"], #
    2: ["Pontine Crossing tract","PCT"], #
    3: ["Genu of corpus callosum", "GCC"],
    4: ["Body of corpus callosum", "BCC"],
    5: ["Splenium of corpus callosum", "SCC"],
    6: ["Fornix (column and body of fornix)","FX"],
    7: ["Corticospinal tract R", "CST"],
    8: ["Corticospinal tract L", "CST"],
    9: ["Medial lemniscus R", "ML"],#
    10: ["Medial lemniscus L","ML"], #
    11: ["Inferior cerebellar peduncle R", "ICP"],  #
    12: ["Inferior cerebellar peduncle L", "ICP"],  #
    13: ["Superior cerebellar peduncle R", "SCP"],  #
    14: ["Superior cerebellar peduncle L", "SCP"],  #
    15: ["Cerebral peduncle R", "CP"],  #
    16: ["Cerebral peduncle L", "CP"],  #
    17: ["Anterior limb of internal capsule R", "ALIC"],  #
    18: ["Anterior limb of internal capsule L", "ALIC"],  #
    19: ["Posterior limb of internal capsule R", "PLIC"], #
    20: ["Posterior limb of internal capsule L", "PLIC"], #
    21: ["Retrolenticular part of internal capsule R", "RLIC"], #
    22: ["Retrolenticular part of internal capsule L", "RLIC"], #
    23: ["Anterior corona radiata R", "ACR"],
    24: ["Anterior corona radiata L", "ACR"],
    25: ["Superior corona radiata R", "SCR"],
    26: ["Superior corona radiata L", "SCR"],
    27: ["Posterior corona radiata R", "PCR"],
    28: ["Posterior corona radiata L", "PCR"],
    29: ["Posterior thalamic radiation R", "PTR"],
    30: ["Posterior thalamic radiation L", "PTR"],
    31: ["Sagittal stratum R", "SS"],
    32: ["Sagittal stratum L", "SS"],
    33: ["External capsule R", "EC"], #
    34: ["External capsule L", "EC"], #
    35: ["Cingulum (cingulate gyrus) R", "CGC"],
    36: ["Cingulum (cingulate gyrus) L", "CGC"],
    37: ["Cingulum (hippocampus) R", "CGH"],
    38: ["Cingulum (hippocampus) L", "CGH"],
    39: ["Fornix (cres) / Stria terminalis R", "FX"],
    40: ["Fornix (cres) / Stria terminalis L", "FX"],
    41: ["Superior longitudinal fasciculus R", "SLF"],
    42: ["Superior longitudinal fasciculus L", "SLF"],
    43: ["Superior fronto-occipital fasciculus R", "SFO"],
    44: ["Superior fronto-occipital fasciculus L", "SFO"],
    45: ["Uncinate fasciculus R", "UNC"],
    46: ["Uncinate fasciculus L", "UNC"],
    47: ["Tapetum R", "TP"],
    48: ["Tapetum L", "TP"]}

  dtiBiomkStructTemplate_updated = {
          'CST':'Frontal',
          'ACR':'Frontal',
          'SCR':'Frontal',
          'TP':'Frontal',
          'PCR':'Parietal',
          'PTR':'Parietal',
          'SS':'Temporal',
          'UNC':'Temporal',
          'SLF':'Occipital',
          'SFO':'Occipital',
          'CGC':'Cingulate',
          'GCC':'Cingulate',
          'BCC':'Cingulate',
          'SCC':'Cingulate',
          'CGH':'Hippocampus',
          'FX':'Hippocampus',
          'ALIC':'TBC',
          'PLIC':'TBC',
          'RLIC':'TBC',
          'ICP':'TBC',
          'SCP':'TBC',
          'CP':'TBC',
          'EC':'TBC',
          'PCT':'TBC',
          'EC':'TBC',
          'ML':'TBC',
          'n':'NA'
  }
  dtiSS['region'] = dtiSS['region'].map(lambda x: \
       'DTI FA '+dtiBiomkStructTemplate_updated[mappingIDtoRegion[x][1]])

  print(dtiSS)
  # print(asd)
  dtiSS_means = dtiSS.groupby(['Scan1Study','region', 'metric'])['mean']\
                  .mean().reset_index()

  idx = dtiSS_means.metric == 'fa'
  print('idx', idx)
  # dtiSS_means.drop(idx, inplace=True)
  dtiSS_means = dtiSS_means[idx]
  dtiSS_means.reset_index(drop=True, inplace=True)

  print('dtiSS_means', dtiSS_means)
  # print(asd)

  dtiSS_pivoted = dtiSS_means.\
          pivot(index = 'Scan1Study', columns = 'region', values = 'mean')

  unqScans_dti = np.unique(dtiSS_pivoted.index)
  unqScans_tad = np.unique(validDf.scanID)

  Scan_inter = list(set(unqScans_dti) & set(unqScans_tad))

  validDf_u = validDf.set_index('scanID')
  validDf_u.update(dtiSS_pivoted)
  validDf_u = validDf_u.reset_index()

  return validDf_u

def visValidDf(validDf):
  fig = pl.figure(1)
  dtiCols = validDf.loc[:, 'DTI FA Cingulate' : 'DTI FA Temporal'].columns.tolist()
  for b in range(len(dtiCols)):
    pl.clf()
    print(validDf.loc[validDf.diag == 4, dtiCols[b]].dropna())
    print(validDf.loc[validDf.diag == 5, dtiCols[b]].dropna())
    pl.hist(validDf.loc[validDf.diag == 4, dtiCols[b]].dropna(), color='g', label='ctl')
    pl.hist(validDf.loc[validDf.diag == 5, dtiCols[b]].dropna(), color='r', label='pca')
    pl.title(dtiCols[b])
    fig.show()
    fig.savefig('resfiles/tad-drc/valid_%d_%s.png' % (b, dtiCols[b]))



def runAllExpTadpoleDrc(params, expName, dpmBuilder, compareTrueParamsFunc = None):
  """ runs all experiments"""

  res = {}

  dpmBuilder.plotterObj.plotTrajParams = params['plotTrajParams']

  params['patientID'] = AD
  params['excludeID'] = -1
  params['excludeXvalidID'] = -1
  params['excludeStaging'] = [-1]

  params['outFolder'] = 'resfiles/%s' % expName
  params['expName'] = expName

  dpmObjStd, res['std'] = evaluationFramework.runStdDPM(params,
    expName, dpmBuilder, params['runPartMain'])


  # plotAllBiomkDisSpace(dpmObjStd, params, disNr=0)

  # perform the validation against DRC data
  validateDRCBiomk(dpmObjStd, params)


  return res


def plotAllBiomkDisSpace(dpmObj, params, disNr):
  # first predict subject DTI measures

  diag = params['diag']
  indxSubjToKeep = np.where(dpmObj.indxSubjForEachDisD[disNr])[0]

  nrBiomk = len(params['X'])
  print('nrBiomk', nrBiomk)
  Xfilt = [[] for b in range(nrBiomk)]
  Yfilt = [[] for b in range(nrBiomk)]
  for b in range(nrBiomk):
    Xfilt[b] = [params['X'][b][i] for i in indxSubjToKeep]
    Yfilt[b] = [params['Y'][b][i] for i in indxSubjToKeep]

  diagSubjCurrDis = diag[indxSubjToKeep]
  ridCurrDis = params['RID'][indxSubjToKeep]
  nrSubCurrDis = indxSubjToKeep.shape[0]

  XshiftedDisModelBS = [[] for b in range(nrBiomk)]
  ysPredBS = [[] for b in range(nrBiomk)]
  XshiftedDisModelUS, XdisModelUS, YdisModelUS = dpmObj.disModels[disNr].getData()
  xsOrigPred1S = XdisModelUS[0] # all biomarkers should contain all timepoints in the disease model


  for s in range(nrSubCurrDis):
    bTmp = 0 # some biomarker, doesn't matter which one
    ysCurrSubXB = dpmObj.predictBiomkSubjGivenXs(XshiftedDisModelUS[bTmp][s], disNr)

    for b in range(nrBiomk):
      ysPredBS[b] += [ysCurrSubXB[:,b]]

      if Xfilt[b][s].shape[0] > 0:
        # fix problem when a subject has the same xs twice (bad input dataset with same visit twice)
        while np.unique(Xfilt[b][s]).shape[0] < Xfilt[b][s].shape[0]:
          for x in Xfilt[b][s]:
            if np.sum(Xfilt[b][s] == x) > 1:
              idxToRemove = np.where(Xfilt[b][s] == x)[0][0]
              Yfilt[b][s] = np.concatenate((Yfilt[b][s][:idxToRemove], Yfilt[b][s][idxToRemove+1:]))
              Xfilt[b][s] = np.concatenate((Xfilt[b][s][:idxToRemove], Xfilt[b][s][idxToRemove + 1:]))

              break

        XshiftedDisModelBS[b] += [XshiftedDisModelUS[0][s]]
      else:
        XshiftedDisModelBS[b] += [[]]


  for b in range(nrBiomk):
    assert len(params['X'][b]) == len(params['Y'][b])
    assert len(XshiftedDisModelBS[b]) == len(Yfilt[b])

  # part 2. plot the inferred dynamics for DRC data:
  # every biomarker against original DPS
  # also plot extra validation data on top
  xsTrajX = dpmObj.disModels[disNr].getXsMinMaxRange()
  predTrajXB = dpmObj.predictBiomkSubjGivenXs(xsTrajX, disNr)
  trajSamplesBXS = dpmObj.sampleBiomkTrajGivenXs(xsTrajX, disNr, nrSamples = 100)

  fig = dpmObj.plotterObj.plotTrajInDisSpace(xsTrajX, predTrajXB, trajSamplesBXS,
    XshiftedDisModelBS, Yfilt, diagSubjCurrDis,
    None, None, None, replaceFig=True)
  fig.savefig('%s/allBiomkDisSpace%s.png' % (params['outFolder'], params['disLabels'][disNr]))



def validateDRCBiomk(dpmObj, params):
  # first predict subject DTI measures

  diag = params['diag']
  disNr = 1 # predict for DRC subjects
  indxSubjToKeep = np.where(dpmObj.indxSubjForEachDisD[disNr])[0]

  nrBiomk = len(params['X'])
  print('nrBiomk', nrBiomk)
  Xfilt = [[] for b in range(nrBiomk)]
  Yfilt = [[] for b in range(nrBiomk)]
  for b in range(nrBiomk):
    Xfilt[b] = [params['X'][b][i] for i in indxSubjToKeep]
    Yfilt[b] = [params['Y'][b][i] for i in indxSubjToKeep]

  diagSubjCurrDis = diag[indxSubjToKeep]
  ridCurrDis = params['RID'][indxSubjToKeep]
  nrSubCurrDis = indxSubjToKeep.shape[0]

  XshiftedDisModelBS = [[] for b in range(nrBiomk)]
  ysPredBS = [[] for b in range(nrBiomk)]
  XshiftedDisModelUS, XdisModelUS, YdisModelUS = dpmObj.disModels[disNr].getData()
  xsOrigPred1S = XdisModelUS[0] # all biomarkers should contain all timepoints in the disease model


  for s in range(nrSubCurrDis):
    bTmp = 0 # some biomarker, doesn't matter which one

    ysCurrSubXB = dpmObj.predictBiomkSubjGivenXs(XshiftedDisModelUS[bTmp][s], disNr)

    for b in range(nrBiomk):
      ysPredBS[b] += [ysCurrSubXB[:,b]]

      if Xfilt[b][s].shape[0] > 0:
        # fix problem when a subject has the same xs twice (bad input dataset with same visit twice)
        while np.unique(Xfilt[b][s]).shape[0] < Xfilt[b][s].shape[0]:
          for x in Xfilt[b][s]:
            if np.sum(Xfilt[b][s] == x) > 1:
              idxToRemove = np.where(Xfilt[b][s] == x)[0][0]
              Yfilt[b][s] = np.concatenate((Yfilt[b][s][:idxToRemove], Yfilt[b][s][idxToRemove+1:]))
              Xfilt[b][s] = np.concatenate((Xfilt[b][s][:idxToRemove], Xfilt[b][s][idxToRemove + 1:]))

              break

        XshiftedDisModelBS[b] += [XshiftedDisModelUS[0][s]]
      else:
        XshiftedDisModelBS[b] += [[]]


  for b in range(nrBiomk):
    assert len(params['X'][b]) == len(params['Y'][b])
    assert len(XshiftedDisModelBS[b]) == len(Yfilt[b])


  # now get the validation set. This is already only for the DRC subjects
  Xvalid = params['Xvalid']
  Yvalid = params['Yvalid']
  RIDvalid = params['RIDvalid']
  diagValid = params['diagValid']

  labels = params['labels']
  print('labels', labels)
  dtiColsIdx = [i for i in range(len(labels)) if labels[i].startswith('DTI')]

  assert len(ysPredBS) == len(Yvalid)

  nrDtiCols = len(dtiColsIdx)
  mse = [0 for b in dtiColsIdx]

  subjWithValidIndx = np.where([ys.shape[0] > 0 for ys in Yvalid[dtiColsIdx[0]]])[0]
  nrSubjWithValid = subjWithValidIndx.shape[0]
  YvalidFilt = [0 for b in range(nrBiomk)]
  XvalidFilt = [0 for b in range(nrBiomk)]
  diagValidFilt = diagValid[subjWithValidIndx]
  for b in range(nrBiomk):
    XvalidFilt[b] = [Xvalid[b][s] for s in subjWithValidIndx]
    YvalidFilt[b] = [Yvalid[b][s] for s in subjWithValidIndx]

  RIDvalidFilt = RIDvalid[subjWithValidIndx]

  XvalidShifFilt = [[[] for s in range(nrSubjWithValid)] for b in range(nrBiomk)]

  for b in range(nrDtiCols):
    mseList = []
    for s in range(RIDvalidFilt.shape[0]):
      # for each validation subject
      idxCurrDis = np.where(RIDvalidFilt[s] == ridCurrDis)[0][0]
      xsOrigFromModel = xsOrigPred1S[idxCurrDis]

      assert np.where(xsOrigFromModel == XvalidFilt[dtiColsIdx[b]][s][0])[0].shape[0] == 1
      idxXsWithValid = np.where(xsOrigFromModel == XvalidFilt[dtiColsIdx[b]][s][0])[0][0]
      ysPredCurrSubj = ysPredBS[dtiColsIdx[b]][idxCurrDis][idxXsWithValid]

      assert YvalidFilt[dtiColsIdx[b]][s].shape[0] > 0

      mseList += [(ysPredCurrSubj - YvalidFilt[dtiColsIdx[b]][s][0]) ** 2]

      # also compose the shifted Xs for the validation subjects
      xsShiftedFromModel = XshiftedDisModelBS[0][idxCurrDis]
      XvalidShifFilt[dtiColsIdx[b]][s] = np.array([xsShiftedFromModel[idxXsWithValid]])

      assert XvalidShifFilt[dtiColsIdx[b]][s].shape[0] == YvalidFilt[dtiColsIdx[b]][s].shape[0]


    mse[b] = np.mean(mseList)

  # print('mse', mse)
  # print(ads)


  # part 2. plot the inferred dynamics for DRC data:
  # every biomarker against original DPS
  # also plot extra validation data on top
  xsTrajX = dpmObj.disModels[disNr].getXsMinMaxRange()
  predTrajXB = dpmObj.predictBiomkSubjGivenXs(xsTrajX, disNr)
  trajSamplesBXS = dpmObj.sampleBiomkTrajGivenXs(xsTrajX, disNr, nrSamples=100)

  print('XshiftedDisModelBS', XshiftedDisModelBS)
  print('XvalidShifFilt', XvalidShifFilt)
  print('predTrajXB', predTrajXB[:,0])
  print('xsTrajX', xsTrajX)
  print('ysPredBS', ysPredBS)
  # print(ads)


  # fig = dpmObj.plotterObj.plotTrajInDisSpace(xsTrajX, predTrajXB, trajSamplesBXS,
  #   XshiftedDisModelBS, Yfilt, diagSubjCurrDis,
  #   XvalidShifFilt, YvalidFilt, diagValidFilt, replaceFig=True)
  # fig.savefig('%s/validPCA.png' % params['outFolder'])


  # plot just the DTI biomarkers
  dtiColsArrayIndx = np.array(dtiColsIdx)
  print('dtiColsArrayIndx', dtiColsArrayIndx)
  predTrajDtiXB = predTrajXB[:,dtiColsArrayIndx]
  trajSamplesDtiBXS = trajSamplesBXS[dtiColsArrayIndx,:,:]
  # print(len(XshiftedDisModelBS))
  # print(ads)
  XvalidShifDtiFilt = [XvalidShifFilt[b] for b in dtiColsIdx]
  YvalidFiltDti = [YvalidFilt[b] for b in dtiColsIdx]

  labelsDti = [params['labels'][b] for b in dtiColsIdx]

  fig = dpmObj.plotterObj.plotTrajInDisSpace(xsTrajX, predTrajDtiXB, trajSamplesDtiBXS,
    XvalidShifDtiFilt, YvalidFiltDti, diagValidFilt,
    None, None, None, labelsDti, replaceFig=True)
  fig.savefig('%s/validDtiPCA.png' % params['outFolder'])

  # plot just the trajectories by modality groups
  # fig = dpmObj.plotterObj.plotTrajInDisSpaceOverlap(xsTrajX, predTrajXB,
  #   trajSamplesBXS, params, replaceFig=True)
  # fig.savefig('%s/trajDisSpaceOverlap_%s_%s.png' % (params['outFolder'],
  # params['disLabels'][disNr], params['expName']))


if __name__ == '__main__':
  main()


