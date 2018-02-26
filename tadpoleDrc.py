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
import aux
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
from aux import *
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
plotTrajParams['diagLabels'] = {CTL:'CTL ADNI', MCI:'MCI ADNI', AD:'AD ADNI',
  CTL2:'CTL DRC', PCA:'PCA DRC', AD2:'AD DRC'}

plotTrajParams['freesurfPath'] = freesurfPath
# plotTrajParams['ylimitsRandPoints'] = (-3,2)
plotTrajParams['blenderPath'] = blenderPath
plotTrajParams['isSynth'] = True


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

  df[cols] = df[cols].apply(pd.to_numeric, errors='coerce', axis=1)
  pickle.dump(df, open('tadpoleCleanDf.npz', 'wb'), protocol=pickle.HIGHEST_PROTOCOL)
  df = pickle.load(open('tadpoleCleanDf.npz', 'rb'))

  dataDf = df[['RID', 'Month_bl']]
  dataDf['scanID'] = np.nan
  print('dataDf', dataDf.loc[:20,:])

  mapGender = {'Female' : 0, 'Male' : 1}
  df['PTGENDER'] = df['PTGENDER'].map(mapGender)

  # print(ads)

  dataDf['gender-0f1m'] = df['PTGENDER']
  dataDf['age'] = df['AGE'] + (df['Month_bl'] / 12)
  dataDf['dataset'] = 1
  ssTagMRI = 'UCSFFSX_11_02_15_UCSFFSX51_08_01_16'
  ICV = df['ST10CV_%s' % ssTagMRI]
  dataDf['ICV'] = ICV
  dataDf['diag'] = df['DXCHANGE']
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


  dataDf = addBiomks(fdgBiomkStruct, df, dataDf, collapseFunc=np.nanmean)

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
    targetDf[biomkLabels[b]] = pd.Series(np.sum(dataFrameCurrSubset,axis=1),
      index=targetDf.index)
    # targetDf = targetDf.assign(asd=np.sum(dataFrameCurrSubset,axis=1))
    # print(targetDf.loc[:5, biomkLabels[b]])
    # print(ads)



  return targetDf

def regressCov(data, regressorVector, diag, diagsCTL = (CTL, CTL2), printFigs=False):
  oldData = copy.deepcopy(data)
  M = np.zeros((data.shape[1], 2))
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
      data.loc[:, currCol] = data.loc[:,currCol] - (Yhat - np.nanmean(
        dataNNcurrCol.loc[indicesCtl]))

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

  params = dict(M=M, labels=data.columns)

  return data, params

def prepareData(finalDataFile, tinyData):

  tadpoleFile = 'TADPOLE_D1_D2.csv'
  dataDfTadpole = loadTadpole(tadpoleFile)
  dataDfTadpole.to_csv('tadpoleCleanDf.csv', sep=',', quotechar='"')
  dataDfTadpole = pd.read_csv('tadpoleCleanDf.csv')

  # print(dsa)

  drcFile = 'drcVolsFSX.csv'

  dataDfDrc = loadDRC(drcFile,columnsFormat=dataDfTadpole.columns)
  dataDfDrc.to_csv('drcCleanDf.csv')
  dataDfDrc = pd.read_csv('drcCleanDf.csv')

  dataDfAll = pd.concat([dataDfTadpole, dataDfDrc], ignore_index=True)
  dataDfAll = dataDfAll[[x for x in dataDfAll.columns if x != 'Unnamed: 0']]

  print(dataDfTadpole.columns.tolist())
  print(dataDfDrc.columns.tolist())
  assert all([x == y for x,y in zip(dataDfTadpole.columns.tolist(), dataDfDrc.columns.tolist())])

  # dataDfAll = dataDfTadpole
  # dataDfAll.set_index('key').join(dataDfDrc.set_index('key'))

  # add extra number to RID to ensure no collisions occur with RIDs of other datasets
  # print((dataDfAll[['RID']]*10 + dataDfAll[['dataset']]).shape, dataDfAll[['RID']]*10 + dataDfAll[['dataset']])

  dataDfAll['RID'] = dataDfAll['RID']*10 + dataDfAll['dataset']
  if tinyData:
    dataDfAll.to_csv('tadpoleDrcAllTiny.csv')
  else:
    dataDfAll.to_csv('tadpoleDrcAll.csv')

  # regress out covariates: age, gender, ICV and dataset
  colsList = dataDfAll.columns.tolist()
  mriCols = [x for x in colsList if x.startswith('Volume')]
  allBiomkCols = dataDfAll.loc[:, 'CDRSB' : ].columns.tolist()

  # also make the MRI volumes increasing

  dataDfAll[mriCols], regParamsICV = regressCov(dataDfAll[mriCols],
    dataDfAll['ICV'], dataDfAll['diag'])

  dataDfAll[allBiomkCols], regParamsAge = regressCov(dataDfAll[allBiomkCols],
    dataDfAll['age'], dataDfAll['diag'])

  dataDfAll[allBiomkCols], regParamsGender = regressCov(dataDfAll[allBiomkCols],
    dataDfAll['gender-0f1m'], dataDfAll['diag'], printFigs=False)

  dataDfAll[allBiomkCols], regParamsDataset = regressCov(dataDfAll[allBiomkCols],
    dataDfAll['dataset'], dataDfAll['diag'], printFigs=False)

  # change directionality of decreasing markers: volume, DTI-FA and FDG
  # This is because the model assumes all biomarkers are increasing
  dataDfAll[mriCols] *= -1
  dtiFaCols = [x for x in colsList if x.startswith('DTI FA')]
  # print(dataDfAll[dtiFaCols])
  dataDfAll[dtiFaCols] *= -1

  fdgCols = [x for x in colsList if x.startswith('FDG')]
  dataDfAll[fdgCols] *= -1

  dataDfAll[['MMSE', 'RAVLT_immediate']] *= -1


  # convert biomarkers to Z-scores
  # meanCtl = np.nanmean(dataDfAll[allBiomkCols][np.in1d(dataDfAll['diag'], [CTL, CTL2])],axis=0)
  # stdCtl = np.nanstd(dataDfAll[allBiomkCols][np.in1d(dataDfAll['diag'], [CTL, CTL2])], axis=0)
  # dataDfAll[allBiomkCols] = (np.array(dataDfAll[allBiomkCols]) - meanCtl[None, :]) / (stdCtl[None, :])

  # convert biomarkers to [0,1] interval
  minB = np.nanmin(dataDfAll[allBiomkCols], axis=0)
  maxB = np.nanmax(dataDfAll[allBiomkCols], axis=0)
  dataDfAll[allBiomkCols] = (np.array(dataDfAll[allBiomkCols]) - minB[None, :]) / (maxB - minB)[None, :]

  print(dataDfAll.shape)
  if tinyData:
    # or try to balance the modalities, currently MRI seems to dominate the fitting.

    hasNonMriImgInd = ~np.isnan(dataDfAll['FDG Temporal']) | (~np.isnan(dataDfAll['DTI FA Temporal'])) \
      | (~np.isnan(dataDfAll['AV45 Temporal'])) | (~np.isnan(dataDfAll['AV1451 Temporal']))
    print('hasNonMriImgInd', np.sum(hasNonMriImgInd))
    drcDatasetInd = dataDfAll.dataset == 2
    print('drcDatasetInd', np.sum(drcDatasetInd))
    idxToDrop = np.logical_not(hasNonMriImgInd | drcDatasetInd)
    print('idxToDrop', np.sum(idxToDrop))
    dataDfAll.drop(dataDfAll.index[idxToDrop], inplace=True)
    dataDfAll.reset_index(drop=True, inplace=True)

    unqRID = np.unique(dataDfAll.RID)
    # adniUnqRID = np.unique(dataDfAll.RID[dataDfAll.dataset == 1])
    # pcaUnqRID = np.unique(dataDfAll.RID[dataDfAll.dataset == 2])
    # print('unqRID', adniUnqRID.shape)
    ridToKeep = np.random.choice(unqRID, 200, replace=False)
    # ridToKeep = np.concatenate((ridToKeep, pcaUnqRID), axis=0)
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

  # print(dataDfAll.dtypes)
  # print(adsa)

  X, Y, RID, list_biomarkers, diag = \
    aux.convert_table_marco(dataDfAll, list_biomarkers=selectedBiomk)

  ds = dict(X=X, Y=Y, RID=RID, list_biomarkers=list_biomarkers,
    dataDfAll=dataDfAll, regParamsICV=regParamsICV,
    regParamsAge=regParamsAge, regParamsGender=regParamsGender,
    regParamsDataset=regParamsDataset, diag=diag)
  pickle.dump(ds, open(finalDataFile, 'wb'), protocol=pickle.HIGHEST_PROTOCOL)

  # print('RID', RID)
  # print('X', len(X), len(X[0]))
  # print('Y', len(Y), len(Y[0]))
  # print(adsa)

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


def main():
  np.random.seed(1)
  random.seed(1)
  pd.set_option('display.max_columns', 50)
  tinyData = True
  regenerateData = False
  if tinyData:
    finalDataFile = 'tadpoleDrcTiny.npz'
  else:
    finalDataFile = 'tadpoleDrcFinalDataWithRegParams.npz'

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

  meanVols = np.array([np.mean(Y[0][s]) for s in range(RID.shape[0])])
  meanVols[diag != CTL2] = np.inf
  idxOfDRCSubjWithLowVol = np.argmin(meanVols)
  print('idxOfDRCSubjWithLowVol', idxOfDRCSubjWithLowVol)
  print(diag[idxOfDRCSubjWithLowVol])
  assert X[0][idxOfDRCSubjWithLowVol].shape[0] == 3
  # print(asd)


  outFolder = 'resfiles/'

  expName = 'tad-drc'

  params = {}

  nrFuncUnits = 6
  nrBiomkInFuncUnits = 5

  # nrBiomk = nrBiomkInFuncUnits * nrFuncUnits
  print(labels)
  # mapBiomkToFuncUnits = np.array(list(range(nrFuncUnits)) * nrBiomkInFuncUnits)
  # should give smth like [0,1,2,3,0,1,2,3,0,1,2,3]

  # change the order of the functional units so that the hippocampus and occipital are fitted first
  unitPermutation = [5,3,2,1,4,0]
  mapBiomkToFuncUnits = np.array((unitPermutation * nrBiomkInFuncUnits) + [-1,-1,-1])
  unitNames = [l.split(' ')[-1] for l in labels]
  unitNames = [unitNames[i] for i in unitPermutation]

  nrBiomk = mapBiomkToFuncUnits.shape[0]
  # print('mapBiomkToFuncUnits', mapBiomkToFuncUnits)
  # print([unitNames[i] for i in mapBiomkToFuncUnits])
  # print(unitNames[0])
  # print([labels[i] for i in np.where(mapBiomkToFuncUnits == 0)[0]])
  # print(asd)

  plotTrajParams['mapBiomkToFuncUnits'] = mapBiomkToFuncUnits
  plotTrajParams['labels'] = labels
  plotTrajParams['nrRowsFuncUnit'] = 3
  plotTrajParams['nrColsFuncUnit'] = 3
  plotTrajParams['colorsTraj'] = [colorsys.hsv_to_rgb(hue, 1, 1) for hue in np.linspace(0, 1, num=nrBiomk, endpoint=False)]
  plotTrajParams['zScoreTraj'] = True

  # if False, plot estimated traj. in separate plot from true traj.
  plotTrajParams['allTrajOverlap'] = False

  params['runIndex'] = args.runIndex
  params['nrProc'] = args.nrProc
  params['cluster'] = args.cluster
  params['plotTrajParams'] = plotTrajParams
  params['penalty'] = args.penalty
  params['nrFuncUnits'] = nrFuncUnits
  params['mapBiomkToFuncUnits'] = mapBiomkToFuncUnits
  params['labels'] = labels


  params['X'] = X
  params['Y'] = Y
  params['RID'] = RID
  params['diag'] = diag
  params['plotTrajParams']['diag'] = params['diag']

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

  params['runPartStd'] = ['R', 'R']
  params['runPartMain'] = ['R', 'I', 'I'] # [mainPart, plot, stage]
  params['masterProcess'] = args.runIndex == 0

  expNameDisOne = '%s' % expName
  modelNames, res = evaluationFramework.runModels(params, expName,
    args.modelToRun, runAllExpTadpoleDrc)

  validateWithDtiDRC()

def validateWithDtiDRC():
  '''perform validation on DTI data from the DRC '''

  dtiSS = pd.read_csv('../data/DRC/DTI/DTI_summary_forRaz.xlsx')

  mappingIDtoRegion = {0 : "Unclassified" ,
    1: "Middle cerebellar peduncle",
    2: "Pontine Crossing tract",
    3: "Genu of corpus callosum",
    4: "Body of corpus callosum",
    5: "Splenium of corpus callosum",
    6: "Fornix (column and body of fornix)",
    7: "Corticospinal tract R",
    8: "Corticospinal tract L",
    9: "Medial lemniscus R",
    10: "Medial lemniscus L",
    11: "Inferior cerebellar peduncle R",
    12: "Inferior cerebellar peduncle L",
    13: "Superior cerebellar peduncle R",
    14: "Superior cerebellar peduncle L",
    15: "Cerebral peduncle R",
    16: "Cerebral peduncle L",
    17: "Anterior limb of internal capsule R",
    18: "Anterior limb of internal capsule L",
    19: "Posterior limb of internal capsule R",
    20: "Posterior limb of internal capsule L",
    21: "Retrolenticular part of internal capsule R",
    22: "Retrolenticular part of internal capsule L",
    23: "Anterior corona radiata R",
    24: "Anterior corona radiata L",
    25: "Superior corona radiata R",
    26: "Superior corona radiata L",
    27: "Posterior corona radiata R",
    28: "Posterior corona radiata L",
    29: "Posterior thalamic radiation R",
    30: "Posterior thalamic radiation L",
    31: "Sagittal stratum R",
    32: "Sagittal stratum L",
    33: "External capsule R",
    34: "External capsule L",
    35: "Cingulum (cingulate gyrus) R",
    36: "Cingulum (cingulate gyrus) L",
    37: "Cingulum (hippocampus) R",
    38: "Cingulum (hippocampus) L",
    39: "Fornix (cres) / Stria terminalis R",
    40: "Fornix (cres) / Stria terminalis L",
    41: "Superior longitudinal fasciculus R",
    42: "Superior longitudinal fasciculus L",
    43: "Superior fronto-occipital fasciculus R",
    44: "Superior fronto-occipital fasciculus L",
    45: "Uncinate fasciculus R",
    46: "Uncinate fasciculus L",
    47: "Tapetum R",
    48: "Tapetum L"}


  dtiSS['region'] = dtiSS['region'].map(mappingIDtoRegion)

  unqScans = np.unique(dtiSS.Scan1Study)
  nrUnqScans = unqScans.shape[0]

  dfNew = pd.DataFrame(np.nan * np.ones((nrUnqScans, len(columnsFormat))),
    columns=columns)

  metrics = ['rd','fa','ad', 'md']
  regions = [mappingIDtoRegion[k] for k in range(len(mappingIDtoRegion.keys()))]
  columns = []

  for m in metrics:
    for r in regions:
      columns += ['%s_%s' % (r, m)]

  columns += []

  for s in range(nrUnqScans):
    pass




def runAllExpTadpoleDrc(params, expName, dpmBuilder, compareTrueParamsFunc = None):
  """ runs all experiments"""

  res = {}

  dpmBuilder.plotterObj.plotTrajParams = params['plotTrajParams']

  params['patientID'] = AD
  params['excludeID'] = -1
  params['excludeXvalidID'] = -1
  params['excludeStaging'] = [-1]

  params['outFolder'] = 'resfiles/%s' % expName

  dpmObjStd, res['std'] = evaluationFramework.runStdDPM(params,
    expName, dpmBuilder, params['runPartMain'])

  return res



if __name__ == '__main__':
  main()


