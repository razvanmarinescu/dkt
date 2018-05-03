from env import *
import pandas as pd
import numpy as np


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
