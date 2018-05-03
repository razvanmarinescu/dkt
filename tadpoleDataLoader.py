from env import *
import pandas as pd
import numpy as np


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
