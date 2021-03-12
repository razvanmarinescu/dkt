
# Disease Knowledge Transfer across Neurodegenerative Diseases
Razvan V. Marinescu, Marco Lorenzi, Stefano B. Blumberg, Alexandra L. Young, Pere P. Morell, Neil P. Oxtoby, Arman Eshaghi, Keir X. Yong, Sebastian J. Crutch, Polina Golland, Daniel C. Alexander

![overall diagram](disease_knowledge_transfer.png)

paper: https://arxiv.org/pdf/1901.03517.pdf<br>
poster: https://github.com/razvanmarinescu/dkt/blob/master/poster-dkt.pdf<br>


Abstract: *We introduce Disease Knowledge Transfer (DKT), a novel technique for transferring biomarker information between related neurodegenerative diseases. DKT infers robust multimodal biomarker trajectories in rare neurodegenerative diseases even when only limited, unimodal data is available, by transferring information from larger multimodal datasets from common neurodegenerative diseases. DKT is a joint-disease generative model of biomarker progressions, which exploits biomarker relationships that are shared across diseases. As opposed to current deep learning approaches, DKT is interpretable, which allows us to understand underlying disease mechanisms, and can also predict the future evolution of subjects instead of solving simpler control vs diseased classification tasks. Here we demonstrate DKT on Alzheimer's disease (AD) variants and its ability to predict trajectories for multimodal biomarkers in Posterior Cortical Atrophy (PCA), in lack of such data from PCA subjects. For this we train DKT on a combined dataset containing subjects with two distinct diseases and sizes of data available: 1) a larger, multimodal typical AD (tAD) dataset from the TADPOLE Challenge, and 2) a smaller unimodal Posterior Cortical Atrophy (PCA) dataset from the Dementia Research Centre (DRC) UK, for which only a limited number of Magnetic Resonance Imaging (MRI) scans are available. We first show that the estimated multimodal trajectories in PCA are plausible as they agree with previous literature. We further validate DKT in two situations: (1) on synthetic data, showing that it can accurately estimate the ground truth parameters and (2) on 20 DTI scans from controls and PCA patients, showing that it has favourable predictive performance compared to standard approaches. While we demonstrated DKT on Alzheimer's variants, we note DKT is generalisable to other forms of related neurodegenerative diseases.*



## Installation

``` 
pip install numpy scipy pandas pickle matplotlib

```

## Running pre-trained models from saved checkpoints

To reproduce the synthetic results in paper pre-trained model, run with "LL" flag. First "L" loads the initialisation, while the second "L" loads the final trained model. 

```
python3 jointSynth.py --runIndex 0 --nrProc 10 --modelToRun 14  --nrRows 3 --nrCols 4 --runPartStd LL --expName synth1
```

To reproduce the results with ADNI + DRC data from pre-trained model, using saved checkpoint, run:

```
python3 tadpoleDrc.py --runIndex 0 --nrProc 10 --modelToRun 14  --nrRows 4 --nrCols 6 --penalty 5 --runPartStd LL --tinyData
```

To run DKT on 3 pre-defined subgroups from TADPOLE (Hippocampal, Cortical, Subcortical) defined by the SuStaIn model (Young et al, Nature Comms, 2018), run:

```
  python3 tadpoleSubtypes.py --runIndex 0 --nrProc 1 --modelToRun 14  --nrRows 4 --nrCols 6 --runPartStd RR --tinyData
```

To run model comparison from saved checkpoints, run with --modelToRun=0 (i.e. run all models). This only works in LL mode. 

```
python3 tadpoleDrc.py --runIndex 0 --nrProc 10 --modelToRun 0  --nrRows 5 --nrCols 7 --runPartStd LL --tinyData
```


## Training new models

To train a new model, run the commands above with "RR" flags. If you run on a different dataset, you have to put them in the same .csv format as those under data/ and data_processed/. For example, to train on the TADPOLE + DRC data, run:

```
python3 tadpoleDrc.py --runIndex 0 --nrProc 10 --modelToRun 14  --nrRows 4 --nrCols 6 --penalty 5 --runPartStd RR --tinyData
```

## Running other models

To run the Disease progression Score model by Jedynak et al, NeuroImage, 2012, simply run:

```
  python3 tadpoleDrc.py --runIndex 0 --nrProc 10 --modelToRun 17  --nrRows 5 --nrCols 7 --runPartStd RR --tinyData
```

