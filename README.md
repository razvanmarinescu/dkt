
# Disease Knowledge Transfer across Neurodegenerative Diseases
## Razvan V. Marinescu, Marco Lorenzi, Stefano B. Blumberg, Alexandra L. Young, Pere P. Morell, Neil P. Oxtoby, Arman Eshaghi, Keir X. Yong, Sebastian J. Crutch, Polina Golland, Daniel C. Alexander

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

To run model comparison from saved checkpoints, run:
```
python3 tadpoleDrc.py --runIndex 0 --nrProc 10 --modelToRun 0  --nrRows 5 --nrCols 7 --runPartStd LL --tinyData
```


## Training new models

To train a new model, run the commands above with "RR" flags. If you run on a different dataset, you have to put them in the same .csv format as those under data/ and data_processed/.

For example, to train on the TADPOLE + DRC data, run:

```
python3 tadpoleDrc.py --runIndex 0 --nrProc 10 --modelToRun 14  --nrRows 4 --nrCols 6 --penalty 5 --runPartStd RR --tinyData
```
