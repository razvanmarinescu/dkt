jointSynth_MarcoModel:
	python3 jointSynth.py --runIndex 0 --nrProc 10 --modelToRun 15 --nrRows 3 --nrCols 3

jointSynth_JMD:
	python3 jointSynth.py --runIndex 0 --nrProc 10 --modelToRun 14  --nrRows 3 --nrCols 4 --runPartStd LR --expName synth1

jointSynth2_JMD:
	python3 jointSynth.py --runIndex 0 --nrProc 10 --modelToRun 14  --nrRows 3 --nrCols 4--runPartStd LL --expName synth2

MarcoTestADNI:
	python3 MarcoTestADNI.py

tadpoleDrc_JMD:
	python3 tadpoleDrc.py --runIndex 0 --nrProc 10 --modelToRun 14  --nrRows 4 --nrCols 6 --penalty 5 --runPartStd LL --tinyData

tadpoleDrc_Sig:
	python3 tadpoleDrc.py --runIndex 0 --nrProc 10 --modelToRun 17  --nrRows 5 --nrCols 7 --runPartStd LL --tinyData

tadpoleDrc_ModelComparison:
	python3 tadpoleDrc.py --runIndex 0 --nrProc 10 --modelToRun 0  --nrRows 5 --nrCols 7 --runPartStd LL --tinyData

tadpoleSubtypes_JMD:
	python3 tadpoleSubtypes.py --runIndex 0 --nrProc 1 --modelToRun 14  --nrRows 4 --nrCols 6 --runPartStd RR --tinyData

tadpoleSubtypes_Sig:
	python3 tadpoleSubtypes.py --runIndex 0 --nrProc 1 --modelToRun 17  --nrRows 5 --nrCols 7 --runPartStd LL --tinyData

tadpoleSubtypes_Valid:
	python3 tadpoleSubtypes.py --runIndex 0 --nrProc 1 --modelToRun 0  --nrRows 4 --nrCols 6 --runPartStd LL --tinyData
