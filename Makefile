jointSynth_MarcoModel:
	python3 jointSynth.py --runIndex 0 --nrProc 10 --modelToRun 15 --nrRows 3 --nrCols 3 --penalty 5

jointSynth_JMD:
	python3 jointSynth.py --runIndex 0 --nrProc 10 --modelToRun 14  --nrRows 3 --nrCols 4 --penalty 5 --runPartStd RR --expName synth1

jointSynth2_JMD:
	python3 jointSynth.py --runIndex 0 --nrProc 10 --modelToRun 14  --nrRows 3 --nrCols 4 --penalty 5 --runPartStd RR --expName synth2

MarcoTestADNI:
	python3 MarcoTestADNI.py

tadpoleDrc_JMD:
	python3 tadpoleDrc.py --runIndex 0 --nrProc 10 --modelToRun 14  --nrRows 4 --nrCols 6 --runPartStd RR --tinyData

tadpoleDrc_Sig:
	python3 tadpoleDrc.py --runIndex 0 --nrProc 10 --modelToRun 17  --nrRows 5 --nrCols 7 --runPartStd LL --tinyData

tadpoleDrc_ModelComparison:
	python3 tadpoleDrc.py --runIndex 0 --nrProc 10 --modelToRun 0  --nrRows 5 --nrCols 7 --runPartStd LL --tinyData
