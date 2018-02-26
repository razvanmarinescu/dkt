jointSynth_MarcoModel:
	python3 jointSynth.py --runIndex 0 --nrProc 10 --modelToRun 15 --nrRows 3 --nrCols 3 --penalty 5

jointSynth_JMD:
	python3 jointSynth.py --runIndex 0 --nrProc 10 --modelToRun 14  --nrRows 3 --nrCols 3 --penalty 20

MarcoTestADNI:
	python3 MarcoTestADNI.py

tadpoleDrc_JMD:
	python3 tadpoleDrc.py --runIndex 0 --nrProc 10 --modelToRun 14  --nrRows 3 --nrCols 3 --penalty 1 --runPart LL

