from launchCommon import *


parser = argparse.ArgumentParser(
  description=r'''Launches in synthetic experiments with different parameters.
  Scripts can run in three modes: parallel, serial or on cluster

  ''')

parser.add_argument('--penalty', dest='penalty', type=int,
  help='penalty value for non-monotonic trajectories. between 0 (no effect) and 10 (strong effect). ')


addParserArgs(parser)

args = parser.parse_args()

launchParams = initCommonLaunchParams(args)

def getRunCmd(model, runIndex, nrClust, agg):
  """
  :param model:
  :param runIndex:
  :param nrClust:
  :param agg: if 1 then plot without Xwindows (for cluster)
              if 0 then plot with Xwindows (for personal machine)
  :return:
  runCmdList command representation as a list of pairs
  runCmdStr command representatition as a string
  """
  runCmdList = [('python3', args.launcherScript),
    ('--runIndex', runIndex),
    ('--nrProc',  args.nrProc),
    ('--modelToRun', model),
    ('--nrOuterIt', args.nrOuterIt),
    ('--nrInnerIt', args.nrInnerIt), ('--penalty', penalty),

  ]

  runCmdStr = ' '.join(['%s %s' % (s,str(a)) for (s,a) in runCmdList])

  return runCmdList, runCmdStr

def getQsubCmd(model, runIndex, nrClust):
  ''' Creates full command for cluster'''
  # if there's an error about tty, add & in the last parameter
  runCmd = ('cd %s; /share/apps/python-3.5.1/bin/' %
            launchParams['REPO_DIR']) + \
           getRunCmd(model,runIndex, nrClust, agg=1)[1]
  runCmd += ' --cluster '
  jobName = "%s_i%d_m%d_c%d_%s_r%.1f_p%d" % (args.launcherScript.split('.')[0],
    runIndex,model,nrClust,args.initClustering,args.rangeFactor,
    args.informPrior)
  qsubCmd = getQsubCmdPart2(launchParams, jobName)
  cmdClust = 'echo "%s" | %s' % (runCmd, qsubCmd) # echo the matlab cmd then pipe it to qsub
  return cmdClust, runCmd

pList = []
#instanceIndices = [2,3,4,5,6,7,8];
instanceIndices = range(args.firstInstance, args.lastInstance+1)
quitFlag = 1

if args.firstModel and args.lastModel:
  modelIndices = range(args.firstModel, args.lastModel + 1)
elif args.models:
  modelIndices = [int(i) for i in args.models.split(',')]
else:
  raise ValueError('need to set either --models or --firstModel & --lastModel')

# print(modelIndices)
# print(adsa)

if args.nrClust:
  nrClustList = [args.nrClust]
elif args.nrClustList:
  nrClustList = [int(i) for i in args.nrClustList.split(',')]
else:
  raise ValueError('set either --nrClust or --nrClustList')


for m in modelIndices:
  for i in instanceIndices:
    for c in nrClustList:
      if not args.cluster:
        if args.serial:
          _, cmd = getRunCmd(m, i, c, agg=0)
          os.system(cmd)
        else:
          cmdArgs,runCmd = getRunCmd(m, i, c, agg=0)
          print(runCmd.split(' '))
          p = subprocess.Popen(runCmd.split(' '))
          pList.append(p)
      else:
        # run on cluster
        cmdClust, runCmd = getQsubCmd(m, i, c)
        print(cmdClust)
        if not args.printOnly:
          os.system(cmdClust)

# if I launch processes on local machine, wait for them to finish.
if not args.cluster and not args.serial:
  nrProcs = len(pList)
  for i in range(nrProcs):
    p = pList.pop()
    print(p)
    p.wait()

    print("---------------------->finished")