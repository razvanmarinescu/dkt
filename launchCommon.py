import os
import glob
import sys
import argparse
from socket import gethostname
from subprocess import check_output as qx
import subprocess
import nibabel as nib
import numpy as np
import csv
from datetime import datetime
import pickle


def addParserArgs(parser):
  parser.add_argument('--firstInstance', dest='firstInstance', type=int,
                      default=1,help='index of first run instance/process')

  parser.add_argument('--lastInstance', dest='lastInstance', type=int,
                     default=10,help='index of last run instance/process')

  parser.add_argument('--nrProc', dest='nrProc', type=int,
                     default=10,help='# of processes')

  parser.add_argument('--firstModel', dest='firstModel', type=int,
                     help='index of first experiment to run')

  parser.add_argument('--lastModel', dest='lastModel', type=int,
                     help='index of last experiment to run')

  parser.add_argument('--models', dest='models',
                     help='index of first experiment to run')

  parser.add_argument('--launcherScript', dest='launcherScript',
                     help='launcher script, such as adniThick.py, etc ..')

  parser.add_argument('--cluster', action="store_true", help='set this if the program should run on the CMIC cluster')
  parser.add_argument('--timeLimit', dest='timeLimit',
                     help='timeLimit for the job in hours')
  parser.add_argument('--printOnly', action="store_true", help='only print experiment to be run, not actualy run it')

  # specific to the VDPM


  parser.add_argument('--nrParamsIt', dest='nrParamsIt', type=int, default=50,
                     help='# of outer iterations to run, for estimating clustering probabilities')

  parser.add_argument('--nrShiftsIt', dest='nrShiftsIt', type=int, default=30,
                     help='# of inner iterations to run, for fitting the model parameters and subj. shifts')

  parser.add_argument('--mem', dest='mem', type=int, default=15,
                     help='memory limit of process in GB')

  parser.add_argument('--agg', dest='agg', type=int, default=0,
                     help='plot figures without using Xwindows, for use on cluster, not linked to cluster as I need to test it locally first')

  parser.add_argument('--serial', action="store_true",
                     help='runs jobs serially instead of spawning processes on local machine or on cluster')

def initCommonLaunchParams(args):
  hostName = gethostname()
  print(hostName)

  launchParams = {}
  if args.cluster:
    launchParams['homeDir'] = '/home/rmarines'
  elif hostName == 'razvan-Precision-T1700' or hostname =='razvan-Inspiron-5547':
    launchParams['homeDir'] = '/home/razvan'
  else:
    raise ValueError('hostname wrong or forgot --cluster flag')

  launchParams['REPO_DIR'] = '%s/phd_proj/jointModellingDisease' % launchParams['homeDir']
  launchParams['OUTPUT_DIR'] = '%s/clusterOutput' % launchParams['REPO_DIR']

  if args.cluster:
    launchParams['WALL_TIME_LIMIT_HOURS'] = int(args.timeLimit)
    launchParams['WALL_TIME_LIMIT_MIN'] = 15
    # MEM_LIMIT in GB
    launchParams['MEM_LIMIT'] = args.mem


  print('informPrior', args.informPrior)

  return launchParams

def getQsubCmdPart2(launchParams, jobName):
  qsubCmd = 'qsub -S /bin/bash -l tmem=%.1fG -l h_vmem=%.1fG -l ' \
            'h_rt=%d:%d:0 %s -N %s -j y -wd %s %s' % (launchParams['MEM_LIMIT'],
  launchParams['MEM_LIMIT'], launchParams['WALL_TIME_LIMIT_HOURS'],
  launchParams['WALL_TIME_LIMIT_MIN'],
  launchParams['tscratchStr'], jobName,
  launchParams['OUTPUT_DIR'], launchParams['reserveStr'])

  return qsubCmd
