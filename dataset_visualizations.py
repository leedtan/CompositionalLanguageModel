"""
Script of training model V1
"""

import modelV1_tf as m1
import argparse
import pickle

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument(
      "--trainFile", default='data/length_split/tasks_train_length.txt')
  parser.add_argument(
      "--testFile", default='data/length_split/tasks_test_length.txt')
  parser.add_argument("--mapFile", default='data/maps.p')
  parser.add_argument("--outputPath")
  parser.add_argument("--batchSize", type=int, default=32)
  parser.add_argument("--nIter", type=int, default=100000)
  parser.add_argument("--seed", type=int, default=100)
  parser.add_argument("--testIter", type=int, default=500)
  parser.add_argument("--flgSave", action='store_true')

  parser.add_argument("--hidden_filters", type=int, default=128)
  parser.add_argument("--hidden_filters_subprogram", type=int, default=128)
  parser.add_argument("--num_layers_encoder", type=int, default=2)
  parser.add_argument("--num_layers_subprogram", type=int, default=2)
  parser.add_argument("--size_emb", type=int, default=64)
  parser.add_argument("--init_mag", type=float, default=1e-3)
  parser.add_argument("--l2_lambda", type=float, default=1e-3)
  parser.add_argument("--lrInit", type=float, default=0.1)

  args = parser.parse_args()

  max_cmd_len = 10
  max_actions_per_subprogram = 9
  max_num_subprograms = 7
  num_cmd = 14
  num_act = 9
  #--nIter 500 --testIter 50 --num_layers_encoder 2
  # --num_layers_subprogram 2 --flgSave --trainFile
  # /Users/jshliu/Google\ Drive/NYUClass/DSGA_3001_Cognitive/project_local/data/length_split/tasks_train_length.txt
  # --testFile /Users/jshliu/Google\ Drive/NYUClass/DSGA_3001_Cognitive/project_local/data/length_split/tasks_test_length.txt
  #  --mapFile /Users/jshliu/Google\ Drive/NYUClass/DSGA_3001_Cognitive/project_local/data/maps.p
  # --outputPath /Users/jshliu/Google\ Drive/NYUClass/DSGA_3001_Cognitive/project_local/code/run/length_split/ > log_length_split.txt

  train_paras = {
      'batchSize': args.batchSize,
      'nIter': args.nIter,
      'seed': args.seed,
      'testIter': args.testIter,
      'flgSave': args.flgSave,
      'savePath': args.outputPath,
      'lrInit': args.lrInit
  }
  model_paras = {
      'hidden_filters': args.hidden_filters,
      'num_layers_encoder': args.num_layers_encoder,
      'size_emb': args.size_emb,
      'num_cmd': num_cmd,
      'num_act': num_act,
      'init_mag': args.init_mag,
      'max_cmd_len': max_cmd_len,
      'max_num_subprograms': max_num_subprograms,
      'max_actions_per_subprogram': max_actions_per_subprogram,
      'l2_lambda': args.l2_lambda,
      'hidden_filters_subprogram': args.hidden_filters_subprogram,
      'num_layers_subprogram': args.num_layers_subprogram,
  }

  print("Loading Data")
  command_map, _, action_map, _ = pickle.load(open(args.mapFile, 'rb'))

  trainset = m1.DataSet(
      args.trainFile,
      command_map,
      action_map,
      max_cmd_len,
      max_actions_per_subprogram,
      max_num_subprograms,
      delimiter=':::',
      seed=100)
  testset = m1.DataSet(
      args.testFile,
      command_map,
      action_map,
      max_cmd_len,
      max_actions_per_subprogram,
      max_num_subprograms,
      delimiter=':::',
      seed=100)
  print('Length of training set: ', trainset._dataSize)
  print('Length of test set: ', testset._dataSize)

  print('Model parameters: ', model_paras)
  import numpy as np
  import matplotlib.pyplot as plt

  def bar(data, name):
    bincounts = np.bincount(data)
    plt.bar(np.arange(data.max() + 1), bincounts)
    plt.savefig(name + '.jpg')
    plt.close()

  testsublengths = testset.struct_np.flatten()
  testsublengths = testsublengths[np.nonzero(testsublengths)]
  bar(testsublengths, 'sublengths_test')

  val_lengths = np.argmin(testset.struct_np > 0, 1)
  val_lengths[val_lengths == 0] = val_lengths.max() + 1
  bar(val_lengths, 'seq_lengths_test')

  trn_sublengths = trainset.struct_np.flatten()
  trn_sublengths = trn_sublengths[np.nonzero(trn_sublengths)]
  bar(trn_sublengths, 'sublengths_train')

  trn_lengths = np.argmin(trainset.struct_np > 0, 1)
  trn_lengths[trn_lengths == 0] = 7
  bar(trn_lengths, 'seq_lengths_train')

  model = m1.m1(model_paras)

  print('Training parameters: ', train_paras)
  trainModel = m1.trainModel(model, train_paras, trainset, testset)
  modelResult, lsTrainAcc, lsTestAcc = trainModel.run()
  pickle.dump([lsTrainAcc, lsTestAcc], open(args.outputPath + 'acc.p', 'wb'))

  print('Best test accuracy: ', max(lsTestAcc))
