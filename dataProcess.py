# Script for reading Brenden's file and convert to tasks with length tags

import argparse
import pandas as pd
import os
import numpy as np
import pickle

def readTaskFile(filename):
    df = pd.read_csv(filename, sep = ': ', names = ['_', 'Input', 'Output'], engine='python')
    df = df.drop('_', axis = 1)
    df['Input'] = df['Input'].map(lambda x: x[0:-4])
    return df

def readLengthFile(filename):
    df = pd.read_csv(filename, sep = ' ::: ', names = ['Input', 'Output', 'Length'], engine='python')
    df['Input'] = df['Input'].map(lambda x: x[4:])
    uni_commands = set(' '.join(list(df['Input'])).split())
    uni_actions = set(' '.join(list(df['Output'])).split())
    return df, uni_commands, uni_actions

def getLengthTag(dfTask, dfLength):
    dfTask2 = pd.merge(dfTask, dfLength[['Input', 'Length']], on = 'Input')
    if len(dfTask2) == len(dfTask):
        return dfTask2
    else:
        print('%d records not matched'%(len(dfTask) - len(dfTask2)))

def build_fmap_invmap(unique, num_unique):
    fmap = dict(zip(unique, np.arange(num_unique) )) # Use 0 for padding
    invmap = dict(zip(np.arange(num_unique), unique))
    return fmap, invmap

def buildDict(uni_commands, uni_actions):
    uni_commands.add('end_command')
    uni_actions.add('end_subprogram')
    uni_actions.add('start_subprogram')
    uni_actions.add('end_action')
    uni_actions.add('start_action')
    num_cmd = len(uni_commands)
    num_act = len(uni_actions)
    command_map, command_invmap = build_fmap_invmap(uni_commands, num_cmd)
    action_map, action_invmap = build_fmap_invmap(uni_actions, num_act)
    return command_map, command_invmap, action_map, action_invmap


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--filePath') # Path of files to be converted
    parser.add_argument('--lengthFileName') # Master file with length tags
    parser.add_argument('--outPath')  # output folder

    args = parser.parse_args()
    dfLength, uni_commands, uni_actions = readLengthFile(args.lengthFileName)
    command_map, command_invmap, action_map, action_invmap = buildDict(uni_commands, uni_actions)
    pickle.dump([command_map, command_invmap, action_map, action_invmap], open(args.outPath + 'maps.p', 'wb'))
    n = len(args.filePath) # To be used to separate root path and subdirectory below

    """
    for dirName, subdirList, fileList in os.walk(args.filePath):
        #print('entries: %s, %s, %s'%(dirName, subdirList, fileList))
        for file in fileList:
            if file[-4:] == '.txt':
                inputPath = os.path.join(dirName, file)
                print(inputPath)
                dfTask = readTaskFile(inputPath)
                dfTask2 = getLengthTag(dfTask, dfLength)
                if dfTask2 is not None:
                    outPath = os.path.join(args.outPath, dirName[n:]) # Subdirectory name
                    if not os.path.isdir(outPath):
                        os.makedirs(outPath)
                    outFile = os.path.join(args.outPath, inputPath[n:])
                    with open(outFile, 'w') as f:
                        for _, row in dfTask2.iterrows():
                            f.write("::: " + row['Input'] + " ::: " + row['Output'] + " ::: " + row['Length'] + '\n')
    """





