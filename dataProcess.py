# Script for reading Brenden's file and convert to tasks with length tags

import argparse
import pandas as pd
import os

def readTaskFile(filename):
    df = pd.read_csv(filename, sep = ': ', names = ['_', 'Input', 'Output'], engine='python')
    df = df.drop('_', axis = 1)
    df['Input'] = df['Input'].map(lambda x: x[0:-4])
    return df

def readLengthFile(filename):
    df = pd.read_csv(filename, sep = ' ::: ', names = ['Input', 'Output', 'Length'], engine='python')
    df['Input'] = df['Input'].map(lambda x: x[4:])
    return df

def getLengthTag(dfTask, dfLength):
    dfTask2 = pd.merge(dfTask, dfLength[['Input', 'Length']], on = 'Input')
    if len(dfTask2) == len(dfTask):
        return dfTask2
    else:
        print('%d records not matched'%(len(dfTask) - len(dfTask2)))




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--filePath') # Path of files to be converted
    parser.add_argument('--lengthFileName') # Master file with length tags
    parser.add_argument('--outPath')  # output folder

    args = parser.parse_args()
    dfLength = readLengthFile(args.lengthFileName)
    n = len(args.filePath) # To be used to separate root path and subdirectory below

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
EOS_token = 1
SOS_token = 0

def _build_vocab(filename):
  data = _read_words(filename)

  counter = collections.Counter(data)
  count_pairs = sorted(counter.items(), key=lambda x: (-x[1], x[0]))

  words, _ = list(zip(*count_pairs))
  word_to_id = dict(zip(words, range(len(words))))

  return word_to_id


def _file_to_word_ids(filename, word_to_id):
  data = _read_words(filename)
  return [word_to_id[word] for word in data if word in word_to_id]

df = readLengthFile('tasks_with_length_tags.txt')
df2 = readTaskFile('tasks_test_simple.txt')

df3 = getLengthTag(df2, df)

df3 = pd.merge(df2, df[['Input', 'Length']], on = 'Input')

"""
