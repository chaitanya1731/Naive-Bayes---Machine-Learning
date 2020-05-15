import collections
import os
import re
import sys
import copy
import math
from document import Document

trainDataset = {}
testDataset = {}

classes = ["ham", "spam"]
conditionalProb = {}
conditionalProbWithoutStopwords = {}
prior = {}
priorWithoutStopwords = {}


def getDataset(storage_dict, directory, true_class):
    for file in os.listdir(directory):
        filePath = os.path.join(directory, file)
        if os.path.isfile(filePath):
            with open(filePath, encoding='cp437') as text_file:
                text = text_file.read()
                storage_dict.update({filePath: Document(text, getWordsCollection(text), true_class)})


def getWordsCollection(text):
    wordsCollection = collections.Counter(re.findall(r'\w+', text))
    return dict(wordsCollection)


def getStopwords():
    words = []
    with open('stopwords.txt', 'r') as txt:
        words = (txt.read().splitlines())
    return words


def getFilteredDataset(stopwords, dataset):
    datasetWithoutStopwords = copy.deepcopy(dataset)
    for word in stopwords:
        for j in datasetWithoutStopwords:
            if word in datasetWithoutStopwords[j].getWordFrequency():
                del datasetWithoutStopwords[j].getWordFrequency()[word]
    return datasetWithoutStopwords


def getVocabulary(dataset):
    wordsInDataset = ""
    vocabulary = []
    for x in dataset:
        wordsInDataset += dataset[x].getText()
    for y in getWordsCollection(wordsInDataset):
        vocabulary.append(y)
    return vocabulary


def trainNaiveBayes(trainingDataset, priors, cond):
    vocab = getVocabulary(trainingDataset)
    n = len(trainingDataset)
    for c in classes:
        n_c = 0.0
        text_c = ""
        for i in trainingDataset:
            if trainingDataset[i].getTrueClass() == c:
                n_c += 1
                text_c += trainingDataset[i].getText()
        priors[c] = float(n_c) / float(n)

        wordCount = getWordsCollection(text_c)
        for t in vocab:
            if t in wordCount:
                cond.update({t + "_" + c: (float((wordCount[t] + 1.0)) / float((len(text_c) + len(wordCount))))})
            else:
                cond.update({t + "_" + c: (float(1.0) / float((len(text_c) + len(wordCount))))})


def applyNaiveBayes(testDataset, priors, cond):
    score = {}
    for c in classes:
        score[c] = math.log10(float(priors[c]))
        for t in testDataset.getWordFrequency():
            if (t + "_" + c) in cond:
                score[c] += float(math.log10(cond[t + "_" + c]))
    if score["spam"] > score["ham"]:
        return "spam"
    else:
        return "ham"


if __name__ == '__main__':
    if len(sys.argv) != 4:
        print('Please enter correct arguments')
        sys.exit()
    trainSpamPath, trainHamPath = sys.argv[1] + "/spam/", sys.argv[1] + "/ham/"
    testSpamPath, testHamPath = sys.argv[2] + "/spam/", sys.argv[2] + "/ham/"

    getDataset(trainDataset, trainSpamPath, classes[1])
    getDataset(trainDataset, trainHamPath, classes[0])

    getDataset(testDataset, testSpamPath, classes[1])
    getDataset(testDataset, testHamPath, classes[0])

    stopwords = getStopwords()

    trainDatasetWithoutStopwords = getFilteredDataset(stopwords, trainDataset)
    testDatasetWithoutStopwords = getFilteredDataset(stopwords, testDataset)

    trainNaiveBayes(trainDataset, prior, conditionalProb)
    trainNaiveBayes(trainDatasetWithoutStopwords, priorWithoutStopwords, conditionalProbWithoutStopwords)

    CorrectGuessTrainData = 0
    for i in trainDataset:
        trainDataset[i].setLearnedClass(applyNaiveBayes(trainDataset[i], prior, conditionalProb))
        if trainDataset[i].getLearnedClass() == trainDataset[i].getTrueClass():
            CorrectGuessTrainData += 1

    CorrectGuessTestData = 0
    for i in testDataset:
        testDataset[i].setLearnedClass(applyNaiveBayes(testDataset[i], prior, conditionalProb))
        if testDataset[i].getLearnedClass() == testDataset[i].getTrueClass():
            CorrectGuessTestData += 1

    CorrectGuessTestWithoutStopwords = 0
    for i in testDatasetWithoutStopwords:
        testDatasetWithoutStopwords[i].setLearnedClass(applyNaiveBayes(testDatasetWithoutStopwords[i],
                                                                       priorWithoutStopwords, conditionalProbWithoutStopwords))
        if testDatasetWithoutStopwords[i].getLearnedClass() == testDatasetWithoutStopwords[i].getTrueClass():
            CorrectGuessTestWithoutStopwords += 1

    trainAccuracy = (100.0 * float(CorrectGuessTrainData) / float(len(trainDataset)))
    testAccuracyWithStopwords = (100.0 * float(CorrectGuessTestData) / float(len(testDataset)))
    testAccuracyWithoutStopwords = (100.0 * float(CorrectGuessTestWithoutStopwords)
                                    / float(len(testDatasetWithoutStopwords)))

    print("Training \t Test_with_stopwords \t Test_without_stopwords")
    print('%f' % trainAccuracy + '\t %f' % testAccuracyWithStopwords +
          '\t\t %f' % testAccuracyWithoutStopwords)

    output = open("Accuracy.txt", "w")
    output.write('Training \t Test_with_stopwords \t Test_without_stopwords')
    output.write("\n")
    output.write('%f' % trainAccuracy + '\t %f' % testAccuracyWithStopwords +
                 '\t\t %f' % testAccuracyWithoutStopwords)
    output.write("\n")
    output.close()
