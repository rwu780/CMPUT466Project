import numpy as np
import loadData as ld
import algorithms as al
import time
from sklearn.model_selection import cross_val_score

def getaccuracy(ytest, predictions):
    correct = 0
    for i in range(len(ytest)):
        if ytest[i] == predictions[i]:
            correct += 1

    return (correct/float(len(ytest)))*100.0


def geterror(ytest, predictions):
    return (100.0 - getaccuracy(ytest, predictions))

if __name__ == '__main__':

    dataFile = 'dataset/bank.csv'
    dataloader = ld.DataLoader(dataFile)
    numruns = 1

    # Follw the same testing format as in Assignments
    classalgs = {'Random': al.Classifier(), # Baseline Algorithm
                 'Linear SVM': al.SVMClassifier(), # Linear SVM
                 'Logistic Regression L2 regularizer': al.LogisticRegressionClassifier({'regularizer':'l2'}),
                 'Logistic Regression No regularizer': al.LogisticRegressionClassifier(),
                 'Neural Network': al.NeuralNetwork(),
                }

    numalgs = len(classalgs)

    parameters = ({'regwgt':0.0, 'nh':(300,1)},)

    accuracy = {}
    runningTime = {}
    for learnername in classalgs:
        accuracy[learnername] = 0
        runningTime[learnername] = 0

    print("========== Perform Hold-out test==========")
    
    trainset, testset = dataloader.splitTrainAndTestData()
    print(('Running on train={0} and test = {1} samples').format(trainset[0].shape[0], testset[0].shape[0]))

    for learnername, learner in classalgs.items():
        learner.reset(parameters[0])
        print("Running on " + learnername)
        start = time.time()

        learner.learn(trainset[0], trainset[1])
        predictions = learner.predict(testset[0])

        stop = time.time()

        ac = getaccuracy(testset[1], predictions)
        accuracy[learnername] = ac
        runningTime[learnername] = stop-start

    print("========== Result ==========")
    for learnername in classalgs:
        print('Accuracy for ' + learnername + ': ' + str(accuracy[learnername]) + '%')
        print('Time to run for ' + learnername + ': ' + str(runningTime[learnername]) + ' sec')
        print('----------')


    print("========== K-fold cross validation ==========")
    accuracy = {}
    runningTime = {}
    for learnername in classalgs:
        accuracy[learnername] = 0
        runningTime[learnername] = 0

    feature, target = dataloader.splitInputOutputData()

    for learnername, learner in classalgs.items():

        # No need to run random again
        if learnername is 'Random':
            continue

        learner.reset(parameters[0])
        print("Running on " + learnername)

        start = time.time()
        
        a = learner.getAlg()
        scores = cross_val_score(a, feature, target, cv = 10)

        stop = time.time()

        accuracy[learnername] = scores * 100
        runningTime[learnername] = stop - start

    print("========== Result ==========")
    for learnername in classalgs:

        # No need to display random again
        if learnername is 'Random':
            continue

        mean = np.mean(accuracy[learnername])
        std = scores.std() ** 2
        print("Accuracy for " + learnername + ": " + str(mean) + '% +- ' + str(std))
        print('Time to run for ' + learnername + ': ' + str(runningTime[learnername]) + ' sec')
        print('-----------')

