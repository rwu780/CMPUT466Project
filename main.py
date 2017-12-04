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
                 'Logistic Regression L2 regularizer': al.LogisticRegressionClassifier({'regularizer':'l2', 'regularizerValue':0.01}),
                 'Logistic Regression No regularizer': al.LogisticRegressionClassifier(),
                 'Neural Network': al.NeuralNetwork(),
                }

    numalgs = len(classalgs)

    parameters = (
        {'regwgt':0.0, 'nh':(50, ),  'regularizerValue': 0.01 },
        {'regwgt':0.0, 'nh':(100, ),  'regularizerValue': 0.1 },
        {'regwgt':0.0, 'nh':(300,),  'regularizerValue': 1 },
        #{'regwgt':0.0, 'nh':(500,), },
        #{'regwgt':0.0, 'nh':(800,), },
        )
    
    numparams = len(parameters)
    
    accuracy = {}
    runningTime = {}
    for learnername in classalgs:
        accuracy[learnername] = np.zeros(numparams)
        runningTime[learnername] = np.zeros(numparams)
    
    print("========== Perform Hold-out test==========")
    
    trainset, testset = dataloader.splitTrainAndTestData()
    print(('Running on train={0} and test = {1} samples').format(trainset[0].shape[0], testset[0].shape[0]))

    for p in range(numparams):
        params = parameters[p]
        for learnername, learner in classalgs.items():
            learner.reset(params)
            print("Running on " + learnername + ' on parameters ' + str(learner.getparams()))

            start = time.time()

            learner.learn(trainset[0], trainset[1])
            predictions = learner.predict(testset[0])

            stop = time.time()

            ac = getaccuracy(testset[1], predictions)
            
            accuracy[learnername][p] = ac
            runningTime[learnername][p] = stop-start
            print("Accuracy: " + str(ac))
            print("Time: " + str(runningTime[learnername][p]))

    print("========== Result for Hold-out Test ==========")

    for learnername, learner in classalgs.items():
        bestAccuracy = np.mean(accuracy[learnername][0])
        bestparams = 0
        runTime = np.mean(runningTime[learnername][0])

        for p in range(numparams):
            aveAccuracy = np.mean(accuracy[learnername][p])
            if bestAccuracy < aveAccuracy:
                bestAccuracy = aveAccuracy
                bestparams = p
                runTime = np.mean(runningTime[learnername][p])

        learner.reset(parameters[bestparams])
        print('Best parameters for ' + learnername + ': ' + str(learner.getparams()))
        print('Accuracy for ' + learnername + ': ' + str(bestAccuracy) + '%')
        print('Time to run for ' + learnername + ': ' + str(runTime) + ' sec')
        print('----------')
    
    print("========== K-fold cross validation ==========")
    accuracy = {}
    runningTime = {}
    for learnername in classalgs:
        accuracy[learnername] = np.zeros((numparams, 10))
        runningTime[learnername] = np.zeros(numparams)

    feature, target = dataloader.splitInputOutputData()

    for p in range(numparams):
        params = parameters[p]
        for learnername, learner in classalgs.items():

            # No need to run random again
            if learnername is 'Random':
                continue

            # Take too long to run, only run once
            if learnername is 'Logistic Regression L2 regularizer':
                if p == 0:
                    params = parameters[0]
                else:
                    continue


            learner.reset(params)
            print("Running on " + learnername + "with parameters: " + str(learner.getparams()))

            start = time.time()
            
            a = learner.getAlg()
            scores = cross_val_score(a, feature, target, cv = 10)

            stop = time.time()

            accuracy[learnername][p] = scores * 100
            std = np.std(accuracy[learnername][p])

            print("Accuracy: " + str(np.mean(accuracy[learnername][p])) + '+- ' + str(std))
            runningTime[learnername][p] = stop - start
            print("Time: " + str(runningTime[learnername][p]))

    print("========== Result ==========")
    for learnername, learner in classalgs.items():
        bestAccuracy = np.mean(accuracy[learnername][0])
        bestparams = 0
        runTime = np.mean(runningTime[learnername][0])

        # No need to display random again
        if learnername is 'Random':
            continue

        for p in range(numparams):
            aveAccuracy = np.mean(accuracy[learnername][p])
            if bestAccuracy < aveAccuracy:
                bestAccuracy = aveAccuracy
                bestparams = p
                runTime = np.mean(runningTime[learnername][p])

            #std = scores.std() ** 2
        learner.reset(parameters[bestparams])
        std = np.std(accuracy[learnername][bestparams])

        print("Best parameters " + learnername + ":" + str(learner.getparams()))
        print("Accuracy for " + learnername + ": " + str(bestAccuracy) + '% +- ' + str(std))
        print('Time to run for ' + learnername + ': ' + str(runTime) + ' sec')
        print('-----------')

