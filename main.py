import numpy as np
import loadData as ld
import algorithms as al
import math

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
    dataloader = ld.DataLoader()
    numruns = 1

    # Follw the same testing format as in Assignments
    classalgs = {'Random': al.Classifier(),
                 'Naive Bayes': al.NaiveBayes(),
                 'Logistic Regression L2 regularizer': al.LogisticRegressionClassifier({'regularizer':'l2'}),
                 'Logistic Regression No regularizer': al.LogisticRegressionClassifier(),
                 'Neural Network': al.NeuralNetwork(),
                }

    numalgs = len(classalgs)

    parameters = (
        {'regwgt': 0.0, 'nh' : (300,1)},
        {'regwgt': 0.01, 'nh' : (300,2)},
        {'regwgt': 0.05, 'nh' : (100, 1)},
        {'regwgt':0.1, 'nh': (100, 2)}
        )

    numparams = len(parameters)

    errors = {}
    for learnername in classalgs:
        errors[learnername] = np.zeros((numparams, numruns))

    for r in range(numruns):
        trainset, testset = dataloader.loadData(dataFile)

        print(('Running on train={0} and test = {1} samples for run {2}').format(trainset[0].shape[0], testset[0].shape[0], r))

        for p in range(numparams):
            params = parameters[p]
            for learnername, learner in classalgs.items():
                learner.reset(params)
                print('Running learner = '+learnername+' on parameters ' + str(learner.getparams()))

                learner.learn(trainset[0], trainset[1])
                predictions = learner.predict(testset[0])
                error = geterror(testset[1], predictions)
                print('Error for ' + learnername + ': ' + str(error))
                errors[learnername][p, r] = error

    for learnername, learner in classalgs.items():
        besterror = np.mean(errors[learnername][0,:])
        bestparams = 0
        for p in range(numparams):
            aveerror = np.mean(errors[learnername][p,:])
            if aveerror < besterror:
                besterror = aveerror
                bestparams = p

        learner.reset(parameters[bestparams])
        print('Best parameters for ' + learnername + ': ' + str(learner.getparams()))
        print('Average error for ' + learnername + ': ' + str(besterror) + ' +- ' + str(np.std(errors[learnername][bestparams,:])/math.sqrt(numruns)))
