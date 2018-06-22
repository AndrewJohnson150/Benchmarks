# Metamorphic testing
# Relations and their explanations may be found at: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3019603/

import p1b2_baseline_keras2
import p1b2

import keras
import numpy as np
import copy

import unittest

class p1b2Tests(unittest.TestCase):

    # Note: test cases only run when they start with 'test'

    @classmethod
    def setUpClass(self):
        self._srcModel = p1b2_baseline_keras2.main(DeterministicResults=True)
        (self.X_train, self.y_train), (self.X_test, self.y_test) = p1b2.load_data()
        self._origPredictions = self._srcModel.predict_classes(self.X_test)

    def differencesInAccuracy(self):
        #In order for this test to work, you must change the setUp to predict() in place of predict_classes()
        (X_train, y_train), (X_test, y_test) = self.__getCopiesOfData()
        newPreds = p1b2_baseline_keras2.main(X_train,y_train,X_test,y_test,True).predict(X_test)
        scores = p1b2.evaluate(self._origPredictions, y_test)
        print('old model on test data:', scores)
        newScores = p1b2.evaluate(newPreds, y_test)
        print('new model on test data:', newScores)
        diff = scores['accuracy'] - newScores['accuracy'] 
        print('difference in scores:', diff)

    def modelIsDeterministic(self):
        newPreds = p1b2_baseline_keras2.main(DeterministicResults=True).predict_classes(self.X_test)
        assert np.array_equal(self._origPredictions,newPreds), "Model is not deterministic"

    def test_MR0_ConsistenceWithAffineTransform(self):
        (X_train, y_train), (X_test, y_test) = self.__getCopiesOfData()
        numFeatures = X_train.shape[1]
        
        #Makes an array of random size filled with random numbers between 0 and numFeatures without repitition
        randomSubset = np.random.choice(range(numFeatures),np.random.randint(numFeatures),False)

        k = np.random.randint(1,100)
        b = np.random.randint(100)

        for i in range(randomSubset.shape[0]):
            c = randomSubset[i]
            X_train[:,c] = X_train[:,c]*k + b
            X_test[:,c] = X_test[:,c]*k + b

        transformedPredictions = p1b2_baseline_keras2.main(X_train,y_train,X_test,y_test,True).predict_classes(X_test)

        assert np.array_equal(self._origPredictions,transformedPredictions), "affine transformations change the outcome of the model"

    def test_MR11_PermutationOfClassLabels(self):
        (X_train, y_train), (X_test, y_test) = self.__getCopiesOfData()

        #print('pre permute train:')
        #print(y_train[50:55,:])
        #print('pre permute test:')
        #print(y_test[50:55,:])
        p = self.__permuteLabels(y_train,y_test)
        #print('permutation: ', p)
        #print('new train:')
        #print(y_train[50:55,:])
        #print('new test:')
        #print(y_test[50:55,:])

        newModel = p1b2_baseline_keras2.main(X_train,y_train,X_test,y_test,True)
        newPredictions = newModel.predict_classes(X_test)

        for x in range(X_test.shape[1]):
            assert newPredictions[x] == p[self._origPredictions[x]], "permuting class labels changes outcome of model"
            #print(p)
            #print(self._origPredictions[x])
            #print(newPredictions[x])
            #print(p[self._origPredictions[x]])
        
    def test_MR12_PermutationOfAttributes(self):
        (X_train, y_train), (X_test, y_test) = self.__getCopiesOfData()

        X_train, X_test = self.__shuffleColumnsInUnison(X_train,X_test)

        shuffledModelPredictions = p1b2_baseline_keras2.main(X_train,y_train,X_test,y_test,True).predict_classes(X_test)
        #for x in range(X_test.shape[0]):
        #     if not (self._origPredictions[x]==shuffledModelPredictions[x]):
        #         print('mismatch:')
        #         print(self._origPredictions[x])
        #         print(shuffledModelPredictions[x])
        assert np.array_equal(self._origPredictions,shuffledModelPredictions), "permuting the order of the features changes the outcome of the model"
    
    def test_MR21_AddUninformativeAttribute(self):
        (X_train, y_train), (X_test, y_test) = self.__getCopiesOfData()
        tempTrain = np.zeros((X_train.shape[0],X_train.shape[1]+1))
        tempTest = np.zeros((X_test.shape[0],X_test.shape[1]+1))
        tempTrain[:,:-1] = X_train
        tempTest[:,:-1] = X_test
        newModel = p1b2_baseline_keras2.main(tempTrain,y_train,tempTest,y_test,True)
        newPreds = newModel.predict_classes(tempTest)
        assert np.array_equal(self._origPredictions,newPreds), "adding an uninformative attribute changes outcome"
   
    def test_MR22_AddInformativeAttribute(self):
        (X_train, y_train), (X_test, y_test) = self.__getCopiesOfData()
        tempTrain = np.zeros((X_train.shape[0],X_train.shape[1]+1))
        tempTest = np.zeros((X_test.shape[0],X_test.shape[1]+1))
        tempTrain[:,:-1] = X_train
        tempTest[:,:-1] = X_test
        
        #pick a random class
        n = np.random.randint(10)

        #if a test point is associated with class n, make the new attribute 1
        for x in range(X_train.shape[0]):
            if X_train[x,n] > .5:
                tempTrain[x,-1] = 1
        for x in range(X_test.shape[0]):
            if X_test[x,n] > .5:
                tempTest[x,-1] = 1

        newModel = p1b2_baseline_keras2.main(tempTrain,y_train,tempTest,y_test,True)
        newPreds = newModel.predict_classes(tempTest)
        
        for x in range(X_test.shape[0]):
            if (self._origPredictions[x] == n):
                assert newPreds[x] == n, "adding an informative feature for class n changed previous classifications of n to another class"
   
    def test_MR31_ConsistenceWithRePrediction(self):
        (X_train, y_train), (X_test, y_test) = self.__getCopiesOfData()
        n = np.random.randint(X_test.shape[0])
        nPred = self._origPredictions[n]

        newXTrain = np.zeros((X_train.shape[0]+1,X_train.shape[1]))
        newXTrain[:-1,:] = X_train
        newXTrain[-1,:] = X_test[n,:]
        newYTrain = np.zeros((y_train.shape[0]+1,y_train.shape[1]))
        newYTrain[:-1,:] = y_train
        newYTrain[-1,:] = y_test[n,:]

        newModel = p1b2_baseline_keras2.main(newXTrain,newYTrain,X_test,y_test,True)
        
        assert (newModel.predict_classes(X_test)[n]) == nPred
        
   
    def test_MR32_AdditionalTrainingSample(self):
        (X_train, y_train), (X_test, y_test) = self.__getCopiesOfData()
        n = np.random.randint(X_test.shape[1])

        count = int(np.sum(X_train[:,n]))

        newXTrain = np.zeros((X_train.shape[0]+count,X_train.shape[1]))
        newYTrain = np.zeros((y_train.shape[0]+count,y_train.shape[1]))
        
        newXTrain[:X_train.shape[0],:] = X_train
        newYTrain[:y_train.shape[0],:] = y_train
        
        XCount = count
        for x in range(count):
            if np.argmax(y_train[x,:]) == n:
                newXTrain[XCount,:] = X_train[x,:]
                newYTrain[XCount,:] = y_train[x,:]
                XCount = XCount + 1

        newModel = p1b2_baseline_keras2.main(newXTrain,newYTrain,X_test,y_test,True)
        newPreds = newModel.predict_classes(X_test)

        for x in range(X_test.shape[0]):
            if (self._origPredictions[x] == n):
                assert newPreds[x] == n, "doubling the training samples for class n changed some classifications from n to another class"

    
        
   
    def test_MR41_AddClassByDuplicatingSamples(self):
        (X_train, y_train), (X_test, y_test) = self.__getCopiesOfData()
        cl = np.random.randint(y_train.shape[1])
        
        count = int(np.sum(y_train[:,cl]))

        #One class will remain empty but this should have no impact on the result
        newNumberOfClasses = (y_train.shape[1] * 2)

        newXTrain = np.zeros((X_train.shape[0]*2 - count, X_train.shape[1]))
        newYTrain = np.zeros((y_train.shape[0]*2 - count, newNumberOfClasses))
        newYTest = np.zeros((y_test.shape[0], newNumberOfClasses))
        newXTrain[:X_train.shape[0],:] = X_train
        newYTrain[:y_train.shape[0],:y_train.shape[1]] = y_train
        newYTest[:y_test.shape[0],:y_test.shape[1]] = y_test
        
        count = y_train.shape[0]
        for x in range(y_train.shape[0]):
            if np.argmax(y_train[x,:]) != cl:
                newYTrain[count,np.argmax(y_train[x,:])+y_train.shape[1]] = 1 
                newXTrain[count,:] = X_train[x,:]
                count = count + 1

        newModel = p1b2_baseline_keras2.main(newXTrain,newYTrain,X_test,newYTest,True)
        newPreds = newModel.predict_classes(X_test)

        for x in range(X_test.shape[0]):
            if (self._origPredictions[x] == cl):
                assert newPreds[x] == cl, "adding new classes by doubling the training samples for classes other than n made our classifier worse for class n"



    def test_MR42_AddClassesByReLabelingSamples(self):
        (X_train, y_train), (X_test, y_test) = self.__getCopiesOfData()
        cl = np.random.randint(y_train.shape[1])

        newNumberOfClasses = y_train.shape[1] * 2

        newYTrain = np.zeros((y_train.shape[0],newNumberOfClasses))

        #y_test will need the same shape
        newYTest = np.zeros((y_test.shape[0],newNumberOfClasses))
        newYTest[:,:y_test.shape[1]] = y_test

        for x in range(y_train.shape[0]):
            if y_train[x,cl] == 1:
                newYTrain[x,cl] = 1
            else:
                if np.random.random() < .5:
                    newYTrain[x, np.argmax(y_train[x,:]) + y_train.shape[1]] = 1
                else:
                    newYTrain[x,np.argmax(y_train[x,:])] = 1
            

        newModel = p1b2_baseline_keras2.main(X_train,newYTrain,X_test,newYTest,True)
        newPreds = newModel.predict_classes(X_test)

        for x in range(X_test.shape[0]):
            if (self._origPredictions[x] == cl):
                assert newPreds[x] == cl, "relabeling the class of training samples for classes besides n changed some classifications of n"

   
    def test_MR51_RemovalOfClasses(self):
        (X_train, y_train), (X_test, y_test) = self.__getCopiesOfData()
        #Class to remove
        cl = np.random.randint(y_test.shape[1])

        trainCount = int(np.sum(y_train[:,cl]))
        testCount = int(np.sum(y_test[:,cl]))

        newXTrain = np.zeros((X_train.shape[0] - trainCount, X_train.shape[1]))
        newYTrain = np.zeros((y_train.shape[0] - trainCount, y_train.shape[1] - 1))
        newXTest = np.zeros((X_test.shape[0] - testCount, X_test.shape[1]))
        newYTest = np.zeros((y_test.shape[0] - testCount, y_test.shape[1] - 1))

        count = 0
        for x in range(X_train.shape[0]):
            if y_train[x,cl] > .5:
                pass
            else:
                newXTrain[count,:] = X_train[x,:]
                newYTrain[count,:] = np.delete(y_train[x,:],cl,0)
                count = count + 1

        count = 0
        for x in range(X_test.shape[0]):
            if not y_test[x,cl] > .5:
                newXTest[count,:] = X_test[x,:]
                newYTest[count,:] = np.delete(y_test[x,:],cl,0)
                count = count + 1
        newModel = p1b2_baseline_keras2.main(newXTrain,newYTrain,newXTest,newYTest,True)
        newPreds = newModel.predict_classes(X_test)

        count = 0
        for x in range(X_test.shape[0]):
            if self._origPredictions[x] != cl:
                assert self._origPredictions[x] == newPreds[count], "removing a class causes the model to change predictions for the remaining classes"
                count = count + 1



    def test_MR52_RemovalOfSamples(self):
        (X_train, y_train), (X_test, y_test) = self.__getCopiesOfData()
        cl = np.random.randint(y_test.shape[1])

        samplesToKeep = np.zeros(X_train.shape[0])

        for x in range(X_train.shape[0]):
            if (np.random.random() < .5) or y_train[x,cl] > .5:
                samplesToKeep[x] = 1

        newXTrain = np.zeros(( int(np.sum(samplesToKeep)) , X_train.shape[1] ))
        newYTrain = np.zeros(( int(np.sum(samplesToKeep)) , y_train.shape[1] ))
        count = 0
        for x in range(X_train.shape[0]):
            if samplesToKeep[x]:
                newXTrain[count,:] = X_train[x,:]
                newYTrain[count,:] = y_train[x,:]
                count = count+1

        newModel = p1b2_baseline_keras2.main(newXTrain,newYTrain,X_test,y_test,True)
        newPreds = newModel.predict_classes(X_test)
        
        for x in range(X_test.shape[0]):
            if self._origPredictions[x] == cl:
                assert self._origPredictions[x] == newPreds[int(np.sum(samplesToKeep[:x]))], "Removing samples from classes other than n causes classifications of n to change to other classes"

    

    def __shuffleColumns(self, x):
        x = np.transpose(x)
        np.random.shuffle(x)
        x = np.transpose(x)

    #https://stackoverflow.com/questions/4601373/better-way-to-shuffle-two-numpy-arrays-in-unison
    def __shuffleColumnsInUnison(self, a, b,):
        a = np.transpose(a)
        b = np.transpose(b)

        assert len(a) == len(b)
        p = np.random.permutation(len(a))
        
        a = np.transpose(a[p])
        b = np.transpose(b[p]) 
        return a, b

    def __getCopiesOfData(self):
        return (copy.copy(self.X_train),copy.copy(self.y_train)),(copy.copy(self.X_test),copy.copy(self.y_test))

    def __permuteLabels(self,y_train,y_test):
        p = np.arange(y_train.shape[1])
        np.random.shuffle(p)

        for x in range(y_train.shape[0]):
            i = np.where(y_train[x,:] > .5)
            y_train[x,i] = 0
            y_train[x,p[i]] = 1
        
        for x in range(y_test.shape[0]):
            i = np.where(y_test[x,:] > .5)
            y_test[x,i] = 0
            y_test[x,p[i]] = 1

        return p
        

if __name__ == '__main__':
    unittest.main()