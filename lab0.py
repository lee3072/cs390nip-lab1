
import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.utils import to_categorical
import random


# Setting random seeds to keep everything deterministic.
random.seed(1618)
np.random.seed(1618)
# tf.set_random_seed(1618)   # Uncomment for TF1.
tf.random.set_seed(1618)

# Disable some troublesome logging.
# tf-.logging.set_verbosity(tf.logging.ERROR)   # Uncomment for TF1.
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Information on dataset.
NUM_CLASSES = 10
IMAGE_SIZE = 784

# Use these to set the algorithm to use.
# ALGORITHM = "guesser"
#ALGORITHM = "custom_net"
ALGORITHM = "tf_net"





class NeuralNetwork_2Layer():
    def __init__(self, inputSize, outputSize, neuronsPerLayer, learningRate = 0.1):
        self.inputSize = inputSize
        self.outputSize = outputSize
        self.neuronsPerLayer = neuronsPerLayer
        self.lr = learningRate
        self.W1 = np.random.randn(self.inputSize, self.neuronsPerLayer)
        self.W2 = np.random.randn(self.neuronsPerLayer, self.outputSize)

    # Activation function.
    def __sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    # Activation prime function.
    def __sigmoidDerivative(self, x):
        s = self.__sigmoid(x)
        return s * (1 - s)

    # Batch generator for mini-batches. Not randomized.
    def __batchGenerator(self, l, n):
        for i in range(0, len(l), n):
            yield l[i : i + n]

    # Training with backpropagation.
    def train(self, xVals, yVals, epochs = 100000, minibatches = True, mbs = 100):
        #Done: Implement backprop. allow minibatches. mbs should specify the size of each minibatch.
        #reference: https://thispointer.com/python-use-of-yield-keyword-generators-explained-with-examples/
        #reference: https://www.tensorflow.org/guide/eager
        xValsBatchGen = self.__batchGenerator(xVals,mbs)
        yValsBatchGen = self.__batchGenerator(yVals,mbs)
        while True:
            try:
                xValsBatch = next(xValsBatchGen)
                yValsBatch = next(yValsBatchGen)
                for i in range(epochs):
                    layer1, layer2 = self.__forward(xValsBatch)
                    layer2_delta = (yValsBatch - layer2) * self.__sigmoidDerivative(layer2)
                    layer1_delta = layer2_delta.dot(self.W2.T) * self.__sigmoidDerivative(layer1)
                    self.W1 += layer1.T.dot(layer2_delta) * self.lr
                    self.W2 += xVals.T.dot(layer1_delta) * self.lr    
            except StopIteration:
                break
        
                                          

    # Forward pass.
    def __forward(self, input):
        layer1 = self.__sigmoid(np.dot(input, self.W1))
        layer2 = self.__sigmoid(np.dot(layer1, self.W2))
        return layer1, layer2

    # Predict.
    def predict(self, xVals):
        _, layer2 = self.__forward(xVals)
        return layer2



# Classifier that just guesses the class label.
def guesserClassifier(xTest):
    ans = []
    for entry in xTest:
        pred = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        pred[random.randint(0, 9)] = 1
        ans.append(pred)
    return np.array(ans)



#=========================<Pipeline Functions>==================================

def getRawData():
    mnist = tf.keras.datasets.mnist
    (xTrain, yTrain), (xTest, yTest) = mnist.load_data()
    print("Shape of xTrain dataset: %s." % str(xTrain.shape))
    print("Shape of yTrain dataset: %s." % str(yTrain.shape))
    print("Shape of xTest dataset: %s." % str(xTest.shape))
    print("Shape of yTest dataset: %s." % str(yTest.shape))
    return ((xTrain, yTrain), (xTest, yTest))



def preprocessData(raw):
    ((xTrain, yTrain), (xTest, yTest)) = raw            
    
    #Done: Add range reduction here (0-255 ==> 0.0-1.0).
    xTrain, xTest = xTrain/255.0, xTest/255.0   #reference https://colab.research.google.com/github/tensorflow/docs/blob/master/site/en/tutorials/quickstart/beginner.ipynb?hl=ko#scrollTo=7FP5258xjs-v

    yTrainP = to_categorical(yTrain, NUM_CLASSES)
    yTestP = to_categorical(yTest, NUM_CLASSES)
    print("New shape of xTrain dataset: %s." % str(xTrain.shape))
    print("New shape of xTest dataset: %s." % str(xTest.shape))
    print("New shape of yTrain dataset: %s." % str(yTrainP.shape))
    print("New shape of yTest dataset: %s." % str(yTestP.shape))
    return ((xTrain, yTrainP), (xTest, yTestP))



def trainModel(data):
    xTrain, yTrain = data
    if ALGORITHM == "guesser":
        return None   # Guesser has no model, as it is just guessing.
    elif ALGORITHM == "custom_net":
        print("Building and training Custom_NN.")
        print("Not yet implemented.")                   #TODO: Write code to build and train your custon neural net.
        return None
    elif ALGORITHM == "tf_net":
        #Done: Write code to build and train your keras neural net.
        model = tf.keras.models.Sequential([tf.keras.layers.Flatten(),tf.keras.layers.Dense(512,activation=tf.nn.relu),tf.keras.layers.Dense(10,activation=tf.nn.softmax)])
        model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])
        model.fit(xTrain,np.argmax(yTrain, axis=1),epochs=5) # Reverse the to_categorical to perform fit, reference: https://github.com/keras-team/keras/issues/4981
        return model
    else:
        raise ValueError("Algorithm not recognized.")



def runModel(data, model):
    if ALGORITHM == "guesser":
        return guesserClassifier(data)
    elif ALGORITHM == "custom_net":
        print("Testing Custom_NN.")
        print("Not yet implemented.")                   #TODO: Write code to run your custon neural net.
        return None
    elif ALGORITHM == "tf_net":
        #Done: Write code to run your keras neural net.
        # predicted data will be an array representing the predicted number,
        # convert it to a single number to assign to a category
        return to_categorical(np.argmax(model.predict(data), axis=1), NUM_CLASSES)
    else:
        raise ValueError("Algorithm not recognized.")



def evalResults(data, preds):   #TODO: Add F1 score confusion matrix here.
    xTest, yTest = data
    acc = 0
    for i in range(preds.shape[0]):
        if np.array_equal(preds[i], yTest[i]):   acc = acc + 1
    accuracy = acc / preds.shape[0]
    print("Classifier algorithm: %s" % ALGORITHM)
    print("Classifier accuracy: %f%%" % (accuracy * 100))
    print()



#=========================<Main>================================================

def main():
    raw = getRawData()
    data = preprocessData(raw)
    model = trainModel(data[0])
    preds = runModel(data[1][0], model)
    evalResults(data[1], preds)



if __name__ == '__main__':
    main()
