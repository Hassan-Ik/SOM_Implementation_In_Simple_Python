import math
import numpy as np
import random

class SOM:
    def __init__(self, x_axis, y_axis, input_dimension, sigma, learning_rate, random_seed=3):
        """
            Self Organizing Maps Implementation in simple python code
            
            x_axis: X axis length of SOM grid
            y_axis: Y axis length of SOM grid
            
            input_dimension: total elements of a single input data row
            
            sigma: Neurons neighborhood radius 
            
            learning_rate: Learning rate to be used for SOM learning
            
            random_seed: for random values distribution
        """
        # Creating a random generator for random values; for initializing weights
        self.random_generator = np.random.RandomState(random_seed)
        
        self.x_axis = x_axis
        self.y_axis = y_axis
        self.input_dimension = input_dimension
        self.sigma = sigma
        self.learning_rate = learning_rate
        self.weights =  np.array([[[0 for x in range(self.input_dimension)] for x in range(self.x_axis)] for x in range(y_axis)], dtype=float)
        
    def get_weights(self):
        """
            Getting the weights of the SOM grid
        """
        return self.weights
    
    def check_input_dimension(self, data):
        """
            Check if data row total elements and provided input dimension is equal else throw error 
        """
        if len(data[0]) != self.input_dimension:
            raise ValueError("Received {} features, expected {}.".format(self.input_dimension, len(data[0])))
            
    def check_x_and_y_axis_len(self, x_axis, y_axis):
        """
            Function to check if SOM; x_axis and y_axis provided are equal to 0.
        """
        if x_axis ==0: 
            raise ValueError("Error! SOM X-Axis is 0!")
        if y_axis==0:
            raise ValueError("Error! SOM Y-Axis is 0!")
    
    def random_weights_init(self, data):
        """
            For Initializing Random Weights to be provided to Neurons
        """
        self.check_input_dimension(data)
        for i in range(self.y_axis):
            for j in range(self.x_axis):
                rand_i = self.random_generator.randint(len(data))
                self.weights[i][j] = data[rand_i]
                        
    def node_distance(self, inputs):
        """
            For Checking the distance between two nodes of the SOM grid
        """
        tmp = 0
        for i in len(self.inputs):
            tmp += np.power(data[i] - self.weights[i], 2)
        return np.sqrt(tmp)
    
    def find_BMU(self, data_axis):
        """
            Finding the Best Matching Unit; finding the neuron which is closest to the provided data value
        """
        distSq = (np.square(self.weights - data_axis)).sum(axis=2)
        return np.unravel_index(np.argmin(distSq, axis=None), distSq.shape)
    
    def currentNeighbourhoodRadius(self, currentIteration, lambda1):
        """
            Return Updated Neighborhood Radius for the different iterations
        """
        return self.sigma * np.exp(- currentIteration / lambda1)
    
    def currentLearningRate(self, currentIteration, lambda1):
        """
            Return Updated Learning Rate for the different iteration 
        """
        return self.learning_rate * np.exp(-currentIteration / lambda1)
    
    def update_weights(self, BMU, currentIteration, input_data, lambda1):
        """
            Updating the weights of the row data with BMU
        """
        # Learning rate selection for each epoch
        lr = self.currentLearningRate(currentIteration, lambda1)
        
        # Neighborhood radius selection for each epoch
        radius = self.currentNeighbourhoodRadius(currentIteration, lambda1)
        
        # Iterating through randomly initialized weights and update weights
        for i in range(len(self.weights[0])):
            for j in range(len(self.weights)):
                tmpDist = np.power(BMU[0] - i, 2) + np.power(BMU[1] - j, 2)
                theta = np.exp(-tmpDist / (2*np.power(radius, 2)))
                for k in range(self.input_dimension):
                    self.weights[i][j][k] = self.weights[i][j][k] + lr * theta * (input_data[k] - self.weights[i][j][k])
    
    def train(self,data, epochs):
        """
            For Training the Self Organizing Maps
        """
        if(epochs ==0):
            raise ValueError("Error! The amount of epochs should not be 0!")
        
        if(self.learning_rate > 1 or self.learning_rate <= 0):
            raise ValueError("Error! The Learning Step should be in the range (0,1]!")
        
        # For weights updation process
        lambda1 = epochs / np.log(self.sigma)
        
        data = np.array(data)

        for i in range(epochs):
            np.random.shuffle(data)
            for j in range(len(data)):
                BMUcoordinates = self.find_BMU(data[j]);
                self.update_weights(BMUcoordinates, i, data[j], lambda1);
            
    def pushData(self, input_data, input_dimension):
        if (len(input_data) == 0 or len(input_data) != input_dimension):
            raise ValueError("Error! The fed vector of attributes has a wrong dimensionality!!")
    
        trainingData.append(input_data)