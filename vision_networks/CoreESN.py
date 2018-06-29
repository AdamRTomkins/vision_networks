import numpy as np
from pyESN import ESN
from matplotlib import pyplot as plt
import csv
import networkx as nx
import networkx as nx
from matplotlib import pyplot, patches

from CoreESN import *
from utils import *


import pandas as pd

class MyESN(object):
    def __init__(self):
        
        self.trial_length = 100
        
        #data generation
        self.trial_length = 100
        self.path_start = 0.2
        self.path_end = 0.8
        self.n_repeats = 30
        self.shuffle_trials = True
        
        self.n_outputs = 4
        self.n_inputs = 13
        self.n_reservoir = 434
        
        self.n_trials = self.n_outputs * self.n_repeats
        
        
        # ESN
        self.show_state = False
        self.spectral_radius = 1.05
        self.noise = 0.00
        self.random_state = 42
        self.input_multiplier = 10
        
        #self.create_experiment()
    
    def create_experiment(self):
            
            self.create_esn()
            self.generate_data()
            self.fit()
            self.predict()
            self.analyse_results()
            
    
    def generate_data(self):

        image = bars_signal(n=100,on_size=3,off_size=10,on_val = 0.5,off_val = 0.1)

    
        centers = {}
        # Move X directions
        centers[0] = generate_centers(trial_length = self.trial_length,path_start = self.path_start,path_end = self.path_end)
        # Move Y directions
        centers[1] = generate_centers(trial_length = self.trial_length,path_start = self.path_start,path_end = self.path_end)
        # Move X directions
        centers[2] = generate_centers(trial_length = self.trial_length,path_start = self.path_end,path_end = self.path_start)
        # Move Y directions
        centers[3] = generate_centers(trial_length = self.trial_length,path_start = self.path_end,path_end = self.path_start)

        # store data for inspection
        self.centers = centers
        self.images = {}
        self.trials = []

        # create trials array
        t = 0
        for i in np.arange(self.n_trials):
            self.trials.append(t)
            t+=1
            if t == self.n_outputs:t=0

        if self.shuffle_trials:
            from random import shuffle
            shuffle(self.trials)


        self.trial_data = []
        self.trial_class = []
        image_backup = image.copy()

        for trial in self.trials:
            cs = centers[trial]

            if trial == 1  or trial == 3: 
                image = image_backup.T.copy()
            else:
                image = image_backup.copy()
                
            self.images[trial] = image
            data, classes, ps, names = single_trial(image, cs,input_class=trial,hex_radius =0.03)
hex
            self.trial_data.append(data)
            self.trial_class.append(classes)

            self.experiment_classes = np.hstack(self.trial_class)

            self.experiment_targets= -np.ones((self.n_outputs,self.experiment_classes.shape[0]))
            
            for i,c in enumerate(self.experiment_classes):
                self.experiment_targets[c,i] = 1

            self.experiment_data = np.ones((self.trial_data[0].shape[0]*len(self.trial_data),self.trial_data[0].shape[1]))

            for i,d in enumerate(self.trial_data):
                self.experiment_data[i*self.trial_data[0].shape[0]:(i+1)*self.trial_data[0].shape[0],:] = d

            self.experiment_data = self.experiment_data.T
            
            self.trainlen = self. experiment_targets.shape[1]/2
            self.future =  self.experiment_targets.shape[1]/2
        
        
        

        

    
    def create_esn(self):        

        self.generate_liquid()

        self.esn = ESN(n_inputs = self.n_inputs,
                  n_outputs = self.n_outputs,
                  n_reservoir = self.n_reservoir,
                  spectral_radius = self.spectral_radius,
                  noise = self.noise,
                  random_state=self.random_state,
                  teacher_forcing=False,
                  matrix=self.adjacency_matrix,
                  input_matrix = self.input_matrix*self.input_multiplier
                  )

    def fit(self):
        
        pred_training = self.esn.fit(self.experiment_data[:self.trainlen].T,self.experiment_targets[:self.trainlen].T,self.show_state,self.output_node_matrix)
    
    def predict(self):
        
        self.prediction = self.esn.predict(self.experiment_data[:,self.trainlen:self.trainlen+self.future].T)


        #self.analyse_data()

    def analyse_data(self):
        # TODO: Remove assumption of 5050 split
        prediction_classes = np.array(self.trials[-(len(self.trials)/2):])
        #prediction_classes = prediction_classes[2:]

        classes = np.arange(self.n_outputs)

        results = {c:{o: [] for o in classes} for c in classes }

        for i,c in enumerate(prediction_classes):

            data = self.prediction[i*self.trial_length:(i*self.trial_length)+self.trial_length,:]

            for cl in classes:
                results[c][cl].append(np.mean(data[:,cl]))

        self.results = results


        
    
    def generate_liquid(self):
        self.G = generate_ffbo_graph()
        self.n_reservoir = self.G.number_of_nodes()
        self.input_names =['home', 'A',  'B', 'C', 'D', 'E',  'F',  'J',   'K'   ,'L'   ,'P'   ,'Q'   ,'R']

        # Create Input Matrix
        self.input_matrix = np.zeros((self.n_reservoir, self.n_inputs))
        for j,c in enumerate(self.input_names):
            for i,n in enumerate(self.G.nodes()):
                if 'L1' in n or  'L2' in n or  'L4' in n:
                    if c in n:
                        self.input_matrix[i,j] = 1                

        #Create a list of network outputs that can be used to train upon
        # We focus oon the T4 and Tm* neurons
        self.output_node_matrix = np.zeros((self.input_matrix.shape[0],self.input_matrix.shape[1]))

        for i,n in enumerate(self.G.nodes()):
            if 'Tm3' in n or 'Mi1' in n or 'Mi9' in n or 'Mi4' in n or 'T4' in n:    
                self.output_node_matrix[i,:] = 1

        self.output_node_matrix = self. output_node_matrix[:,0].T

        self.adjacency_matrix = np.array(nx.adjacency_matrix(self.G, weight='weight').todense())
        self.adjacency_matrix[self.adjacency_matrix>0] = 1
        self.adjacency_matrix[self.adjacency_matrix<1] = 0

        #return n_reservoir, adjacency_matrix,input_matrix,output_node_matrix

    def analyse_results(self):
        """ reurn the proportion of correct trials
            results : nested dictionary where keys are the expected trial_results: {class:results}
        """
        performances = {}
        for c in self.results:
            target = self.results[c][c]
            others = set(self.results[c].keys())
            others.remove(c)

            other_data = np.vstack([self.results[c][d] for d in others])
            ratio =  sum(target > np.max(other_data)) / (len(target)*1.0)
            
            
            performances[c] = ratio
        performance  = np.mean(performances.values())
        self.performance = performance
        return performance
    

        