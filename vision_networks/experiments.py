import numpy as np
from pyESN import ESN
from matplotlib import pyplot as plt
import csv
import networkx as nx
import networkx as nx
from matplotlib import pyplot, patches
 
from utils import *
 
import pandas as pd

def bars_signal(n=100,on_size=10,off_size=10,on_val=1,off_val=0):
    image = np.zeros((n,n))
    
    i = 0
    while i < n:
        image[i:i+on_size] = on_val
        i = i+on_size
        image[i:i+off_size] = off_val
        i = i+off_size
    
    return image

def single_trial(image, centers,input_class,hex_radius =0.1,rotation = 0):

    input_data = []
    input_class = np.array([input_class for _ in centers])
    
    for m in centers:
        ps,d,names = extract_data(image,(2,2),m, hex_radius,rotation=0)
        input_data.append(d)

    return np.array(input_data), input_class, ps, names



#image = bars_signal(n=100)
image = bars_signal(n=100,on_size=3,off_size=10,on_val = 0.5,off_val = 0.1)


def generate_centers(trial_length = 100,path_start = 0.2,path_end = 0.8):
    path_steps = (path_end-path_start)/trial_length
    centers = [(n,n) for n in np.arange(path_start,path_end,path_steps)]
    return centers



def graph_input_multipliers(input_multipliers, n_reps= 5, spectral_radius=1,plot=False):

    df = pd.DataFrame(columns=["performance", "input_multiplier", "spectral_radius"])

    n_steps = input_multipliers.shape[0]
    n_reps = n_reps

    print "steps = %s" % (n_reps*n_steps)

    for r in np.arange(n_reps):
        for i,x in enumerate(input_multipliers):
            esn = MyESN()
            esn.input_multiplier = x
            esn.random_state = np.random.randint(10000)
            esn.spectral_radius=spectral_radius
            esn.create_experiment()

            df = df.append({
                 "performance": esn.performance,
                 "input_multiplier":  esn.input_multiplier,
                 "spectral_radius": esn.spectral_radius
                   }, ignore_index=True)
            
    if plot:
        import seaborn as sns
        sns.set_style("darkgrid")
        
        ax = sns.pointplot(x="input_multiplier", y="performance", data=df)

    return df

input_multipliers = np.arange(0.1,3,0.1)
df_input_multipliers = graph_input_multipliers(input_multipliers, n_reps= 20, spectral_radius=1,plot=True)

import seaborn as sns
sns.set_style("darkgrid")

ax = sns.pointplot(x="input_multiplier", y="performance", data=df_input_multipliers)
plt.savefig('input_multiplier')
df_input_multipliers



def graph_spectral_radius(spectral_radius, n_reps= 5, input_multiplier=1,plot=False):

    df = pd.DataFrame(columns=["performance", "input_multiplier", "spectral_radius"])

    n_steps = spectral_radius.shape[0]
    n_reps = 5

    print "steps = %s" % (n_reps*n_steps)

    for r in np.arange(n_reps):
        for i,x in enumerate(spectral_radius):
            esn = MyESN()
            esn.input_multiplier = input_multiplier
            esn.random_state = np.random.randint(10000)
            esn.spectral_radius=x
            esn.create_experiment()

            df = df.append({
                 "performance": esn.performance,
                 "input_multiplier":  esn.input_multiplier,
                "spectral_radius": esn.spectral_radius
                   }, ignore_index=True)
            
    if plot:
        import seaborn as sns
        sns.set_style("darkgrid")
        
        ax = sns.pointplot(x="spectral_radius", y="performance", data=df)

    return df

spectral_radius = np.arange(0.5,1.5,0.1)
df_spectral_radius = graph_spectral_radius(spectral_radius, n_reps= 20, input_multiplier=1.3,plot=True)sns.set_style("darkgrid")
        
ax = sns.pointplot(x="spectral_radius", y="performance", data=df_spectral_radius)
plt.savefig('spectral_radius.png')