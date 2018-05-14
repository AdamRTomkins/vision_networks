import numpy as np
import networkx as nx
from pyESN import ESN
from matplotlib import pyplot as plt
import csv

from matplotlib import pyplot, patches

def generate_ffbo_graph(filename="ffbo_connectivity.csv",write=False,log=False):
    G = nx.DiGraph()
    with open(filename, "rb") as f:
        reader = csv.reader(f, delimiter=",")
        for i, line in enumerate(reader):
            if i !=0: 
                pre = line[0]
                post = line[1]
                w = line[2]
                if pre not in G.nodes(): G.add_node(pre)
                if post not in G.nodes(): G.add_node(post)
                try:
                    if int(w) != 0:
                        G.add_edge(pre,post,weight=int(w))
                except:
                    if log: print "Failed on %s %s %s " % (pre,post,w)

    if write: 
        nx.write_gexf(G, "vision_network.gexf")
        
    return G


def draw_adjacency_matrix(G, node_order=None, partitions=[], colors=[]):
    """
    - G is a netorkx graph
    - node_order (optional) is a list of nodes, where each node in G
          appears exactly once
    - partitions is a list of node lists, where each node in G appears
          in exactly one node list
    - colors is a list of strings indicating what color each
          partition should be
    If partitions is specified, the same number of colors needs to be
    specified.
    """
    adjacency_matrix = nx.to_numpy_matrix(G, dtype=np.bool, nodelist=node_order)

    #Plot adjacency matrix in toned-down black and white
    fig = pyplot.figure(figsize=(5, 5)) # in inches
    pyplot.imshow(adjacency_matrix,
                  cmap="Greys",
                  interpolation="none")
    
    # The rest is just if you have sorted nodes by a partition and want to
    # highlight the module boundaries
    assert len(partitions) == len(colors)
    ax = pyplot.gca()
    for partition, color in zip(partitions, colors):
        current_idx = 0
        for module in partition:
            ax.add_patch(patches.Rectangle((current_idx, current_idx),
                                          len(module), # Width
                                          len(module), # Height
                                          facecolor="none",
                                          edgecolor=color,
                                          linewidth="1"))
            current_idx += len(module)
            

import math 
def hex_points(center=(0,0), r=1,rotation=0):
    cx = center[0]
    cy = center[0]


    names =['home', 'A',  'B', 'C', 'D', 'E',  'F',  'J',   'K'   ,'L'   ,'P'   ,'Q'   ,'R']
    angles =       [ 0,   60,  120, 180, 240,  300,  90,    120,   150,   270,   300,   330 ] 
    rmulti =       [ 1,    1,    1,  1,   1,    1,    2,     2,     2,     2,     2,      2]
    points = np.zeros((2,len(names)))
    points[0,0] = cx
    points[1,0] = cy
    
    angles = (np.array(angles) + rotation) % 360
        
    
    angles = [a* (math.pi/180) for a in angles] 
    for  i,a in enumerate(angles):
        points[0,i+1] = cx + ((r * rmulti[i]) *  math.cos(a))
        points[1,i+1] = cy + ((r * rmulti[i]) *math.sin(a))
        
    return points,names


def extract_data(image,scale,center,r,rotation=0):
    """image : a 2D pixel array """
    px,py = image.shape
    points,names = hex_points(center,r,rotation)
    
    pxs = (np.array(image.shape).reshape((2,1))* points).astype(int)
    
    l = []
    for c in pxs.T:
        try:
            l.append(image[c[0],c[1]])
        except:
            print "Error %s" % c
            
    data = np.array(l)
    
    #data = np.array([image[c[0],c[1]] for c in pxs.T ])

    return pxs,data.T,names

def generate_image(reps=10,width=100,offset=0):
    x = np.sin(offset+(np.linspace((reps/2)*-np.pi, (reps/2)*np.pi, width)))
    x = np.vstack([x for i in np.arange(x.shape[0])])
    return x    


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
        ps,d,names = extract_data(image,(2,2),m, hex_radius,rotation=rotation)
        input_data.append(d)

    return np.array(input_data), input_class, ps, names



#image = bars_signal(n=100)
image = bars_signal(n=100,on_size=3,off_size=10,on_val = 0.5,off_val = 0.1)


def generate_centers(trial_length = 100,path_start = 0.2,path_end = 0.8):
    path_steps = (path_end-path_start)/trial_length
    centers = [(n,n) for n in np.arange(path_start,path_end,path_steps)]
    return centers
