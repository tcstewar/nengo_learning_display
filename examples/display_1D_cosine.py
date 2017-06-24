import nengo_learning_display
import nengo_learning_display.one_dim
import imp
imp.reload(nengo_learning_display.one_dim)
import nengo
import numpy as np


model = nengo.Network()
with model:
    stim = nengo.Node(lambda t: (np.sin(10*t), np.cos(10*t)))
    
    pre = nengo.Ensemble(n_neurons=100, dimensions=2)
    
    post = nengo.Ensemble(n_neurons=100, dimensions=2)
    
    c = nengo.Connection(pre, post, 
            function = lambda x: [0,0],
            learning_rule_type=nengo.PES())
    
    nengo.Connection(post, c.learning_rule)
    nengo.Connection(stim, c.learning_rule, transform=-1)
    
    
    nengo.Connection(stim, pre)
    
    
    theta = np.linspace(-np.pi, np.pi, 30)
    domain = np.array([np.cos(theta), np.sin(theta)]).T
    
    
    plot = nengo_learning_display.one_dim.Plot1D(c, 
                domain=domain, 
                range=(-1.5,1.5))
    
    
    

def on_step(sim):
    plot.update(sim)