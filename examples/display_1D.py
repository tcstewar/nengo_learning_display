import nengo_learning_display

import nengo
import numpy as np

model = nengo.Network()
with model:
    stim = nengo.Node(lambda t: np.sin(10*t))
    
    pre = nengo.Ensemble(n_neurons=100, dimensions=1)
    
    post = nengo.Ensemble(n_neurons=100, dimensions=1)
    
    c = nengo.Connection(pre, post, 
            function=lambda x: 0,
            learning_rule_type=nengo.PES())
    
    nengo.Connection(post, c.learning_rule)
    nengo.Connection(stim, c.learning_rule, transform=-1)
    
    
    nengo.Connection(stim, pre)
    
    plot = nengo_learning_display.Plot1D(c, 
                domain=np.linspace(-2,2,30), 
                range=(-1.5,1.5))
    
    
    

def on_step(sim):
    plot.update(sim)