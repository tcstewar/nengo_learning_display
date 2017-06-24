import nengo_learning_display
import nengo_learning_display.two_dim
import imp
imp.reload(nengo_learning_display.two_dim)
import nengo
import numpy as np


model = nengo.Network()
with model:
    stim = nengo.Node(lambda t: (np.sin(10*t), np.cos(10*t)))
    
    pre = nengo.Ensemble(n_neurons=100, dimensions=2)
    
    post = nengo.Ensemble(n_neurons=100, dimensions=2)
    
    c = nengo.Connection(pre, post, 
            function = lambda x: (0,0),
            learning_rule_type=nengo.PES())
    
    def func(x):
        return x[0], x[0]*x[1]
    nengo.Connection(post, c.learning_rule)
    nengo.Connection(stim, c.learning_rule, function=func, transform=-1)
    
    nengo.Connection(stim, pre)
    
    grid = np.meshgrid(np.linspace(-1, 1, 20), np.linspace(-1,1,20))
    grid = np.array(grid).T

    plot = nengo_learning_display.two_dim.Plot2D(c, 
                domain=grid, 
                range=(-0.5,0.5),
                dimension=1)
    
    
    

def on_step(sim):
    plot.update(sim)