import nengo
import numpy as np

class Plot1D(nengo.Node):
    def __init__(self, connection, domain, range):
        self.connection = connection

        ensemble = connection.pre_obj

        self.decoder_probe = nengo.Probe(connection, 'weights')

        self.ensemble = ensemble

        domain = np.array(domain)
        if len(domain.shape) == 1:
            domain = domain.reshape(-1, 1)

        if domain.shape[1] != ensemble.dimensions:
            raise Exception('domain must be (# points x dimensions)')
        self.domain = domain

        self.sim = None
        self.w = None
        self.range = range

        self.svg_x = np.linspace(0, 100, len(domain))

        self.a = None

        template = '''
            <svg width="100%%" height="100%%" viewbox="0 0 100 100">
                %s
                <line x1=50 y1=0 x2=50 y2=100 stroke="#aaaaaa"/>
                <line x1=0 y1=50 x2=100 y2=50 stroke="#aaaaaa"/>
            </svg>'''

        self.palette = ["#1c73b3", "#039f74", "#d65e00", 
                        "#cd79a7", "#f0e542", "#56b4ea"]

        def plot(t):
            if self.w is None:
                return
            y = np.dot(self.a, self.w.T)
            
            min_y = self.range[0]
            max_y = self.range[1]
            data = (-y - min_y) * 100 / (max_y - min_y)

            paths = []
            for i, row in enumerate(data.T):
                path = []
                for j, d in enumerate(row):
                    path.append('%1.0f %1.0f' % (self.svg_x[j], d))
                paths.append('<path d="M%s" fill="none" stroke="%s"/>' %
                             ('L'.join(path), 
                              self.palette[i % len(self.palette)]))

            plot._nengo_html_ = template % (''.join(paths))     

        super(Plot1D, self).__init__(plot, size_in=0, size_out=0)
        self.output._nengo_html_ = template % ''

    def update(self, sim):
        if sim is None:
            return
        if self.a is None:
            _, self.a = nengo.utils.ensemble.tuning_curves(self.ensemble, 
                                                           sim, self.domain)
        self.w = sim._probe_outputs[self.decoder_probe][-1]
        del sim._probe_outputs[self.decoder_probe][:]
