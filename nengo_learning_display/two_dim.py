import nengo
import numpy as np
import base64

try:
    from cStringIO import StringIO       # Python 2
except ImportError:
    from io import BytesIO as StringIO   # Python 3

class Plot2D(nengo.Node):
    def __init__(self, connection, domain, range, dimension=0):
        import PIL
        from PIL import Image
        self.connection = connection

        ensemble = connection.pre_obj

        self.decoder_probe = nengo.Probe(connection, 'weights')

        self.ensemble = ensemble

        if domain.shape[2] != ensemble.dimensions:
            raise Exception('domain must be (X-pts x Y-pts x dimensions)')
        self.domain = domain

        self.sim = None
        self.w = None
        self.range = range

        self.a = None
        self.dimension = dimension

        template = '''
            <svg width="100%%" height="100%%" viewbox="0 0 100 100">
                %s
            </svg>'''

        def plot(t):
            if self.w is None:
                return
            y = np.dot(self.a, self.w.T)

            y = (y - self.range[0])/(self.range[1] - self.range[0])

            y = np.clip(y * 255, 0, 255)

            y = y.astype('uint8')

            png = Image.fromarray(y[:,:])
            buffer = StringIO()
            png.save(buffer, format="PNG")
            img_str = base64.b64encode(buffer.getvalue()).decode('utf-8')
                                
            img = '''<image width="100%%" height="100%%"
                      xlink:href="data:image/png;base64,%s" 
                      style="image-rendering: pixelated;">
                  ''' % img_str

            plot._nengo_html_ = template % img

        super(Plot2D, self).__init__(plot, size_in=0, size_out=0)
        self.output._nengo_html_ = template % ''

    def update(self, sim):
        if sim is None:
            return
        if self.a is None:
            _, self.a = nengo.utils.ensemble.tuning_curves(self.ensemble, 
                                                           sim, self.domain)
        self.w = sim._probe_outputs[self.decoder_probe][-1][self.dimension]
        del sim._probe_outputs[self.decoder_probe][:]
