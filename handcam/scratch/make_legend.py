import pylab

font = {'family': 'normal',
        'weight': 'bold',
        'size': 22}
import matplotlib

matplotlib.rc('font', **font)
fig = pylab.figure()
figlegend = pylab.figure(figsize=(6,4))
ax = fig.add_subplot(111)
lines = ax.plot(range(10), pylab.randn(10), range(10), pylab.randn(10), range(10), pylab.randn(10), range(10), pylab.randn(10))
figlegend.legend(lines, ('x', 'y', 'z', 'w (pose only)'), 'center')
fig.show()
figlegend.show()
figlegend.savefig('legend.png')