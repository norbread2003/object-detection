# This import registers the 3D projection, but is otherwise unused.
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import

fig = plt.figure()
ax = fig.add_subplot(111)
img = plt.imread("img_lights.png")
ax.imshow(img)

# ax.set_picker(5)
handle = ax.plot(np.random.rand(100), 'o', picker=5)  # 5 points tolerance


def on_pick(event):
    if event.mouseevent.dblclick == 0:
        return
    line = event.artist
    xdata, ydata = line.get_data()
    ind = event.ind
    print('on pick line:', np.array([xdata[ind], ydata[ind]]).T)
    xdata = np.delete(xdata, ind)
    ydata = np.delete(ydata, ind)
    line.set_data(xdata, ydata)
    event.canvas.draw_idle()
    pass


cid = fig.canvas.mpl_connect('pick_event', on_pick)

plt.show()
