from cycler import cycler
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import colors
from matplotlib.ticker import MultipleLocator
from matplotlib import ticker 


def plot1D(ax, dataList, args, increase = 0, colorMap = "viridis"):

    colorFunc = getattr(plt.cm, colorMap)
    colorCycle = colorFunc(np.linspace(0, 1, num = len(dataList) + increase))
    for data, color in zip(dataList,colorCycle):
        if len(data) == 2:
            ax.plot(data[0], data[1], c = color)
        else:
            ax.plot(data, c = color)

    if ('xinf' in args.keys()) and ('xsup' in args.keys()):
        ax.set_xlim(args['xinf'],args['xsup'])
    elif ('xinf' in args.keys()) or ('xsup' in args.keys()):
       raise NameError("You have specified only a single limit for the xlims!")
    else:
        pass

    if ('yinf' in args.keys()) and ('ysup' in args.keys()):
        ax.set_ylim(args['yinf'],args['ysup'])
    elif ('yinf' in args.keys()) or ('ysup' in args.keys()):
       raise NameError("You have specified only a single limit for the ylims!")
    else:
        pass

    if 'yscale' in args.keys():
        ax.set_yscale(args['yscale'])
    if 'xscale' in args.keys():
        ax.set_xscale(args['xscale'])

    if ('xmajor' in args.keys()) and ('xminor' in args.keys()):
        if ('xscale' in args) and not(args['xscale'] == 'log'):
            ax.xaxis.set_major_locator(MultipleLocator(args['xmajor']))
            ax.xaxis.set_minor_locator(MultipleLocator(args['xminor']))
        elif ('xscale' not in args): 
            ax.xaxis.set_major_locator(MultipleLocator(args['xmajor']))
            ax.xaxis.set_minor_locator(MultipleLocator(args['xminor']))
    elif ('xmajor' in args.keys()) and not('xminor' in args.keys()): 
        if ('xscale' in args) and not(args['xscale'] == 'log'):
            ax.xaxis.set_major_locator(MultipleLocator(args['xmajor']))
        elif ('xscale' not in args): 
            ax.xaxis.set_major_locator(MultipleLocator(args['xmajor']))
            ax.xaxis.set_minor_locator(MultipleLocator(args['xminor']))
    elif not('xmajor' in args.keys()) and ('xminor' in args.keys()): 
        raise NameError("You are trying to set the minor ticks, without major ones!")

    if ('ymajor' in args.keys()) and ('yminor' in args.keys()):
        if ('yscale' in args) and not(args['yscale'] == 'log'):
            ax.yaxis.set_major_locator(MultipleLocator(args['ymajor']))
            ax.yaxis.set_minor_locator(MultipleLocator(args['yminor']))
        elif ('yscale' not in args): 
            ax.yaxis.set_major_locator(MultipleLocator(args['ymajor']))
            ax.yaxis.set_minor_locator(MultipleLocator(args['yminor']))
    elif ('ymajor' in args.keys()) and not('yminor' in args.keys()): 
        if ('yscale' in args) and not(args['yscale'] == 'log'):
            ax.yaxis.set_major_locator(MultipleLocator(args['ymajor']))
        elif ('yscale' not in args): 
            print("hello")
            ax.yaxis.set_major_locator(MultipleLocator(args['ymajor']))
    elif not('ymajor' in args.keys()) and ('yminor' in args.keys()): 
        raise NameError("You are trying to set the minor ticks, without major ones!")

    ax.tick_params(direction = "in", right = True, top = True)
    if ('xminor' in args.keys()) or ('yminor' in args.keys()):
        ax.tick_params(which = "minor", direction = "in", right = True, top = True)

def barplot(ax, dataList, args, increase = 0, width = 0.8, 
            colorMap = "viridis"):
    if 'color' not in args:
        colorFunc = getattr(plt.cm, colorMap)
        colorCycle = colorFunc(np.linspace(0, 1, num = len(dataList) + increase))
        for data, color in zip(dataList, colorCycle):
            if len(data) == 2:
                if 'kwargs' in args:
                    ax.bar(data[0], data[1], width = width,
                           color = color, **args['kwargs'])
                else:
                    ax.bar(data[0], data[1], width = width,
                           color = color)
            else:
                if 'kwargs' in args:
                    ax.bar(data, width = width, c = color,
                           **args['kwargs'])
                else:
                    ax.bar(data, width = width, c = color)
    else:
        color = args['color']
        data = dataList
        if len(data) == 2:
            if 'kwargs' in args:
                ax.bar(data[0], data[1], width = width,
                           color = color, **args['kwargs'])
            else:
                ax.bar(data[0], data[1], width = width,
                           color = color)
        else:
            if 'kwargs' in args:
                ax.bar(data, width = width, c = color,
                       **args['kwargs'])
            else:
                ax.bar(data, width = width, c = color)

    if ('xinf' in args.keys()) and ('xsup' in args.keys()):
        ax.set_xlim(args['xinf'],args['xsup'])
    elif ('xinf' in args.keys()) or ('xsup' in args.keys()):
       raise NameError("You have specified only a single limit for the xlims!")
    else:
        pass

    if ('yinf' in args.keys()) and ('ysup' in args.keys()):
        ax.set_ylim(args['yinf'],args['ysup'])
    elif ('yinf' in args.keys()) or ('ysup' in args.keys()):
       raise NameError("You have specified only a single limit for the ylims!")
    else:
        pass

    if 'yscale' in args.keys():
        ax.set_yscale(args['yscale'])
    if 'xscale' in args.keys():
        ax.set_xscale(args['xscale'])

    if ('xmajor' in args.keys()) and ('xminor' in args.keys()):
        if ('xscale' in args) and not(args['xscale'] == 'log'):
            ax.xaxis.set_major_locator(MultipleLocator(args['xmajor']))
            ax.xaxis.set_minor_locator(MultipleLocator(args['xminor']))
        elif ('xscale' not in args): 
            ax.xaxis.set_major_locator(MultipleLocator(args['xmajor']))
            ax.xaxis.set_minor_locator(MultipleLocator(args['xminor']))
    elif ('xmajor' in args.keys()) and not('xminor' in args.keys()): 
        if ('xscale' in args) and not(args['xscale'] == 'log'):
            ax.xaxis.set_major_locator(MultipleLocator(args['xmajor']))
        elif ('xscale' not in args): 
            ax.xaxis.set_major_locator(MultipleLocator(args['xmajor']))
            ax.xaxis.set_minor_locator(MultipleLocator(args['xminor']))
    elif not('xmajor' in args.keys()) and ('xminor' in args.keys()): 
        raise NameError("You are trying to set the minor ticks, without major ones!")

    if ('ymajor' in args.keys()) and ('yminor' in args.keys()):
        if ('yscale' in args) and not(args['yscale'] == 'log'):
            ax.yaxis.set_major_locator(MultipleLocator(args['ymajor']))
            ax.yaxis.set_minor_locator(MultipleLocator(args['yminor']))
        elif ('yscale' not in args): 
            ax.yaxis.set_major_locator(MultipleLocator(args['ymajor']))
            ax.yaxis.set_minor_locator(MultipleLocator(args['yminor']))
    elif ('ymajor' in args.keys()) and not('yminor' in args.keys()): 
        if ('yscale' in args) and not(args['yscale'] == 'log'):
            ax.yaxis.set_major_locator(MultipleLocator(args['ymajor']))
        elif ('yscale' not in args): 
            ax.yaxis.set_major_locator(MultipleLocator(args['ymajor']))
    elif not('ymajor' in args.keys()) and ('yminor' in args.keys()): 
        raise NameError("You are trying to set the minor ticks, without major ones!")

    ax.tick_params(direction = "in", right = True, top = True)
    if ('xminor' in args.keys()) or ('yminor' in args.keys()):
        ax.tick_params(which = "minor", direction = "in", right = True, top = True)
    elif ('yscale' in args) and (args['yscale'] == 'log'):
        ax.tick_params(which = "minor", direction = "in", right = True, top = True)

def plotContour(ax, dataList, args, colorMap = "viridis", other=None):
    cmap = plt.get_cmap(colorMap)
    level = args['levels']
    print(level)
    if len(dataList) > 3:
        print("To be implemented") 
        c = 0
    else:
        x = dataList[0]
        y = dataList[1]
        z = dataList[2]
        if other == None:
            if 'lognorm' in args:
#                c = ax.contourf(x, y, z, level, cmap=cmap, 
#                        locator=ticker.LogLocator())
                c = ax.pcolormesh(x, y, z, cmap=cmap, 
                        norm=colors.LogNorm(vmin=1e-7, vmax=1e-3))
            else:
                c = ax.contourf(x, y, z, level, cmap=cmap)
                if 'sublevels' in args:
                    sublevel = args['sublevels']
                    c2 = ax.contour(x, y, z, c.levels[::sublevel], colors="xkcd:black")
        else:
            c = ax.contourf(x, y, z, levels = other.levels, cmap=cmap)
            if 'sublevels' in args:
                sublevel = args['sublevels']
                c2 = ax.contour(x, y, z, levels = c.levels[::sublevel], colors="xkcd:black")
    if ('xinf' in args.keys()) and ('xsup' in args.keys()):
        ax.set_xlim(args['xinf'],args['xsup'])
    elif ('xinf' in args.keys()) or ('xsup' in args.keys()):
       raise NameError("You have specified only a single limit for the xlims!")
    else:
        pass

    if ('yinf' in args.keys()) and ('ysup' in args.keys()):
        ax.set_ylim(args['yinf'],args['ysup'])
    elif ('yinf' in args.keys()) or ('ysup' in args.keys()):
       raise NameError("You have specified only a single limit for the ylims!")
    else:
        pass

    if 'yscale' in args.keys():
        ax.set_yscale(args['yscale'])
    if 'xscale' in args.keys():
        ax.set_xscale(args['xscale'])

    if ('xmajor' in args.keys()) and ('xminor' in args.keys()):
        if ('xscale' in args) and not(args['xscale'] == 'log'):
            ax.xaxis.set_major_locator(MultipleLocator(args['xmajor']))
            ax.xaxis.set_minor_locator(MultipleLocator(args['xminor']))
        elif ('xscale' not in args): 
            ax.xaxis.set_major_locator(MultipleLocator(args['xmajor']))
            ax.xaxis.set_minor_locator(MultipleLocator(args['xminor']))
    elif ('xmajor' in args.keys()) and not('xminor' in args.keys()): 
        if ('xscale' in args) and not(args['xscale'] == 'log'):
            ax.xaxis.set_major_locator(MultipleLocator(args['xmajor']))
        elif ('xscale' not in args): 
            ax.xaxis.set_major_locator(MultipleLocator(args['xmajor']))
            ax.xaxis.set_minor_locator(MultipleLocator(args['xminor']))
    elif not('xmajor' in args.keys()) and ('xminor' in args.keys()): 
        raise NameError("You are trying to set the minor ticks, without major ones!")

    if ('ymajor' in args.keys()) and ('yminor' in args.keys()):
        if ('yscale' in args) and not(args['yscale'] == 'log'):
            ax.yaxis.set_major_locator(MultipleLocator(args['ymajor']))
            ax.yaxis.set_minor_locator(MultipleLocator(args['yminor']))
        elif ('yscale' not in args): 
            ax.yaxis.set_major_locator(MultipleLocator(args['ymajor']))
            ax.yaxis.set_minor_locator(MultipleLocator(args['yminor']))
    elif ('ymajor' in args.keys()) and not('yminor' in args.keys()): 
        if ('yscale' in args) and not(args['yscale'] == 'log'):
            ax.yaxis.set_major_locator(MultipleLocator(args['ymajor']))
        elif ('yscale' not in args): 
            ax.yaxis.set_major_locator(MultipleLocator(args['ymajor']))
            ax.yaxis.set_minor_locator(MultipleLocator(args['yminor']))
    elif not('ymajor' in args.keys()) and ('yminor' in args.keys()): 
        raise NameError("You are trying to set the minor ticks, without major ones!")
    
#    ax.tick_params(direction = "in", right = True, top = True)
#    if ('xminor' in args.keys()) or ('yminor' in args.keys()):
#        ax.tick_params(which = "minor", direction = "in", right = True, top = True)

    return c

if __name__ == "__main__":
    fig, ax = plt.subplots(1,1)
    d = np.random.random((3,10))
    plotargs = {'xinf': 1.0, 'xsup': 11., 'xmajor': 5, 'xminor': 1, 'yscale': 'log'}
    dat = []
    for i in range(d.shape[0]):
        dat.append(d[i,:]) 
    plot1D(ax, dat, plotargs, increase = 2, colorMap="gnuplot")
    #d = np.random.random((3,10))
    #dat = []
    #for i in range(d.shape[0]):
    #    dat.append(d[i,:]) 
    #plot1D(ax[1], dat, colorMap="viridis")
plt.show()
