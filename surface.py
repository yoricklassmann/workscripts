import plotly.graph_objects as go
import plotly.io as pio
import numpy as np


inp = np.genfromtxt('bmaPES.dat')
x  = inp[:,0].reshape(1001,1001)
y  = inp[:,1].reshape(1001,1001)
z1 = inp[:,2].reshape(1001,1001)
z2 = inp[:,3].reshape(1001,1001)
zmin = np.amin(z1)
if zmin < 0:
    zmin = zmin * 1.05
else:
    zmin = zmin * 0.95

z2max = np.amax(z2) * 1.05
z1max = np.amax(z1) * 1.00
x2 = x.copy()
y2 = y.copy()
z2 = np.ma.masked_where(z2 > z1max * 1.05, z2)
for i in np.arange(1001):
    for j in np.arange(1001):
        if z2[i, j] is np.ma.masked:
            x2[i, j] = np.nan
            y2[i, j] = np.nan
            z2[i, j] = np.nan

for i, xi in enumerate(x2):
    if np.isnan(xi[0]):
        xi[:] = np.nan
        y2[i][:] = np.nan
        z2[i][:] = np.nan
            

fig = go.Figure(data=[
          go.Surface(x=x,
                     y=y,
                     z=z1,
                     showscale=False,
                     colorscale='viridis',
                     cmin=zmin,
                     cmax=z1max,
                     ),
          go.Surface(x=x2,
                     y=y2,
                     z=z2,
                     colorscale='viridis',
                     cmin=zmin,
                     cmax=z1max,
                     showscale=False,
                     #opacity=0.8
                     ),
])
# Default parameters which are used when `layout.scene.camera` is not provided
#camera = dict(
#        up=dict(x=0, y=0, z=1),
#        center=dict(x=0, y=0, z=0),
#        eye=dict(x=1, y=-2, z=1.25)
#)
#fig.update_layout(scene_camera=camera,width=1000)
#fig.update_layout(paper_bgcolor='rgb(253, 246, 227)')
    
fig.update_scenes(xaxis=dict(
                    range=[-50,50],
                    title='',
#                    title='X (bohr)',
#                    backgroundcolor='rgb(253, 246, 227)',
#                    gridcolor='rgb(101, 123, 131)',
#                    zerolinecolor='rgb(101, 123, 131)',
#                    tickfont_family='Helvetica' 
                    showgrid=False,
                    zeroline=False,
                    showbackground=False,
                    showaxeslabels=False,
                    showticklabels=False
                  ),
                  yaxis=dict(
#                    title='Y (bohr)',
                    title='',
                    range=[-20,20],
                    showgrid=False,
                    zeroline=False,
                    showbackground=False,
                    showaxeslabels=False,
                    showticklabels=False
#                    backgroundcolor='rgb(253, 246, 227)',
#                    gridcolor='rgb(101, 123, 131)',
#                    zerolinecolor='rgb(101, 123, 131)',
#                    tickfont_family='Helvetica' 
                  ),
                  zaxis=dict(
                    #title='Energy (Hartree)',
                    title='',
                    range=[zmin,z2max],
                    showgrid=False,
                    zeroline=False,
                    showbackground=False,
                    showaxeslabels=False,
                    showticklabels=False
                    #backgroundcolor='rgb(253, 246, 227)',
                    #gridcolor='rgb(101, 123, 131)',
                    #zerolinecolor='rgb(101, 123, 131)',
                    #tickfont_family='Helvetica' 
                  ),
                  camera=dict(
                    up=dict(x=0, y=0, z=1),
                    center=dict(x=0, y=0, z=0),
                    eye=dict(x=-0.3, y=-2, z=-0.25)
                  ),
#                  aspectmode='manual',
#                  aspectratio=dict(x=1.5, y=1, z=1)
                  )

fig.show()
pio.write_image(fig,'surf_v3.svg',
               width=637.5,
               height=637.5,
               scale=4)
