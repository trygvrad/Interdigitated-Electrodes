import streamlit as st
import time
import numpy as np

import pair_conformal as pair_conformal
#import infinite_conformal as infinite_conformal
import infinite_fourier as infinite_fourier
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

def get_plot(epss,t,etas,LAcomp,max_n,num_cells):
    # make case
    simple_geometry_case = infinite_fourier.multiple_recursive_images(etas,t,epss,epss,LAcomp,max_n,accuracy_limit=10**-15,hybrid=True)
    st.write("Capacitance per meter of a pair of electrodes in an infinite series (periodic structure) ", simple_geometry_case.get_C(), 'F/m')
    #print(simple_geometry_case.get_C(), 'F/m')
    y=np.arange(-0.5,0.5001,1/num_cells)
    x=np.arange(-0.5,0.5001,1/num_cells)
    V_simple_geometry_case=np.zeros((len(y),len(x)))
    progress_bar = st.progress(0)
    for j in range(len(y)):
        v = simple_geometry_case.get_V(x,y[j])
        V_simple_geometry_case[j,:] += v
        progress_bar.progress((j+1)/len(y))
    progress_bar.empty()

    xx, yy = np.meshgrid(x, y)
    fig, ax = plt.subplots(1,1)
    # plot potential
    levels=np.arange(-0.5,0.51,0.05)
    CS=ax.contourf(xx,yy, V_simple_geometry_case,levels,cmap='RdBu',alpha=0.7)
    cbar=fig.colorbar(CS, ax=ax)
    ax.set_aspect('equal', 'box')
    # set_labels
    ax.set_xlabel(r"$x' = x/(a+b)$ [-]")
    ax.set_ylabel(r"$y' = y/(a+b)$ [-]")
    cbar.set_label(r'Potential [V]')
    # draw electrodes
    ax.plot([[-0.5,0.5],[-0.5+etas[0]/2,0.5-etas[0]/2]],[[0,0],[0,0]],'k',solid_capstyle='butt')
    # shade the film
    ax.add_patch(matplotlib.patches.Rectangle(
        (-0.5,0),   # (x,y)
        1,          # width
        t[0],          # height
        facecolor=[0,0,0,0.1]))
    # add dielectric constant
    ax.text(0,-0.3,r'$\varepsilon$ = '+str(epss[0]), ha='center',va='center')
    ax.text(0,t[0]/2,r'$\varepsilon$ = '+str(epss[1]), ha='center',va='center')
    ax.text(0,0.4,r'$\varepsilon$ = '+str(epss[2]), ha='center',va='center')
    # flip y axis
    ax.invert_yaxis()
    return fig


def get_plot_pair(epss,t,etas,LAcomp,max_reflections,num_cells):
    # make case
    simple_geometry_case = pair_conformal.multiple_recursive_images(etas,t,epss,epss,LAcomp,max_reflections,accuracy_limit=10**-15)
    st.write("Capacitance per meter of a single pair of electrodes", simple_geometry_case.get_C(), 'F/m')
    #print(simple_geometry_case.get_C(), 'F/m')
    minx=-1
    maxx=1
    miny=-0.5
    maxy=0.5
    delta = (maxx-minx)/(num_cells)
    x = np.arange(minx, maxx+delta*0.5, delta)
    y = np.arange(miny, maxy+delta*0.5, delta)
    V_simple_geometry_case=np.zeros((len(y),len(x)))
    progress_bar = st.progress(0)
    for j in range(len(y)):
        v = simple_geometry_case.get_V(x,y[j])
        V_simple_geometry_case[j,:] += v
        progress_bar.progress((j+1)/len(y))
    progress_bar.empty()

    xx, yy = np.meshgrid(x, y)
    fig, ax = plt.subplots(1,1)
    # plot potential
    levels=np.arange(-0.5,0.51,0.05)
    CS=ax.contourf(xx,yy, V_simple_geometry_case,levels,cmap='RdBu',alpha=0.7)
    cbar=fig.colorbar(CS, ax=ax)
    ax.set_aspect('equal', 'box')
    # set_labels
    ax.set_xlabel(r"$x' = x/(a+b)$ [-]")
    ax.set_ylabel(r"$y' = y/(a+b)$ [-]")
    cbar.set_label(r'Potential [V]')
    # draw electrodes
    ax.plot([[-0.5-etas[0]/2,0.5+etas[0]/2],[-0.5+etas[0]/2,0.5-etas[0]/2]],[[0,0],[0,0]],'k',solid_capstyle='butt')
    # shade the film
    ax.add_patch(matplotlib.patches.Rectangle(
        (minx,0),   # (x,y)
        -minx+maxx,          # width
        t[0],          # height
        facecolor=[0,0,0,0.1]))
    # add dielectric constant
    ax.text(0,-0.3,r'$\varepsilon$ = '+str(epss[0]), ha='center',va='center')
    ax.text(0,t[0]/2,r'$\varepsilon$ = '+str(epss[1]), ha='center',va='center')
    ax.text(0,0.4,r'$\varepsilon$ = '+str(epss[2]), ha='center',va='center')
    # flip y axis
    ax.invert_yaxis()
    return fig

st.header('Capacitance of interdigitated electrodes')
st.write("Widget demonstrating use of https://github.com/trygvrad/Interdigitated-Electrodes for calculating the capacitance in multilayer structures with interdigitated electrodes")
status_text = st.sidebar.empty()
#eta_input=st.sidebar.number_input('Electrode cover fraction (normalized)',0.0,1.0,0.3) #min, max, default
#t_input=st.sidebar.number_input('Thickness of layer (normalized)',0.0,10.0,0.2) #min, max, default
st.sidebar.markdown('Geometry')
a=st.sidebar.number_input('Electrode spacing, a [µm]',0.0,1000000.0,3.0) #min, max, default
b=st.sidebar.number_input('Electrode width, b [µm]',0.0,1000000.0,2.0) #min, max, default
t=st.sidebar.number_input('Thickness of layer, t [µm]',0.0,1000000.0,1.0) #min, max, default
L=st.sidebar.number_input('Length of fingers, L [µm]',0.0,1000000.0,1000.0) #min, max, default
N=st.sidebar.number_input('Number of fingers, N [-]',4,100000,100) #min, max, default
eta_input=b/(a+b)
t_input=t/(a+b)
st.sidebar.markdown('Material properties')
eps_input_0=st.sidebar.number_input('Dielectric costant above the electrodes, ε₀ [-]',0.0,10.0**10,1.0) #min, max, default
eps_input_1=st.sidebar.number_input('Dielectric costant in the film, ε₁ [-]',0.0,10.0**10,5.0) #min, max, default
eps_input_2=st.sidebar.number_input('Dielectric costant in the substrate, ε₂ [-]',0.0,10.0**10,2.0) #min, max, default

if st.sidebar.checkbox('Advanced options'):
    LAcomp=st.sidebar.number_input('Linear algebra components',1,100,4) #min, max, default
    max_n=st.sidebar.number_input('Max number of fourier components',1,180,180) #min, max, default
    num_cells=st.sidebar.number_input('Number of cells on along x-axis in figure',1,500,50) #min, max, default
    max_reflections=st.sidebar.number_input('Max number of reflections (for single pair of electrodes)',1,200,20) #min, max, default
else:
    LAcomp=4 # number of linear algebra componens
    max_n=180 # number of fourier components
    num_cells=50 # number of cells allong each axis in figure
    max_reflections=20
# declare geometry
epss=[eps_input_0,eps_input_1,eps_input_2] # dielectric constant of layer 1, 2 and 3
t=[t_input] #thickness of layer 2
etas=[eta_input,0] # cover fraction of electrodes, between layer 1 and 2 and between layer 2 and 3
# declare model parameters

simple_geometry_case = infinite_fourier.multiple_recursive_images(etas,t,epss,epss,LAcomp,max_n,accuracy_limit=10**-15,hybrid=True)
C_inf = simple_geometry_case.get_C()
simple_geometry_case = pair_conformal.multiple_recursive_images(etas,t,epss,epss,LAcomp,max_reflections,accuracy_limit=10**-15)
C_pair = simple_geometry_case.get_C()
C_E=(2*C_inf*C_pair)/(C_inf+C_pair)
C_tot=L*( (N-3)*C_inf + 2*C_E ) / 1000000 # L in [um], C_inf and C_E in [F/m], C_tot in [F]
from PIL import Image
image = Image.open('streamlit IDE.png')
st.image(image, caption='Notation used to describe the geometry of interdigitated electrodes on a thin film',
      use_column_width=True)
st.subheader('Capacitance of the structure described by the input parameters:')

st.write( C_tot, 'F')

st.subheader('The elctric field that forms the basis for the calulation:')

fig=get_plot(epss,t,etas,LAcomp,max_n,num_cells)
st.pyplot(fig)
fig_pair=get_plot_pair(epss,t,etas,LAcomp,max_reflections,num_cells)
st.pyplot(fig_pair)




# Streamlit widgets automatically run the script from top to bottom. Since
# this button is not connected to any other logic, it just causes a plain
# rerun.
