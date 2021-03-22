# -*- coding: utf-8 -*-
"""
Created on Tue Dec  1 13:35:04 2020

@author: Micah Gross
"""

# initiate app in Anaconda Navigator with
# cd "C:\Users\BASPO\Google Drive\AeroLab_PowerPod"
# streamlit run virtual_elevation_streamlit_app.py

import streamlit as st
import pandas as pd
import numpy as np
import os
import json
from io import BytesIO
# import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline, interp1d
import time
from process_fit_file import get_fit_dfs
# import plotly.express as px
# import plotly.io as pio
# pio.renderers.default = "svg"
#%%
def get_ppod_data(ppod_csv_file):
    df_ppod = pd.read_csv(ppod_csv_file, header=4)
    
    return df_ppod

def align_signals(signal1,subsig2):
    residuals=[]
    if len(signal1)>len(subsig2):
        for i in range(len(signal1)+len(subsig2)):# i=0 # i=i+1 # i=len(subsig2)-10 # i=len(subsig2)+10 # i = len(signal1)-10 # i = len(signal1)+10
            ss1_off=min([i,
                         len(signal1)
                         ])
            ss1_on=max([0,
                        i-len(subsig2)#signal1.index.max()-i#i-len(subsig2)+1
                        ])
            ss2_on=max([0,
                        len(subsig2)-i#Isubsig2.index.max()-i
                        ])
            ss2_off=min([len(subsig2),
                         len(subsig2)-(i-len(signal1))
                             ])
            if i<int(len(subsig2)/2):
                residuals.append(float('inf'))
            elif i>int(len(signal1)+(len(subsig2)/2)):
                residuals.append(float('inf'))
            else:
                residuals.append(np.mean(abs(subsig2.iloc[ss2_on:ss2_off].values-signal1.iloc[ss1_on:ss1_off].values)))# mean squared difference between the two signals when the shorter signal is shifted by i samples from right to left along the longer signal; the mean instead of the sum because it is independent of (normalized to) the number of samples compared
 
    shift = np.argmin(residuals)
    root_mean_square_diff = min(residuals)
    for i in [np.argmin(residuals)]:# i = np.argmin(residuals)
        ss1_off=min([i,
                     len(signal1)
                     ])
        ss1_on=max([0,
                    i-len(subsig2)#signal1.index.max()-i#i-len(subsig2)+1
                    ])
        ss2_on=max([0,
                    len(subsig2)-i#Isubsig2.index.max()-i
                    ])
        ss2_off=min([len(subsig2),
                     len(subsig2)-(i-len(signal1))
                         ])
        
    subsig1 = signal1[ss1_on:ss1_off]#.reset_index(drop=True)
    subsig2 = subsig2[ss2_on:ss2_off]
    
    return subsig1.index, subsig2.index, root_mean_square_diff# len(subsig1.index), len(subsig2.index)
#%%
@st.cache(allow_output_mutation=True)
def merge_data(df_ppod, df_power, df_laps):#, plot=False):
    signal1 = df_ppod['Speed (km/hr)']
    try:
        signal2 = df_power['speed']
    except:
        signal2 = df_power['Km/h']
    lap_stops = [df_laps.loc[i,'timestamp'] for i in range(len(df_laps))]
    starts = [0] + [df_power[df_power['timestamp']==ls].index.min() for ls in lap_stops[:-1]]
    stops = [i-1 for i in starts[1:]] + [df_power.index.max()]
    for stop in stops:# stop = stops[0] # stop = stops[stops.index(stop)+1]
        prev_start = max([s for s in starts if s<stop])
        subsig2 = signal2.loc[prev_start:stop]# subsig2.plot()
        subsig1_index, subsig2_index, root_mean_square_diff = align_signals(signal1,subsig2)
        if root_mean_square_diff<2:
            for col in ['Speed (km/hr)', 'Wind Speed (km/hr)', 'Elevation (meters)', 'Hill slope (%)', 'Temperature (degC)', 'Air Dens (kg/m^3)']:# df_ppod.columns
                df_power.loc[subsig2_index,col] = df_ppod.loc[subsig1_index,col].values
    df_power['Speed (m/s)'] = df_power['Speed (km/hr)']/3.6
    df_power['Wind Speed (m/s)'] = df_power['Wind Speed (km/hr)']/3.6
        
    return df_power

# @st.cache(suppress_st_warning=True)
def get_laps(df_power, df_laps, method='timestamp'):#, plot=False):
    Laps = {}
    # if plot:
    #     plt.close('aligned speed signal')
    #     aligned_speed_fig = plt.figure('aligned speed signal')
    #     plt.plot(df_power['speed'], label='from df_power')
    #     plt.plot(df_power['Speed (km/hr)'], label='from df_ppod')
    #     plt.xlabel('sample number')
    #     plt.ylabel('speed (km/h)')
    # else:
    #     aligned_speed_fig = None
    if method=='timestamp':
        for i,lap_nr in enumerate(np.arange(1,len(df_laps)+1)):# i,lap_nr = 0,list(np.arange(1,len(df_laps)+1))[0] # i,lap_nr = i+1,list(np.arange(1,len(df_laps)+1))[i+1]
            off = df_laps.loc[i,'timestamp']
            on = df_laps.loc[i-1,'timestamp'] if lap_nr>1 else df_power['timestamp'].iloc[0]
            if off>on:
                # print(on, off)
                Laps[lap_nr]=df_power[((df_power['timestamp']>=on) & (df_power['timestamp']<off))]
                idx_on = Laps[lap_nr].index.min()
                # if plot:
                #     plt.plot([idx_on, idx_on], [0, 60], Color='black', label=('lap' if lap_nr==1 else '_noLegend_'))
                #     plt.text(idx_on, 60, 'L'+str(lap_nr))
    elif method=='elapsed_time':
        for i,lap_nr in enumerate(np.arange(1,len(df_laps)+1)):# i,lap_nr = 0,list(np.arange(1,len(df_laps)+1))[0] # i,lap_nr = i+1,list(np.arange(1,len(df_laps)+1))[i+1]
            on = df_laps.loc[:i,'total_elapsed_time'].sum() - df_laps.loc[i,'total_elapsed_time']
            off = df_laps.loc[:i,'total_elapsed_time'].sum()
            if off>on:
                # print(on, off)
                Laps[lap_nr]=df_power[((df_power['Minutes']*60>=on) & (df_power['Minutes']*60<off))]
                # idx_on = Laps[lap_nr].index.min()
                # if plot:
                #     plt.plot([idx_on, idx_on], [0, 60], Color='black', label=('lap' if lap_nr==1 else '_noLegend_'))
                #     plt.text(idx_on, 60, 'L'+str(lap_nr))
    # if plot:
    #     plt.legend()
    #     st.pyplot(aligned_speed_fig)
    return Laps

@st.cache
def virtual_elevation_lap(Laps, lap_nrs, m, shift=0, constant_elevation=False, use_wind=True, correct_elevation_drift=True, print_results=False, **kwargs):
    '''
    # P = (Crr*m*g*v_ground) + (slope*m*g*v_ground) + (m*a*v_ground) + (0.5*CdA*rho*v_ground*v_air**2)
    # slope = (P - (Crr*m*g*v_ground) - (m*a*v_ground) - (0.5*CdA*rho*v_ground*v_air**2))/(m*g*v_ground)
    P = (Crr*m*g*v_ground) + (m*a*v_ground) + (0.5*CdA*rho*v_ground*v_air**2) + (m*g*elev_change)
    elev_change = (P - (Crr*m*g*v_ground) - (m*a*v_ground) - (0.5*CdA*rho*v_ground*v_air**2))/(m*g)
    m, shift, constant_elevation = 85.0, 7, False
    '''
    g=9.81
    if (('Crr' in kwargs) & (kwargs.get('Crr') is not None)):
        Crrs = [kwargs.get('Crr')]# Crrs = [Crr]
    else:
        Crrs = np.arange(0.001, 0.01, 0.001)
    if 'CdA' in kwargs:
        CdAs = [kwargs.get('CdA')]# CdAs = [CdA]
    else:
        CdAs = np.arange (0.15, 0.35, 0.001)
    Elevation = {}
    Virt_Elevation = {}
    CdA_dict = {}
    Crr_dict = {}
    for lap_nr in lap_nrs:# lap_nr = lap_nrs[0] # lap_nr = lap_nrs[13]
        try:# if the lap has available power data synchronized speed data, as would be the case for relevant laps but not necessarily all laps
            if 'int' in str(type(lap_nr)):
                df = Laps[lap_nr]
            elif type(lap_nr)==list:
                df = pd.concat([Laps[l] for l in np.arange(min(lap_nr),max(lap_nr)+1)])
                idx_next_lap = [Laps[l].index.min() for l in np.arange(min(lap_nr),max(lap_nr)+1)]
            try:
                P = df['power'].astype(float)
            except:
                P = df['Watts'].astype(float)
            v_ground = df['Speed (m/s)']
            a = pd.Series(np.gradient(v_ground), index=v_ground.index)
            rho = df['Air Dens (kg/m^3)']
            if use_wind:
                v_air = df['Wind Speed (m/s)']# v_air = pd.Series(0, index=v_ground.index)
            else:
                v_air = v_ground.copy(deep=True)
            if constant_elevation:
                elev = pd.Series(df['Elevation (meters)'].iloc[0], index=df.index)
            else:
                elev = df['Elevation (meters)']
                if ((correct_elevation_drift) & ('int' in str(type(lap_nr)))):
                    # idxmaxs = []
                    # maxs = []
                    # # elev.plot()
                    # for i in idx_next_lap:# i=idx_next_lap[0]
                    #     j = min([j for j in idx_next_lap if j>i]+[df.index.max()])
                    #     idxmaxs.append(elev.loc[i:j].idxmax())
                    #     maxs.append(elev.loc[i:j].max())
                    #     # plt.scatter(elev.loc[i:j].idxmax(), elev.loc[i:j].max())
                    #     # plt.scatter(i, elev.loc[i])
                    xs = [elev.index.min(), elev.index.max()]
                    ys = [elev.loc[x] for x in xs]
                    interp = interp1d(xs,ys)# LinearRegression()
                    # lr.fit(xs,ys)
                    drift = pd.Series(interp(elev.index),
                                      index=elev.index)
                    drift = drift - drift.iloc[0]
                    # plt.plot(elev.index, drift)
                    elev = elev - drift
                    # elev.plot()
                elif ((correct_elevation_drift) & (type(lap_nr)==list)):
                    idxmaxs = []
                    maxs = []
                    # elev.plot()
                    for i in idx_next_lap:# i=idx_next_lap[0]
                        j = min([j for j in idx_next_lap if j>i]+[df.index.max()])
                        idxmaxs.append(elev.loc[i:j].idxmax())
                        maxs.append(elev.loc[i:j].max())
                        # plt.scatter(elev.loc[i:j].idxmax(), elev.loc[i:j].max())
                        # plt.scatter(i, elev.loc[i])
                    cs = CubicSpline(idxmaxs,maxs)
                    drift = pd.Series(cs(elev.index),
                                    index=elev.index)
                    drift = drift - drift.iloc[0]
                    # plt.plot(elev.index, drift)
                    elev = elev - drift
                    # elev.plot()

            residuals = pd.DataFrame(dtype=float)
            if shift!=0:
                elev = elev.shift(shift)
            i=0
            for Crr in Crrs:# Crr = 0.005
                for CdA in CdAs:#np.arange (0.15, 0.35, 0.001):# CdA = 0.22
                    i+=1
                    elev_change = (P - (Crr*m*g*v_ground) - (m*a*v_ground) - (0.5*CdA*rho*v_ground*v_air**2))/(m*g)
                    virt_elev = (elev_change.cumsum()) + elev.iloc[shift]
                    raw_diff = virt_elev - elev
                    residuals.loc[i,'Crr'] = Crr
                    residuals.loc[i,'CdA'] = CdA
                    residuals.loc[i,'mean_square_diff'] = (raw_diff**2).mean()
        
            for i in [residuals['mean_square_diff'].idxmin()]:
                Crr = residuals.loc[i,'Crr']
                CdA = residuals.loc[i,'CdA']
                # if print_results:
                #     print('Lap '+str(lap_nr))
                # if 'Crr' not in kwargs:
                #     if print_results:
                #         print('Crr: '+str(round(Crr,3)))
                # if print_results:
                #     print('CdA: '+str(round(CdA,3)))
                elev_change = (P - (Crr*m*g*v_ground) - (m*a*v_ground) - (0.5*CdA*rho*v_ground*v_air**2))/(m*g)
                virt_elev = (elev_change.cumsum()) + elev.iloc[shift]
            # plt.close('elevation vs. virtual elevation, lap '+str(lap_nr))#+', Crr='+str(round(Crr,3)))
            # plt.figure('elevation vs. virtual elevation, lap '+str(lap_nr))#+', Crr='+str(round(Crr,3)))
            # plt.plot(elev, label='elevation')
            # plt.plot(virt_elev, label='virt elev CdA='+str(round(CdA,3))+', Crr='+str(round(Crr,3)))
            # # plt.plot(virt_elev*1.05, label='virt_elev*1.05')
            # plt.legend()
            if 'int' in str(type(lap_nr)):
                Elevation[lap_nr] = pd.Series(elev, index=elev.index)
                Virt_Elevation[lap_nr] = pd.Series(virt_elev, index=elev.index)
                CdA_dict[lap_nr] = CdA
                Crr_dict[lap_nr] = Crr
            elif type(lap_nr)==list:
                Elevation[str(lap_nr[0])+'-'+str(lap_nr[1])] = pd.Series(elev, index=elev.index)
                Virt_Elevation[str(lap_nr[0])+'-'+str(lap_nr[1])] = pd.Series(virt_elev, index=elev.index)
                CdA_dict[str(lap_nr[0])+'-'+str(lap_nr[1])] = CdA
                Crr_dict[str(lap_nr[0])+'-'+str(lap_nr[1])] = Crr
        except:
            continue
    return Elevation, Virt_Elevation, CdA_dict, Crr_dict
    
def user_input_options():
    Options = {}
    Options['plot_synced_speed'] = st.sidebar.checkbox('plot synchronized speed',
                                                          value=False,
                                                          key='plot_synced_speed')
    return Options

# def plot_VE(Elevation, Virt_Elevation, CdA_dict, Crr_dict, display_lap):
#     plt.close('elevation vs. virtual elevation, lap '+str(display_lap))#+', Crr='+str(round(Crr,3)))
#     fig_VE_lap = plt.figure('elevation vs. virtual elevation, lap '+str(display_lap))#+', Crr='+str(round(Crr,3)))
#     plt.title('Lap: '+str(display_lap))
#     plt.plot(Elevation[display_lap], label='elevation')
#     plt.plot(Virt_Elevation[display_lap], label='virt elev CdA='+str(round(
#         CdA_dict[display_lap]
#         ,3))+', Crr='+str(round(
#             Crr_dict[display_lap],
#             3)))
#     # plt.plot(virt_elev*1.05, label='virt_elev*1.05')
#     plt.legend()
#     st.write('CdA')
#     st.write(CdA_dict[display_lap])
#     st.pyplot(fig_VE_lap)

#%%
st.write("""

# Virtual Elevation with wind correction

""")
st.sidebar.header('Options')
save_variables = st.sidebar.checkbox('save variables',
                                        value=False,
                                        key='save_variables')
uploaded_files = st.file_uploader("upload 4 .csv files", accept_multiple_files=True)
if len(uploaded_files) > 0:
    fit_file = [f for f in uploaded_files if '.fit' in f.name][0]
    ppod_csv_file = [f for f in uploaded_files if (('Velocomp' in f.name) & ('.csv' in f.name))][0]
    if save_variables:
        with open(os.path.join(os.getcwd(),'saved_variables',(fit_file.name.split('.')[0])+'_bytesIO.txt'), 'wb') as fp:
            fp.write(fit_file.getbuffer())
        with open(os.path.join(os.getcwd(),'saved_variables',(ppod_csv_file.name.split('.')[0])+'_bytesIO.txt'), 'wb') as fp:
            fp.write(ppod_csv_file.getbuffer())

    df_power, df_laps, _ = get_fit_dfs(fit_file)
    df_ppod = get_ppod_data(ppod_csv_file)
    if save_variables:
        df_ppod.to_json(os.path.join(os.getcwd(),'saved_variables','df_ppod.json'), orient='index', date_format='iso')
        df_laps.to_json(os.path.join(os.getcwd(),'saved_variables','df_laps.json'), orient='index', date_format='iso')

    if df_ppod is not None:
        plot_synced_speed = st.sidebar.checkbox('plot synchronized speed',
                                                value=False,
                                                key='plot_synced_speed')# plot_synced_speed= True
        if save_variables:
            with open(os.path.join(os.getcwd(),'saved_variables','plot_synced_speed.json'), 'w') as fp:
                json.dump(plot_synced_speed, fp)
            
        df_power = merge_data(df_ppod, df_power, df_laps)
        if save_variables:
            df_power.to_json(os.path.join(os.getcwd(),'saved_variables','df_power.json'), orient='index', date_format='iso')

        if 'timestamp' in df_power.columns:
            Laps = get_laps(df_power, df_laps, method='timestamp',
                              # plot = Options['plot_synced_speed']
                              # plot = plot_synced_speed
                            )
        elif 'Minutes' in df_power.columns:
            Laps = get_laps(df_power, df_laps, method='elapsed_time',
                              # plot = Options['plot_synced_speed']
                              # plot = plot_synced_speed
                              )
        # if plot_synced_speed:
        #     plt.close('aligned speed signal')
        #     aligned_speed_fig = plt.figure('aligned speed signal')
        #     plt.plot(df_power['speed'], label='from df_power')
        #     plt.plot(df_power['Speed (km/hr)'], label='from df_ppod')
        #     plt.xlabel('sample number')
        #     plt.ylabel('speed (km/h)')
        #     for lap_nr in Laps.keys():
        #         idx_on = Laps[lap_nr].index.min()
        #         plt.plot([idx_on, idx_on], [0, 60], Color='black', label=('lap' if lap_nr==1 else '_noLegend_'))
        #         plt.text(idx_on, 60, str(lap_nr))
        #     st.pyplot(aligned_speed_fig)
            
        if Laps is not None:
            system_mass = st.sidebar.number_input('system mass (kg)', value=85.0, step=0.5)# system_mass = 85.0
            constant_elevation = st.sidebar.checkbox('assume constant elevation', value=False)# constant_elevation =True
            use_wind = st.sidebar.checkbox('use wind data', value=True)# use_wind =True
            correct_elevation_drift = st.sidebar.checkbox('correct elevation drift within laps', value=True)# correct_elevation_drift = True
            user_crr = st.sidebar.number_input('assumed Crr', value=0.005, step=0.001, format='%f')# user_crr = 0.005
            shift = st.sidebar.number_input('graphical shift for better alignment', value=7, step=1)# shift = 7
            if save_variables:
                with open(os.path.join(os.getcwd(),'saved_variables','system_mass.json'), 'w') as fp:
                    json.dump(system_mass, fp)
                with open(os.path.join(os.getcwd(),'saved_variables','constant_elevation.json'), 'w') as fp:
                    json.dump(constant_elevation, fp)
                with open(os.path.join(os.getcwd(),'saved_variables','use_wind.json'), 'w') as fp:
                    json.dump(use_wind, fp)
                with open(os.path.join(os.getcwd(),'saved_variables','correct_elevation_drift.json'), 'w') as fp:
                    json.dump(correct_elevation_drift, fp)
                with open(os.path.join(os.getcwd(),'saved_variables','user_crr.json'), 'w') as fp:
                    json.dump(user_crr, fp)
                with open(os.path.join(os.getcwd(),'saved_variables','shift.json'), 'w') as fp:
                    json.dump(shift, fp)
                
            Lap_nr_selections = {}
            for l_nr in Laps.keys():
                Lap_nr_selections[l_nr] = st.sidebar.checkbox('lap '+str(l_nr),
                                                              value=False,
                                                              key=l_nr
                                                              )
            lap_nrs = [int(k) for k in Lap_nr_selections.keys() if Lap_nr_selections[k]==True]

            lap_ranges = st.sidebar.text_input('Additional lap ranges, e.g., 2-5, 7-8')

            if lap_ranges != '':
                for lr_str in lap_ranges.replace(' ','').split(','):
                    lap_nrs.append(lr_str)
            if save_variables:
                with open(os.path.join(os.getcwd(),'saved_variables','lap_nrs.json'), 'w') as fp:
                    json.dump(lap_nrs, fp)
                
            display_lap = st.radio('select lap to display',
                                    options=lap_nrs)

            if save_variables:
                with open(os.path.join(os.getcwd(),'saved_variables','display_lap.json'), 'w') as fp:
                    json.dump(display_lap, fp)
                
            
            if display_lap is not None:
                if type(display_lap)==str:# e.g. display_lap='2-4'
                    l_nrs = [[int(x) for x in display_lap.split('-')]]
                else:# e.g., display_lap=2
                    l_nrs = [display_lap]
                if save_variables:
                    with open(os.path.join(os.getcwd(),'saved_variables','l_nrs.json'), 'w') as fp:
                        json.dump(l_nrs, fp)
                    
                Elevation, Virt_Elevation, CdA_dict, Crr_dict = virtual_elevation_lap(Laps,
                                                                                      lap_nrs=l_nrs,#list(Laps.keys()),
                                                                                      m=system_mass,
                                                                                      shift=shift,#shift,
                                                                                      constant_elevation=constant_elevation,#(True if 'constant_elevation' in options_checklist_value else False),
                                                                                      use_wind=use_wind,#(True if 'use_wind' in options_checklist_value else False),
                                                                                      correct_elevation_drift=correct_elevation_drift,#(True if 'correct_elevation_drift' in options_checklist_value else False),
                                                                                      Crr=user_crr,
                                                                                      )
                slider_CdA = st.slider('adjust CdA manually',
                                        min_value=0.15,
                                        max_value=0.4,
                                        value=float(str(np.round(float(CdA_dict[display_lap]), 3))),
                                        step=0.001,
                                        format='%f',
                                        )
                Elevation, Virt_Elevation, CdA_dict, Crr_dict, = virtual_elevation_lap(Laps,
                                                                                        lap_nrs=l_nrs,#list(Laps.keys()),
                                                                                        m=system_mass,
                                                                                        shift=shift,#shift,
                                                                                        constant_elevation=constant_elevation,#(True if 'constant_elevation' in options_checklist_value else False),
                                                                                        use_wind=use_wind,#(True if 'use_wind' in options_checklist_value else False),
                                                                                        correct_elevation_drift=correct_elevation_drift,#(True if 'correct_elevation_drift' in options_checklist_value else False),
                                                                                        Crr=user_crr,
                                                                                        CdA=slider_CdA,
                                                                                        )
                # plot_VE(Elevation, Virt_Elevation, CdA_dict, Crr_dict, display_lap)

#%%
def recover_saved_variables():
    import os
    from io import BytesIO
    for (_, _, files) in os.walk(os.path.join(os.getcwd(), 'saved_variables')):
        files = [os.path.join(os.getcwd(), 'saved_variables', f) for f in files if '_bytesIO.txt' in f]
    fit_file_name = [f for f in files if 'Velocomp' not in f][0]
    ppod_csv_file_name = [f for f in files if 'Velocomp' in f][0]
    with open(fit_file_name, 'rb') as fh:
        fit_file = BytesIO(fh.read())
    with open(ppod_csv_file_name, 'rb') as fh:
        ppod_csv_file = BytesIO(fh.read())
    del fh
    save_variables = False
    with open(os.path.join(os.getcwd(),'saved_variables','plot_synced_speed.json'), 'r') as fp:
        plot_synced_speed = json.load(fp)
    with open(os.path.join(os.getcwd(),'saved_variables','system_mass.json'), 'r') as fp:
        system_mass = json.load(fp)
    with open(os.path.join(os.getcwd(),'saved_variables','constant_elevation.json'), 'r') as fp:
        constant_elevation = json.load(fp)
    with open(os.path.join(os.getcwd(),'saved_variables','use_wind.json'), 'r') as fp:
        use_wind = json.load(fp)
    with open(os.path.join(os.getcwd(),'saved_variables','correct_elevation_drift.json'), 'r') as fp:
        correct_elevation_drift = json.load(fp)
    with open(os.path.join(os.getcwd(),'saved_variables','user_crr.json'), 'r') as fp:
        user_crr = json.load(fp)
    with open(os.path.join(os.getcwd(),'saved_variables','shift.json'), 'r') as fp:
        shift = json.load(fp)
    with open(os.path.join(os.getcwd(),'saved_variables','lap_nrs.json'), 'r') as fp:
        lap_nrs = json.load(fp)
    with open(os.path.join(os.getcwd(),'saved_variables','display_lap.json'), 'r') as fp:
        display_lap = json.load(fp)
    with open(os.path.join(os.getcwd(),'saved_variables','l_nrs.json'), 'r') as fp:
        l_nrs = json.load(fp)
    
    del fp
        # df_ppod = pd.read_json(os.path.join(os.getcwd(),'saved_variables','df_ppod.json'), orient='index', convert_dates=['Date_time'])
        # df_laps = pd.read_json(os.path.join(os.getcwd(),'saved_variables','df_laps.json'), orient='index', convert_dates=['Date_time'])
        # df_power = pd.read_json(os.path.join(os.getcwd(),'saved_variables','df_power.json'), orient='index', convert_dates=['Date_time'])

