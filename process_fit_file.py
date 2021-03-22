# -*- coding: utf-8 -*-
"""
Created on Fri Mar 19 20:46:38 2021

@author: BASPO
"""

#%%
import os
import fitparse
from copy import copy
import pytz
import pandas as pd
#%%
def get_timestamp(messages):
    for m in messages:
        fields = m.fields
        for f in fields:
            if f.name == 'timestamp':
                return f.value
    return None

def get_event_type(messages):
    for m in messages:
        fields = m.fields
        for f in fields:
            if f.name == 'sport':
                return f.value
    return None

#%%
def get_fit_dfs(fit_file_path):
    # some prerequisite variables
    allowed_fields = ['timestamp','position_lat','position_long', 'distance',
    'enhanced_altitude', 'altitude','enhanced_speed',
                     'speed', 'heart_rate','cadence','fractional_cadence','power',
                     'temperature']
    required_fields = ['timestamp', 'position_lat', 'position_long', 'altitude']
    
    
    
    #for laps
    lap_fields = ['timestamp','start_time','start_position_lat','start_position_long',
                   'end_position_lat','end_position_long','total_elapsed_time','total_timer_time',
                   'total_distance','total_strides','total_calories','enhanced_avg_speed','avg_speed',
                   'enhanced_max_speed','max_speed','total_ascent','total_descent',
                   'event','event_type','avg_heart_rate','max_heart_rate',
                   'avg_running_cadence','max_running_cadence',
                   'lap_trigger','sub_sport','avg_fractional_cadence','max_fractional_cadence',
                   'total_fractional_cycles','avg_vertical_oscillation','avg_temperature','max_temperature']
    #last field above manually generated
    lap_required_fields = ['timestamp', 'start_time','lap_trigger']
    
    #start/stop events
    start_fields = ['timestamp','timer_trigger','event','event_type','event_group']
    start_required_fields = copy(start_fields)
    #
    all_allowed_fields = set(allowed_fields + lap_fields + start_fields)
    
    UTC = pytz.UTC
    CST = pytz.timezone('US/Central')
    
    # actual processing steps
    fitfile = fitparse.FitFile(fit_file_path,
                               data_processor=fitparse.StandardUnitsDataProcessor())
    messages = fitfile.messages
    data = []
    lap_data = []
    start_data = []
    timestamp = get_timestamp(messages)
    event_type = get_event_type(messages)
    if event_type is None:
        event_type = 'other'
    for m in messages:# m = messages[0]
        skip=False
        skip_lap = False 
        skip_start = False 
        if not hasattr(m, 'fields'):
            continue
        fields = m.fields
        #check for important data types
        mdata = {}
        for field in fields:# field = fields[0]
            if field.name in all_allowed_fields:
                if field.name=='timestamp':
                    mdata[field.name] = UTC.localize(field.value).astimezone(CST)
                else:
                    mdata[field.name] = field.value
        for rf in required_fields:
            if rf not in mdata:
                skip=True
        for lrf in lap_required_fields:
            if lrf not in mdata:
                skip_lap = True 
        for srf in start_required_fields:
            if srf not in mdata:
                skip_start = True
        if not skip:
            data.append(mdata)
        elif not skip_lap:
            lap_data.append(mdata)
        elif not skip_start:
            start_data.append(mdata)

    data_df = pd.DataFrame(data)
    lap_data_df = pd.DataFrame(lap_data)
    start_data_df = pd.DataFrame(start_data)
    
    return data_df, lap_data_df, start_data_df

#%%
# execute function
if __name__=='__main__':
    directory = 'C:\\Users\\BASPO\\Google Drive\\AeroLab_PowerPod\\2021-03-19'
    for (_, _, files) in os.walk(directory):
        files = [f for f in files if f.split('.')[-1].lower()=='fit']# f = files[0]
    
    if len(files)==1:
        data_df, lap_data_df, start_data_df = get_fit_dfs(
            fit_file_path = os.path.join(directory, files[0])
            )
    else:
        dict_of_data_dfs = {}
        dict_of_lap_data_dfs = {}
        dict_of_start_data_dfs = {}
        for f in files:# f = files[0]
            data_df, lap_data_df, start_data_df = get_fit_dfs(
                fit_file_path = os.path.join(directory, f)
                )
            dict_of_data_dfs[f.split('.')[0]] = data_df
            dict_of_lap_data_dfs[f.split('.')[0]] = lap_data_df
            dict_of_start_data_dfs[f.split('.')[0]] = start_data_df
        

