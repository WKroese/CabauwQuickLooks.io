#KNMI api data download and plot
#author: Willem Kroese
#date: 04-03-2025

# 	urn:xkdc:ds:nl.knmi::cesar_surface_meteo_la1_t10/v1.0/, https://dataplatform.knmi.nl/dataset/cesar-surface-meteo-la1-t10-v1-0
#   urn:xkdc:ds:nl.knmi::cesar_tower_meteo_la1_t10/v1.2/, https://dataplatform.knmi.nl/dataset/cesar-tower-meteo-la1-t10-v1-2
#

#TODO: https://cloudnet.fmi.fi/site/lutjewad
#https://cloudnet.fmi.fi/site/cabauw

#TODO: add cloudnet (also Lutjewad?), add pandora
#TODO: add automation, 
#TODO: add daily ims generation
#TODO: add drive sync, add folders/drive ims to site

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import logging
import os
import sys
import netCDF4 as nc
import requests
from datetime import datetime
from datetime import timedelta
import matplotlib.dates as mdates
from windrose import WindroseAxes

import config #contains api key

plt.style.use('my_style') #custom style

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(os.environ.get("LOG_LEVEL", logging.INFO))

class OpenDataAPI:
    def __init__(self, api_token: str):
        self.base_url = "https://api.dataplatform.knmi.nl/open-data/v1"
        self.headers = {"Authorization": api_token}

    def __get_data(self, url, params=None):
        return requests.get(url, headers=self.headers, params=params).json()

    def list_files(self, dataset_name: str, dataset_version: str, params: dict):
        return self.__get_data(
            f"{self.base_url}/datasets/{dataset_name}/versions/{dataset_version}/files",
            params=params,
        )

    def get_file_url(self, dataset_name: str, dataset_version: str, file_name: str):
        return self.__get_data(
            f"{self.base_url}/datasets/{dataset_name}/versions/{dataset_version}/files/{file_name}/url"
        )
def download_file_from_temporary_download_url(download_url, filename):
    try:
        with requests.get(download_url, stream=True) as r:
            r.raise_for_status()
            with open(filename, "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
    except Exception:
        logger.exception("Unable to download file using download URL")
        sys.exit(1)

    logger.info(f"Successfully downloaded dataset file to {filename}")

def download_recent(dataset_name, dataset_version,data_dir,instrument=None):
    api_key = config.key 
    logger.info(f"Fetching files of {dataset_name} version {dataset_version}")
    api = OpenDataAPI(api_token=api_key)

    begin = (datetime.today()-timedelta(hours=48)).strftime("%Y-%m-%dT%H:%M:%S+00:00")
    end = datetime.today().strftime("%Y-%m-%dT%H:%M:%S+00:00")

    params = {"maxKeys": 2, "orderBy": "created", "sorting": "desc", "begin": begin, "end": end}
    
    response = api.list_files(dataset_name, dataset_version, params)
    if "error" in response:
        logger.error(f"Unable to retrieve list of files: {response['error']}")
        sys.exit(1)

    for i in response["files"]: 
        print(i.get("filename"))
        current_file = i.get("filename")

        if instrument == 'ceilometer':
            #select only files from cabauw
            if '_06348_' in current_file:
                response = api.get_file_url(dataset_name, dataset_version, current_file)
                write_file = data_dir + current_file
                download_file_from_temporary_download_url(response["temporaryDownloadUrl"], write_file)
        else:
            response = api.get_file_url(dataset_name, dataset_version, current_file)
            write_file = data_dir + current_file
            download_file_from_temporary_download_url(response["temporaryDownloadUrl"], write_file)
        
def make_day_plots_archive(day):
    data_dir = './data/'
    fig_dir = './figures/archive/'
    os.makedirs(fig_dir,exist_ok=True)
    os.makedirs(fig_dir+day,exist_ok=True)
    make_plots(data_dir,fig_dir+day+'/',day)   
    

def check_new_day():
    '''check if new day has started, if so make plots of previous day and save to date folder'''
    today = datetime.today().date().strftime("%Y%m%d")
    with open('last_date.txt','r') as f:
        last_date = f.read()
    if today != last_date:
        make_day_plots_archive(last_date)
        with open('last_date.txt','w') as f:
            f.write(today)

def make_plots(data_dir,fig_dir,day=None):
    today = datetime.today().date().strftime("%Y%m%d")
    yesterday= (datetime.today()-timedelta(days=1)).date().strftime("%Y%m%d")

    now = pd.Timestamp.now()

    if day is not None:
        ds = nc.MFDataset([data_dir+'cesar_tower_meteo_la1_t10_v1.2_'+day+'.nc'])
        date = ds['date'][:]
        t_hour = ds['time'][:]
        t = pd.to_datetime(date,format='%Y%m%d')+pd.to_timedelta(t_hour,'h')
        mask = t<now #hacky fix for now
        xmin, xmax = t[0],t[-1]
    else:
        ds = nc.MFDataset([data_dir+'cesar_tower_meteo_la1_t10_v1.2_'+yesterday+'.nc',data_dir+'cesar_tower_meteo_la1_t10_v1.2_'+today+'.nc'])
        date = ds['date'][:]
        t_hour = ds['time'][:]
        t = pd.to_datetime(date,format='%Y%m%d')+pd.to_timedelta(t_hour,'h')
        mask = (t<now) & (t>now-timedelta(hours=36))
        xmin, xmax = now-timedelta(hours=36),now

    T = ds['TA'][mask,:]-273.15
    TD = ds['TD'][mask,:]-273.15
    F = ds['F'][mask,:]
    vis = ds['ZMA'][mask,:]
    D = ds['D'][mask,:]
    RH = 100*(np.exp((17.625*TD)/(243.04+TD))/np.exp((17.625*T)/(243.04+T))) 

    t = t[mask]

    plt.figure(figsize=(7,4))
    plt.plot(t,RH[:,-1])
    plt.title('relative humidity at 2 m \n'+'$RH_{min}$'+'={:.1f}, '.format(np.min(RH[:,-1]))+'$RH_{max}$'+'={:.1f}'.format(np.max(RH[:,-1])))
    plt.xlabel('time UTC [hours]')
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("%H"))
    plt.ylabel('RH [%]')
    plt.ylim(25,105)
    plt.xlim(xmin,xmax)
    plt.savefig(fig_dir+'rh.png')

    plt.figure(figsize=(7,4))
    plt.plot(t,T[:,-1],label='T')
    plt.plot(t,TD[:,-1],label='dewpoint')
    plt.title('temperature and dewpoint at 2m \n'+'$T_{min}$'+'={:.1f}, '.format(np.min(T[:,-1]))+'$T_{max}$'+'={:.1f}'.format(np.max(T[:,-1])))
    plt.xlabel('time UTC [hours]')
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("%H"))
    plt.ylabel('T [$^\circ$C]')
    plt.legend()
    plt.xlim(xmin,xmax)
    plt.savefig(fig_dir+'temp.png')

    plt.figure(figsize=(7,4))
    plt.plot(t,F[:,-2],label='10m')
    plt.plot(t,F[:,2],label='80m')
    plt.plot(t,F[:,0],label='200m')
    plt.title('wind speed')
    plt.xlabel('time UTC [hours]')
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("%H"))
    plt.ylabel('uv [m/s]')
    plt.legend()
    plt.xlim(xmin,xmax)
    plt.savefig(fig_dir+'wind.png')

    plt.figure(figsize=(7,4))
    plt.scatter(t,D[:,-2],label='10m',s=7)
    plt.scatter(t,D[:,2],label='80m',s=7)
    plt.scatter(t,D[:,0],label='200m',s=7)
    plt.title('wind direction')
    plt.xlabel('time UTC [hours]')
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("%H"))
    plt.ylabel('degrees')
    plt.ylim(-5,365)
    plt.legend()
    plt.xlim(xmin,xmax)
    plt.savefig(fig_dir+'wind_dir.png')

    ax = WindroseAxes.from_ax(figsize=(6,6))
    ax.set_title('windrose at 10 m',size=14)
    ax.bar(D[:,-2], F[:,-2], normed=True, opening=0.9, edgecolor="white")
    ax.set_legend(loc="best",fontsize=13)
    ax.set_ylim(0,ax.get_rmax()*1.1)
    for i in ax.get_xticklabels():
        plt.setp(i, fontsize=14)
    plt.xlim(xmin,xmax)
    plt.savefig(fig_dir+'windrose.png')

    plt.figure(figsize=(7,4))
    plt.plot(t,vis[:,-1]*1e-3,label='2m')
    plt.plot(t,vis[:,2]*1e-3,label='80m')
    plt.plot(t,vis[:,0]*1e-3,label='200m')
    plt.hlines(1,t[0],t[-1],linestyles='dashed',color='black',alpha=0.5)
    plt.text(t[-1]-pd.to_timedelta(4,'h'),1.5,'fog limit',color='black',alpha=0.7)
    plt.title('visibility')
    plt.xlabel('time UTC [hours]')
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("%H"))
    plt.ylabel('[km]')
    plt.legend()
    plt.xlim(xmin,xmax)
    plt.savefig(fig_dir+'visibility.png')


    if day is not None:
        ds = nc.MFDataset([data_dir+'cesar_surface_meteo_la1_t10_v1.0_'+day+'.nc'])
        date = ds['date'][:]
        t_hour = ds['time'][:]
        t = pd.to_datetime(date,format='%Y%m%d')+pd.to_timedelta(t_hour,'h')
        mask = t<now #hacky fix for now
    else:
        ds = nc.MFDataset([data_dir+'cesar_surface_meteo_la1_t10_v1.0_'+yesterday+'.nc',data_dir+'cesar_surface_meteo_la1_t10_v1.0_'+today+'.nc'])
        date = ds['date'][:]
        t_hour = ds['time'][:]
        t = pd.to_datetime(date,format='%Y%m%d')+pd.to_timedelta(t_hour,'h')
        mask = (t<now) & (t>now-timedelta(hours=36))

    P = ds['P0'][mask]
    SWD = ds['SWD'][mask]
    rain = ds['RAIN'][mask]
    
    t = t[mask]

    plt.figure(figsize=(7,4))
    plt.plot(t,P)
    plt.title('surface pressure')
    plt.xlabel('time UTC [hours]')
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("%H"))
    plt.ylabel('pressure [hPa]')
    # plt.ylim(900,1100)
    plt.xlim(xmin,xmax)
    plt.savefig(fig_dir+'pressure.png')

    plt.figure(figsize=(7,4))
    plt.plot(t,SWD)
    plt.title('downward solar radiation')
    plt.xlabel('time UTC [hours]')
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("%H"))
    plt.ylabel('flux [W m$^{-2}$]')
    plt.xlim(xmin,xmax)
    plt.savefig(fig_dir+'swd.png')

    plt.figure(figsize=(7,4))
    plt.plot(t,rain)
    plt.title('precipitation\n total over last 24h: {:.1f} mm'.format(np.sum(rain[::-1][0:240])))
    plt.xlabel('time UTC [hours]')
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("%H"))
    plt.ylabel('[mm]')
    if np.max(rain)<0.1:
        plt.ylim(0,0.1)
    plt.xlim(xmin,xmax)
    plt.savefig(fig_dir+'rain.png')

if __name__ == '__main__':
    fig_dir = './figures/'
    data_dir = './data/'

    dataset_name = 'cesar_tower_meteo_la1_t10' #unvalidated
    dataset_version = 'v1.2'

    download_recent(dataset_name, dataset_version, data_dir)
    
    dataset_name = 'cesar_surface_meteo_la1_t10'
    dataset_version = 'v1.0'

    download_recent(dataset_name, dataset_version, data_dir)

    check_new_day()

    make_plots(data_dir,fig_dir)
