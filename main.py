#KNMI api data download and plot
#author: Willem Kroese
#date: 04-03-2025

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
from dateutil import tz

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

def download_recent(dataset_name, dataset_version,data_dir):
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
        response = api.get_file_url(dataset_name, dataset_version, current_file)
        
        write_file = data_dir + current_file

        download_file_from_temporary_download_url(response["temporaryDownloadUrl"], write_file)


def make_plots(data_dir,fig_dir):
    today = datetime.today().date().strftime("%Y%m%d")
    yesterday= (datetime.today()-timedelta(days=1)).date().strftime("%Y%m%d")

    ds = nc.MFDataset([data_dir+'cesar_tower_meteo_la1_t10_v1.2_'+yesterday+'.nc',data_dir+'cesar_tower_meteo_la1_t10_v1.2_'+today+'.nc'])

    date = ds['date'][:]
    t_hour = ds['time'][:]
    t = pd.to_datetime(date,format='%Y%m%d')+pd.to_timedelta(t_hour,'h')
    now = pd.Timestamp.now()
    mask = (t<now) & (t>now-timedelta(hours=36))

    T = ds['TA'][mask,:]-273.15
    TD = ds['TD'][mask,:]-273.15
    F = ds['F'][mask,:]
    vis = ds['ZMA'][mask,:]
    D = ds['D'][mask,:]
    RH = 100*(np.exp((17.625*TD)/(243.04+TD))/np.exp((17.625*T)/(243.04+T))) 

    t = t[mask]

    plt.figure(figsize=(7,4))
    plt.plot(t,RH[:,-1])
    plt.title('relative humidity at 2 m \n'+'$RH_{min}$'+'={:.1f}, '.format(np.min(RH))+'$RH_{max}$'+'={:.1f}'.format(np.max(RH)))
    plt.xlabel('time UTC [hours]')
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("%H"))
    plt.ylabel('RH [%]')
    plt.ylim(25,105)
    plt.savefig(fig_dir+'rh.png')

    plt.figure(figsize=(7,4))
    plt.plot(t,T[:,-1],label='T')
    plt.plot(t,TD[:,-1],label='dewpoint')
    plt.title('temperature and dewpoint at 2m \n'+'$T_{min}$'+'={:.1f}, '.format(np.min(T))+'$T_{max}$'+'={:.1f}'.format(np.max(T)))
    plt.xlabel('time UTC [hours]')
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("%H"))
    plt.ylabel('T [$^\circ$C]')
    plt.legend()
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
    plt.savefig(fig_dir+'wind_dir.png')

    ax = WindroseAxes.from_ax(figsize=(6,6))
    ax.set_title('windrose at 10 m',size=14)
    ax.bar(D[:,-2], F[:,-2], normed=True, opening=0.9, edgecolor="white")
    ax.set_legend(loc="best",fontsize=13)
    ax.set_ylim(0,ax.get_rmax()*1.1)
    for i in ax.get_xticklabels():
        plt.setp(i, fontsize=14)
    plt.savefig(fig_dir+'windrose.png')

    plt.figure(figsize=(7,4))
    plt.plot(t,vis[:,-1]*1e-3,label='2m')
    plt.plot(t,vis[:,2]*1e-3,label='80m')
    plt.plot(t,vis[:,0]*1e-3,label='200m')
    plt.title('visibility')
    plt.xlabel('time UTC [hours]')
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("%H"))
    plt.ylabel('vis [km]')
    plt.legend()
    plt.savefig(fig_dir+'visibility.png')
    

if __name__ == '__main__':
    fig_dir = './figures/'
    data_dir = './data/'

    dataset_name = 'cesar_tower_meteo_la1_t10' #unvalidated
    dataset_version = 'v1.2'

    download_recent(dataset_name, dataset_version, data_dir)
    make_plots(data_dir,fig_dir)
