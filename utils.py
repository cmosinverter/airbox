import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import math
import os
import glob
import json
import torch
from datetime import datetime, timedelta
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error


sgx_uuid = '04311cdc-a6a5-4d5e-916e-4e75076a8f0a'
spec_uuid = 'c92494ea-5e44-4507-8776-984705d1fd2b'

pollutant = ['O3', 'CO', 'NO2', 'SO2', 'PM2.5', 'PM10']
environment = ['RH', 'AMB_TEMP']


sgx_sensor_dict = {'c1_4#ec_na#0': 'O3',
                   'c1_0#ec_na#0': 'CO',
                   'c1_3#ec_na#0': 'NO2',
                   'c1_5#ec_na#0': 'SO2'}

unit_table = {'O3': 'ppb', 
              'CO': 'ppm',
              'NO2': 'ppb',
              'SO2': 'ppb'}

def setSeed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    return

def getData(file_path, device):
    try:
        df = pd.read_csv(file_path, usecols = ['measure_time', 'sensors'])
    except:
        return None
    df['measure_time'] = list(map(lambda x: datetime.strptime(x[:13], '%Y-%m-%dT%H') + timedelta(hours=8), df['measure_time']))
    
    if device == 'SGX':
        start = 7
    if device == 'SPEC':
        start = 6
        
    for i, gas in enumerate(pollutant[:4]):
        df[device + '-' + gas] = list(map(lambda x: json.loads(x)[start+i]['value'], df['sensors']))
    df = df.drop('sensors', axis=1)
    df = df.sort_values('measure_time')
    return df

def storeData():

    ref = pd.concat([pd.read_csv(path, na_values='x') for path in glob.glob(os.path.join('D:/NTHU/airbox-data/3rdData', '2023', '*.csv'))])
    ref['"monitordate"'] = ref['"monitordate"'].str[:13]
    ref['"monitordate"'] = list(map(lambda x: datetime.strptime(x, '%Y-%m-%d %H'), ref['"monitordate"']))
    ref = ref.drop_duplicates()
    ref = ref.set_index('"monitordate"')
    ref.index.name = None
    ref_data = pd.concat([ref.loc[ref['"itemengname"'] == p, ['"concentration"']] for p in pollutant + environment], axis = 1)
    ref_data.columns = ['REF-' + p for p in pollutant + environment]
    ref_data = ref_data.dropna()
    ref_data.to_csv('data/reference.csv')

    paths = [p for p in glob.glob(os.path.join('D:/NTHU/airbox-data/deviceData', '*', '*.csv')) if sgx_uuid in p]
    sgx_data = pd.concat([getData(path, 'SGX') for path in paths if getData(path, 'SGX') is not None])
    print(f'Number of Samples: {len(sgx_data)}')
    sgx_data = sgx_data.groupby('measure_time').mean()
    sgx_data.drop(sgx_data.loc[(sgx_data.index >= '2023-03-30 14:00:00') & (sgx_data.index <= '2023-04-12 10:00:00')].index, inplace=True)
    sgx_data.to_csv('data/sgx.csv')

def readData():

    sgx = pd.read_csv('data/sgx.csv', index_col = [0])
    sgx.index = pd.to_datetime(sgx.index)
    ref = pd.read_csv('data/reference.csv', index_col = [0])
    ref.index = pd.to_datetime(ref.index)
    return sgx, ref

def score(y_pred, y_true):
    print('***R2 Score: {:.2f}'.format(r2_score(y_pred, y_true)))
    print('***RMSE: {:.4f}'.format(math.sqrt(mean_squared_error(y_pred, y_true))))
    return r2_score(y_pred, y_true), math.sqrt(mean_squared_error(y_pred, y_true))

def visualize_result(y_true, y_pred, dates, title = ""):
    x = range(len(dates))
    fig, ax = plt.subplots(1, 2, figsize = (16, 6))
    if title != "":
        fig.suptitle(title)
    else:
        fig.suptitle('Result')
    ax[0].plot(x, y_true, label = 'Reference')
    ax[0].plot(x, y_pred, label = 'Calibrated')
    ax[0].set_xticks(np.arange(0, len(dates), len(dates)//8))
    ax[0].set_xticklabels(labels = dates[::len(dates)//8].date, rotation = 20)
    ax[0].legend()

    ax[1].scatter(y_pred, y_true)
    ax[1].plot([0, 1.2], [0, 1.2], 'g--', linewidth=2, markersize=12, label = 'ideal')
    ax[1].legend()
    ax[1].set_xlabel('Predict')
    ax[1].set_ylabel('Actual')

    # plot the regression line
    reg = LinearRegression().fit(y_pred.reshape(-1, 1), y_true.reshape(-1, 1))
    intercept = reg.intercept_
    slope = reg.coef_[0]
    y = intercept + slope * y_pred.reshape(-1, 1)
    ax[1].plot(y_pred.reshape(-1, 1), y, color='red', label='Regression Line')
    # Add the linear equation as a text annotation
    equation_text = f'y = {slope.item():.2f}x + {intercept.item():.2f}'
    ax[1].text(np.min(y_pred.reshape(-1, 1)), np.max(y), equation_text, fontsize=12, verticalalignment='top')

    plt.tight_layout()
    if title != "":
        plt.savefig(f'fig/{title}', dpi=300, bbox_inches='tight')
    plt.show()

    return slope.item(), intercept.item()

def create_sequences(data, window_len):
    xs = []
    ys = []

    for i in range(data.shape[1]-window_len+1):
        x = data[:-1, i:i+window_len]
        y = data[-1:, i+window_len-1]
        xs.append(x)
        ys.append(y)

    return torch.unsqueeze(torch.tensor(np.stack(xs), dtype=torch.float32), dim=1), torch.tensor(np.stack(ys), dtype=torch.float32)

