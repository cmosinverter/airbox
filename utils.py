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
import torch.nn as nn
from model import *
from train import *
from datetime import datetime, timedelta
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import MinMaxScaler


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
    ref_data.to_csv('data/reference.csv')

    paths = [p for p in glob.glob(os.path.join('D:/NTHU/airbox-data/deviceData', '*', '*.csv')) if sgx_uuid in p]
    sgx_data = pd.concat([getData(path, 'SGX') for path in paths if getData(path, 'SGX') is not None])
    print(f'Number of Samples: {len(sgx_data)}')
    sgx_data = sgx_data.groupby('measure_time').mean()
    # Drop rows with all zero values
    sgx_data = sgx_data[~(sgx_data == 0).all(axis=1)]
    
    sgx_data.to_csv('data/sgx.csv')

    paths = [p for p in glob.glob(os.path.join('D:/NTHU/airbox-data/deviceData', '*', '*.csv')) if spec_uuid in p]
    spec_data = pd.concat([getData(path, 'SPEC') for path in paths if getData(path, 'SPEC') is not None])
    print(f'Number of Samples: {len(spec_data)}')
    spec_data = spec_data.groupby('measure_time').mean()
    # Drop rows with all zero values
    spec_data = spec_data[~(spec_data == 0).all(axis=1)]
    
    spec_data.to_csv('data/spec.csv')

def readData():
    spec = pd.read_csv('data/spec.csv', index_col = [0])
    spec.index = pd.to_datetime(spec.index)
    sgx = pd.read_csv('data/sgx.csv', index_col = [0])
    sgx.index = pd.to_datetime(sgx.index)
    ref = pd.read_csv('data/reference.csv', index_col = [0])
    ref.index = pd.to_datetime(ref.index)
    return spec, sgx, ref

def score(y_pred, y_true):
    print('***R2 Score: {:.2f}'.format(r2_score(y_pred, y_true)))
    print('***RMSE: {:.4f}'.format(math.sqrt(mean_squared_error(y_pred, y_true))))
    return r2_score(y_pred, y_true), math.sqrt(mean_squared_error(y_pred, y_true))

def visualize_result(y_true, y_pred, dates, title=""):
    x = range(len(dates))
    fig, ax = plt.subplots(1, 2, figsize=(16, 6))
    if title != "":
        fig.suptitle(title)
    else:
        fig.suptitle('Result')
    ax[0].plot(x, y_true, label='Reference')
    ax[0].plot(x, y_pred, label='Calibrated')
    ax[0].set_xticks(np.arange(0, len(dates), len(dates) // 8))
    ax[0].set_xticklabels(labels=list(map(lambda x: str(x)[:10], dates[::len(dates) // 8])), rotation=20)
    ax[0].legend()

    ax[1].scatter(y_pred, y_true)
    ax[1].plot([0, 0.6], [0, 0.6], 'g--', linewidth=2, markersize=12, label='ideal')
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
    plt.savefig(f'fig/{title}', dpi=300, bbox_inches='tight')
    # plt.show()

def visualize_result_trend(y_true, y_pred, dates, title=""):
    x = range(len(dates))
    fig = plt.figure(figsize=(16, 6))
    plt.plot(x, y_true, label='Reference')
    plt.plot(x, y_pred, label='Calibrated')
    plt.xlabel('Date')
    plt.ylabel('Concentration (MinMaxScaled)')
    plt.xticks(np.arange(0, len(dates), len(dates) // 8))
    plt.gca().set_xticklabels(list(map(lambda x: str(x)[:10], dates[::len(dates) // 8])), rotation=20)
    plt.legend()

    plt.tight_layout()
    plt.savefig(f'fig/{title}_trend', dpi=300, bbox_inches='tight')
    # plt.show()

def visualize_result_scatter(y_true, y_pred, dates, title=""):
    x = range(len(dates))
    fig = plt.figure(figsize=(8, 6))
    plt.scatter(y_pred, y_true)
    plt.plot([0, 0.6], [0, 0.6], 'g--', linewidth=2, markersize=12, label='ideal')
    plt.xlabel('Predict')
    plt.ylabel('Actual')
    # plot the regression line
    reg = LinearRegression().fit(y_pred.reshape(-1, 1), y_true.reshape(-1, 1))
    intercept = reg.intercept_
    slope = reg.coef_[0]
    y = intercept + slope * y_pred.reshape(-1, 1)
    plt.plot(y_pred.reshape(-1, 1), y, color='red', label='Regression Line')
    plt.legend()
    # Add the linear equation as a text annotation
    equation_text = f'y = {slope.item():.2f}x + {intercept.item():.2f}'
    plt.text(np.min(y_pred.reshape(-1, 1)), np.max(y), equation_text, fontsize=12, verticalalignment='top')

    plt.tight_layout()
    if title != "":
        plt.savefig(f'fig/{title}_scatter', dpi=300, bbox_inches='tight')
    else:
        plt.savefig('fig/result_scatter', dpi=300, bbox_inches='tight')
    plt.show()

def is_consecutive(time_list):
    for i in range(len(time_list)-1):
        time_diff = time_list[i+1] - time_list[i]
        if time_diff != pd.Timedelta(hours=1):
            return False
    return True


def create_sequences(data, window_len, dates, use_consecutive=False):
    xs = []
    ys = []
    new_dates = []
    for i in range(data.shape[1]-window_len+1):
        if use_consecutive:
            if is_consecutive(dates[i:i+window_len]):
                x = data[:-1, i:i+window_len]
                y = data[-1:, i+window_len-1]
                xs.append(x)
                ys.append(y)
                new_dates.append(dates[i+window_len-1])
            else:
                continue
        else:
            x = data[:-1, i:i+window_len]
            y = data[-1:, i+window_len-1]
            xs.append(x)
            ys.append(y)
            new_dates.append(dates[i+window_len-1])
    
    return torch.unsqueeze(torch.tensor(np.stack(xs), dtype=torch.float32), dim=1), torch.tensor(np.stack(ys), dtype=torch.float32), new_dates

def sfs(data, win_len, kernel_width, hidden_size, num_epochs, batch_size, lr, target_gas):

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    criterion = nn.MSELoss()
    # Prepare the data
    features = list(data.columns[6:])

    # Create a set to store selected features
    selected_features = set(range(len(features)))

    remove_sequence = []

    while len(selected_features) > 3:
        scores = []
        
        for i in range(len(selected_features)):
            # Set the random seed
            setSeed(0)

            # Remove one feature from the feature set each time
            feature_subset_idx = list(selected_features - {list(selected_features)[i]})
            print('Current training: ', feature_subset_idx)

            # Create a new training set and validation set
            tmp_data = data.dropna(subset = [features[idx] for idx in feature_subset_idx] + [f'REF-{target_gas}'])
            tmp_data = tmp_data.abs()

            # Split the data into training and test sets
            dates = tmp_data.index

            # Keep the dates ealier than 2023-05-31 23:00:00
            dates = dates[dates <= pd.to_datetime('2023-05-31 23:00:00')]

            split_date_1 = pd.to_datetime('2023-03-31 23:00:00')
            split_date_2 = pd.to_datetime('2023-04-30 23:00:00')

            train = dates[dates <= split_date_1]
            val = dates[(dates > split_date_1) & (dates <= split_date_2)]

            train_data = tmp_data.loc[train, [features[idx] for idx in feature_subset_idx] + [f'REF-{target_gas}']]
            val_data = tmp_data.loc[val, [features[idx] for idx in feature_subset_idx] + [f'REF-{target_gas}']]

            # Scaler
            scaler = MinMaxScaler()
            scaler.fit(train_data)
            train_data = np.transpose(scaler.transform(train_data))
            val_data = np.transpose(scaler.transform(val_data))
            # Create sequences
            X_train, y_train, train = create_sequences(train_data, win_len, train, use_consecutive = False)
            X_val, y_val, val = create_sequences(val_data, win_len, val, use_consecutive = False)
            

            # Select the corresponding features
            input_features = X_train.shape[2]

            # Create a new model
            model = CNN_GRU(kernel_width=kernel_width, input_features=input_features, hidden_size=hidden_size).to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=lr)

            # Train the model
            training(model, X_train, y_train, X_val, y_val, num_epochs=num_epochs, batch_size=batch_size, optimizer=optimizer, criterion=criterion, device=device)

            # Make predictions & score
            y_val_cal = model(X_val.to(device))
            r2, rmse = score(y_val, y_val_cal.cpu().detach().numpy())
            scores.append(r2)


        least_important_feature = list(selected_features)[np.argmax(scores)]
        selected_features.remove(least_important_feature)
        remove_sequence.append((features[least_important_feature], round(max(scores), 2)))
        print('The least important feature: ', features[least_important_feature], round(max(scores), 2))
            
    print('The selected features: ', [features[idx] for idx in selected_features])
    print('The removed sequence: ', remove_sequence)
    return [features[idx] for idx in selected_features]