import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
import torch
import torch.nn as nn
from utils import *
from model import *
from train import *
import argparse
import glob
import os


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--train", help="start training", action="store_true")
    parser.add_argument("--test", help="start testing", action="store_true")
    parser.add_argument("--store_data", help="store the data to folder", action="store_true")
    parser.add_argument("--epochs", help="number of epochs", type=int, default=500)
    parser.add_argument("--batch_size", help="batch size", type=int, default=128)
    parser.add_argument("--win_len", help="window length", type=int, default=12)
    parser.add_argument("--hidden_size", help="hidden size", type=int, default=128)
    parser.add_argument("--lr", help="learning rate", type=float, default=1e-4)
    parser.add_argument("--feature_set", help="which feature subset to use (1~4)", type=int, default=4)
    parser.add_argument("--target_gas", help="gas for training", type=str, default='O3')
    parser.add_argument("--model_type", help="model type", required=True, type=str, default='GRU')
    args = parser.parse_args()
    
    # set the hyperparameters
    num_epochs = args.epochs
    batch_size = args.batch_size
    win_len = args.win_len
    hidden_size = args.hidden_size
    lr = args.lr
    target_gas = args.target_gas
    feature_subset = args.feature_set
    model_type = args.model_type

    # set random seed for reproducibility
    setSeed(0)

    # process the raw data and store the processed to folder
    if args.store_data == True:
        storeData()

    # read the data
    spec_data, sgx_data, ref_data = readData()
    data = pd.concat([ref_data, sgx_data, spec_data], axis = 1)
    data = data.reindex(data.index, fill_value=np.nan)


    # feature selection
    if feature_subset == 1:
        columns = ['REF-AMB_TEMP', 'REF-RH', f'SGX-{target_gas}', f'REF-{target_gas}']
    elif feature_subset == 2:
        columns = ['REF-AMB_TEMP', 'REF-RH', f'SPEC-{target_gas}', f'REF-{target_gas}']
    elif feature_subset == 3:
        columns = ['REF-AMB_TEMP', 'REF-RH', f'SPEC-{target_gas}', f'SGX-{target_gas}', f'REF-{target_gas}']
    elif feature_subset == 4:
        columns =  ['SPEC-O3', 'SGX-CO', 'SGX-NO2', 'SPEC-SO2', 'REF-AMB_TEMP', 'REF-RH'] + [f'REF-{target_gas}'] # O3
        # columns =  ['SGX-CO', 'REF-AMB_TEMP', 'SPEC-SO2'] + [f'REF-{target_gas}'] # CO
    elif feature_subset == 5:
        columns = sfs(data = data,
                        win_len = win_len,
                        hidden_size = hidden_size,
                        num_epochs = num_epochs,
                        batch_size = batch_size, 
                        target_gas = target_gas,
                        lr = lr)
        columns += [f'REF-{target_gas}']

    # Missing value and outlier removal
    data.dropna(subset=columns, inplace=True)
    data = data.abs()
    
    # Data Division
    dates = data.index
    print('The total valid samples:', len(dates))

    # Keep the dates ealier than 2023-05-31 23:00:00
    dates = dates[dates <= pd.to_datetime('2023-06-27 23:00:00')]

    batches = np.array_split(dates, 8)
    # Select the first 2 batches as source data
    train = batches[0].append(batches[1])
    # Select target data
    val = batches[6]
    # Select test data
    test = batches[7]
    # print('Train size: {:d}, Validation size: {:d}, Test size: {:d}'.format(len(train), len(val), len(test)))

    # prepare data
    train_data = data.loc[train, columns]
    val_data = data.loc[val, columns]
    test_data = data.loc[test, columns]

    # Min Max Scaler
    scaler = MinMaxScaler()
    scaler.fit(data.loc[train, columns])
    train_data = scaler.transform(train_data)
    val_data = scaler.transform(val_data)
    test_data = scaler.transform(test_data)
    
    # Transpose
    train_data = np.transpose(train_data)
    val_data = np.transpose(val_data)
    test_data = np.transpose(test_data)

    # create sequences
    use_consecutive = True
    X_train, y_train, train = create_sequences(train_data, win_len, train, use_consecutive = use_consecutive)
    X_val, y_val, val = create_sequences(val_data, win_len, val, use_consecutive = use_consecutive)
    X_test, y_test, test = create_sequences(test_data, win_len, test, use_consecutive = use_consecutive)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using {device} device")

    # model initialization
    input_features = len(columns) - 1
    
    if model_type == 'GRU':
        model = GRU(input_features=input_features, hidden_size=hidden_size, win_len=win_len).to(device)
    elif model_type == 'Transformer':
        model = Encoder(d_model=input_features, nhead=1, num_layers=2).to(device)
    
    # loss & optimizer
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    #  training
    if args.train == True:
        
        training(model, X_train, y_train, X_val, y_val, num_epochs, batch_size, optimizer, criterion, device)
        
    # Testing
    if args.test == True:

        # Find the latest model in the model directory
        list_of_files = glob.glob('model/*.pth')
        
        # Get the latest model
        
        latest_model = max(list_of_files, key=os.path.getctime)

        # Load the latest model
        model.load_state_dict(torch.load(latest_model))

        # Set the model to evaluation model
        model.eval()

        # # Calculate the loss for the train set
        # y_train_cal = model(X_train.to(device))[:, -1, :]
        # print('Train')
        # score(y_train[:, -1, :], y_train_cal.cpu().detach().numpy())
        # visualize_result(y_train.numpy()[:, -1, :], y_train_cal.cpu().detach().numpy(), train, f'{target_gas} Train {model_type}')
        
        # # Calculate the loss for the val set
        # y_val_cal = model(X_val.to(device))[:, -1, :]
        # print('Val')
        # score(y_val[:, -1, :], y_val_cal.cpu().detach().numpy())
        # visualize_result(y_val.numpy()[:, -1, :], y_val_cal.cpu().detach().numpy(), val, f'{target_gas} Val {model_type}')

        # # Calculate the loss for the test set
        # y_test_cal = model(X_test.to(device))[:, -1, :]
        # print('Test')
        # score(y_test[:, -1, :], y_test_cal.cpu().detach().numpy())
        # visualize_result(y_test.numpy()[:, -1, :], y_test_cal.cpu().detach().numpy(), test, f'{target_gas} Test {model_type}')
        
        # Calculate the loss for the train set
        y_train_cal = model(X_train.to(device))
        print('Train')
        score(y_train, y_train_cal.cpu().detach().numpy())
        visualize_result(y_train.numpy(), y_train_cal.cpu().detach().numpy(), train, f'{target_gas} Train {model_type}')
        
        # Calculate the loss for the val set
        y_val_cal = model(X_val.to(device))
        print('Val')
        score(y_val, y_val_cal.cpu().detach().numpy())
        visualize_result(y_val.numpy(), y_val_cal.cpu().detach().numpy(), val, f'{target_gas} Val {model_type}')

        # Calculate the loss for the test set
        y_test_cal = model(X_test.to(device))
        print('Test')
        score(y_test, y_test_cal.cpu().detach().numpy())
        visualize_result(y_test.numpy(), y_test_cal.cpu().detach().numpy(), test, f'{target_gas} Test {model_type}')



        








