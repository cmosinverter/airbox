import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from datetime import datetime, timedelta
from sklearn.preprocessing import MinMaxScaler
import torch
import torch.nn as nn
from utils import *
from model import *
import argparse
import glob
import os
import datetime
from datetime import datetime
from dataloader import SesnorDataset
from torch.utils.data import DataLoader

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--train", help="start training", action="store_true")
    parser.add_argument("--test", help="start testing", action="store_true")
    parser.add_argument("--store_data", help="store the data to folder", action="store_true")
    parser.add_argument("--epochs", help="number of epochs", type=int, default=100)
    parser.add_argument("--batch_size", help="batch size", type=int, default=128)
    parser.add_argument("--win_len", help="window length", type=int, default=24)
    parser.add_argument("--kernel_width", help="kernel width", type=int, default=4)
    parser.add_argument("--hidden_size", help="hidden size", type=int, default=64)
    parser.add_argument("--lr", help="learning rate", type=float, default=1e-4)
    parser.add_argument("--alpha", help="alpha", type=float, default=1)
    parser.add_argument("--feature_set", help="which feature subset to use (1~4)", type=int, default=4)
    parser.add_argument("--target_gas", help="gas for training", type=str, default='CO')
    args = parser.parse_args()
    
    # set the hyperparameters
    num_epochs = args.epochs
    batch_size = args.batch_size
    win_len = args.win_len
    kernel_width = args.kernel_width
    hidden_size = args.hidden_size
    lr = args.lr
    alpha = args.alpha
    target_gas = args.target_gas
    feature_subset = args.feature_set

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
        columns =  [ 'SGX-CO','REF-AMB_TEMP', 'SGX-SO2'] + [f'REF-{target_gas}']
    elif feature_subset == 5:
        columns = sfs(data = data,
                        win_len = win_len,
                        kernel_width = kernel_width,
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

    # Keep the dates ealier than 2023-06-27 23:00:00
    dates = dates[dates <= pd.to_datetime('2023-06-27 23:00:00')]
    
    # Split the data into 8 batches
    batches = np.array_split(dates, 8)

    # Select the first 2 batches as source data
    source = batches[0].append(batches[1])
    # Select target data
    target = batches[6]
    # Select test data
    test = batches[7]
    
    # Scaler for normalization
    source_data = data.loc[source, columns]
    scaler = MinMaxScaler()
    scaler.fit(source_data)
    
    # Create dataloader
    source_dataset = SesnorDataset(subset=columns, dates=source, scaler=scaler, win_len=win_len, use_consecutive=False, source=True)
    target_dataset = SesnorDataset(subset=columns, dates=target, scaler=scaler, win_len=win_len, use_consecutive=False, source=False)
    test_dataset   = SesnorDataset(subset=columns, dates=test  , scaler=scaler, win_len=win_len, use_consecutive=False, source=False)
    
    source_dataloader = DataLoader(source_dataset, batch_size=batch_size, shuffle=True)
    target_dataloader = DataLoader(target_dataset, batch_size=batch_size, shuffle=True)
    test_dataloader   = DataLoader(test_dataset  , batch_size=batch_size, shuffle=True)
    
    #  checking device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using {device} device")

    # model initialization
    input_features = len(columns)-1
    model = DAN(kernel_width=kernel_width, input_features=input_features, hidden_size=hidden_size).to(device)

    # loss & optimizer
    regression_criterion = nn.MSELoss()
    domian_criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    #  training
    if args.train == True:
        
        timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        
        for epoch in range(num_epochs):
            
            len_dataloader = min(len(source_dataloader), len(target_dataloader))
            data_source_iter = iter(source_dataloader)
            data_target_iter = iter(target_dataloader)
            
            total_s_regression_loss = 0.0
            total_s_domain_loss = 0.0
            total_t_domain_loss = 0.0
            
            i = 0
            while i < len_dataloader:
                p = float(i + epoch * len_dataloader) / epoch / len_dataloader
                alpha = 2. / (1. + np.exp(-10 * p)) - 1
                
                X_source, y_source, domain_source = data_source_iter.next()
                X_source, y_source, domain_source = X_source.to(device), y_source.to(device), domain_source.to(device)
                
                class_output, domain_output = model(X_source, alpha)
                
                
            # Print the loss for every epochs
            print(f'Epoch [{epoch+1}/{num_epochs}], Regression Loss: {regression_loss.item():.4f}, Domain Loss: {domain_loss.item():.4f}')
        model_name = f'dann_{timestamp}'
        # Save the model with the new name
        torch.save(model.state_dict(), f'model/{model_name}.pth')
    # Testing
    if args.test == True:

        # Find the latest model in the model directory
        list_of_files = glob.glob('model/*.pth')
        latest_model = max(list_of_files, key=os.path.getctime)

        # Load the latest model
        model.load_state_dict(torch.load(latest_model))

        # Set the model to evaluation model
        model.eval()

        # Calculate the loss for the source set
        y_source_cal, _ = model(X_source.to(device), alpha)
        print('Source')
        score(y_source, y_source_cal.cpu().detach().numpy())
        visualize_result(y_source.numpy(), y_source_cal.cpu().detach().numpy(), source, f'{target_gas} Source DANN')
        visualize_result_trend(y_source.numpy(), y_source_cal.cpu().detach().numpy(), source, f'{target_gas} Source DANN')

        # Calculate the loss for the target set
        y_target_cal, _ = model(X_target.to(device), alpha)
        print('Target')
        score(y_target, y_target_cal.cpu().detach().numpy())
        visualize_result(y_target.numpy(), y_target_cal.cpu().detach().numpy(), target, f'{target_gas} Target DANN')
        visualize_result_trend(y_target.numpy(), y_target_cal.cpu().detach().numpy(), target, f'{target_gas} Target DANN')

        # Calculate the loss for the test set
        y_test_cal, _ = model(X_test.to(device), alpha)
        print('Test')
        score(y_test, y_test_cal.cpu().detach().numpy())
        visualize_result(y_test.numpy(), y_test_cal.cpu().detach().numpy(), test, f'{target_gas} Test DANN')
        visualize_result_trend(y_test.numpy(), y_test_cal.cpu().detach().numpy(), test, f'{target_gas} Test DANN')