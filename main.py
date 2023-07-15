import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from datetime import datetime, timedelta
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.linear_model import LinearRegression
import torch
import torch.nn as nn
from utils import *
from model import *
import argparse
import glob
import os
import datetime


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--train", help="start training", action="store_true")
    parser.add_argument("--test", help="start testing", action="store_true")
    parser.add_argument("--store_data", help="store the data to folder", action="store_true")
    parser.add_argument("--epochs", help="number of epochs", type=int, default=80)
    parser.add_argument("--batch_size", help="batch size", type=int, default=32)
    parser.add_argument("--win_len", help="window length", type=int, default=24)
    parser.add_argument("--kernel_width", help="kernel width", type=int, default=4)
    parser.add_argument("--hidden_size", help="hidden size", type=int, default=32)
    parser.add_argument("--lr", help="learning rate", type=float, default=1e-4)
    parser.add_argument("--single_sensor_1", help="use single sensor (SGX)", action="store_true")
    parser.add_argument("--single_sensor_2", help="use single sensor (SPEC)", action="store_true")
    parser.add_argument("--target_gas", help="gas for training", type=str, default='O3')
    args = parser.parse_args()
    
    # set the hyperparameters
    num_epochs = args.epochs
    batch_size = args.batch_size
    win_len = args.win_len
    kernel_width = args.kernel_width
    hidden_size = args.hidden_size
    lr = args.lr
    target_gas = args.target_gas

    # set random seed for reproducibility
    setSeed(0)

    # store the data to folder
    if args.store_data == True:
        storeData()

    # read the data
    spec_data, sgx_data, ref_data = readData()
    data = pd.concat([ref_data, sgx_data, spec_data], axis = 1)
    data = data.reindex(data.index, fill_value=np.nan)

    if args.single_sensor_1:
        columns = ['REF-AMB_TEMP', 'REF-RH', f'SGX-{target_gas}', f'REF-{target_gas}']
    elif args.single_sensor_2:
        columns = ['REF-AMB_TEMP', 'REF-RH', f'SPEC-{target_gas}', f'REF-{target_gas}']
    else:
        columns = ['SGX-NO2', 'SGX-CO', f'SPEC-{target_gas}', f'REF-{target_gas}']

    data.dropna(subset=columns, inplace=True)
    data = data.abs()

    # Data Division
    dates = data.index
    print('The total valid samples:', len(dates))

    # Keep the dates ealier than 2023-05-31 23:00:00
    dates = dates[dates <= pd.to_datetime('2023-05-31 23:00:00')]

    split_date_1 = pd.to_datetime('2023-03-31 23:00:00')
    split_date_2 = pd.to_datetime('2023-04-30 23:00:00')

    train = dates[dates <= split_date_1]
    val = dates[(dates > split_date_1) & (dates <= split_date_2)]
    test = dates[dates > split_date_2]
    print('Train size: {:d}, Validation size: {:d}, Test size: {:d}'.format(len(train), len(val), len(test)))

    # prepare data
    train_data = data.loc[train, columns]
    val_data = data.loc[val, columns]
    test_data = data.loc[test, columns]

    # Scaler
    scaler = MinMaxScaler()
    scaler.fit(train_data)
    train_data = np.transpose(scaler.transform(train_data))
    val_data = np.transpose(scaler.transform(val_data))
    test_data = np.transpose(scaler.transform(test_data))
    # print(train_data.shape, val_data.shape, test_data.shape)

    # create sequences
    X_train, y_train, train = create_sequences(train_data, win_len, train, use_consecutive = False)
    X_val, y_val, val = create_sequences(val_data, win_len, val, use_consecutive = False)
    X_test, y_test, test = create_sequences(test_data, win_len, test, use_consecutive = False)
    # print(X_train.shape, y_train.shape, X_val.shape, y_val.shape, X_test.shape, y_test.shape)
    #  checking device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using {device} device")

    # model initialization
    input_features = X_train.shape[2]
    # model = Simple_CNN_GRU(kernel_width=kernel_width, input_features=input_features, hidden_size=hidden_size).to(device)
    model = CNN_GRU(kernel_width=kernel_width, input_features=input_features, hidden_size=hidden_size).to(device)

    # loss & optimizer
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    #  training
    if args.train == True:
        
        train_log = []
        val_log = []

        # Generate a unique name for the model based on the current timestamp
        timestamp = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

        for epoch in range(num_epochs):
            train_loss = 0.0

            # training data indices
            indices = torch.tensor(list(range(len(X_train))))

            for i in range(0, X_train.shape[0], batch_size):
                optimizer.zero_grad()

                # get batch of data
                indices_batch = indices[i:i+batch_size]
                batch_X, batch_y = X_train[indices_batch].to(device), y_train[indices_batch].to(device)

                # forward pass
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)

                # backward pass and optimization
                loss.backward()
                optimizer.step()

                # loss calculation
                train_loss += loss.item()

            # print average loss for the epoch
            y_val_cal = model(X_val.to(device))
            avg_val_loss = criterion(y_val_cal, y_val.to(device)).item()
            avg_train_loss = train_loss / (X_train.shape[0] / batch_size)
            print('Epoch [{}/{}], Train Loss: {:.4f}, Validation Loss: {:.4f}'.format(epoch+1, num_epochs, avg_train_loss, avg_val_loss))
            train_log.append(avg_train_loss)
            val_log.append(avg_val_loss)

        model_name = f'cnn_gru_{timestamp}'
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

        # Calculate the loss for the train set
        y_train_cal = model(X_train.to(device))
        print('Train')
        score(y_train, y_train_cal.cpu().detach().numpy())
        visualize_result(y_train.numpy(), y_train_cal.cpu().detach().numpy(), train, f'{target_gas} Train CNN_GRU')
        visualize_result_trend(y_train.numpy(), y_train_cal.cpu().detach().numpy(), train, f'{target_gas} Train CNN_GRU')

        # Calculate the loss for the val set
        y_val_cal = model(X_val.to(device))
        print('Val')
        score(y_val, y_val_cal.cpu().detach().numpy())
        visualize_result(y_val.numpy(), y_val_cal.cpu().detach().numpy(), val, f'{target_gas} Val CNN_GRU')
        visualize_result_trend(y_val.numpy(), y_val_cal.cpu().detach().numpy(), val, f'{target_gas} Val CNN_GRU')

        # Calculate the loss for the test set
        y_test_cal = model(X_test.to(device))
        print('Test')
        score(y_test, y_test_cal.cpu().detach().numpy())
        visualize_result(y_test.numpy(), y_test_cal.cpu().detach().numpy(), test, f'{target_gas} Test CNN_GRU')
        visualize_result_trend(y_test.numpy(), y_test_cal.cpu().detach().numpy(), test, f'{target_gas} Test CNN_GRU')





        








