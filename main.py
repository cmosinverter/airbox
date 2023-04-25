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
from utils import score, visualize_result, storeData, readData, create_sequences, setSeed
from model import GRU, CNN_GRU, Simple_CNN_GRU
import argparse


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--train", help="start training", action="store_true")
    parser.add_argument("--test", help="start testing", action="store_true")
    parser.add_argument("--store_data", help="store the data to folder", action="store_true")

    args = parser.parse_args()

    # set random seed for reproducibility
    setSeed(0)

    # store the data to folder
    if args.store_data == True:
        storeData()

    # read the data
    sgx_data, ref_data = readData()

    # target & additional features
    target_gas = 'CO'
    external_gas_feature = ['SGX-SO2']

    all_feature = ['SGX-' + target_gas] + external_gas_feature
    all_feature

    # Data Division
    dates = sgx_data.loc[sgx_data.index.isin(ref_data.index), :].index
    train, val, test = np.split(dates, [int(.7*len(dates)), int(.85*len(dates))])
    print('Train size: {:d}, Validation size: {:d}, Test size: {:d}'.format(len(train), len(val), len(test)))

    # prepare data
    train_data = pd.concat([sgx_data.loc[train, all_feature], ref_data.loc[train, ['REF-RH', 'REF-AMB_TEMP']], ref_data.loc[train , ['REF-' + target_gas]]] , axis=1)
    val_data = pd.concat([sgx_data.loc[val, all_feature], ref_data.loc[val, ['REF-RH', 'REF-AMB_TEMP']], ref_data.loc[val , ['REF-' + target_gas]]] , axis=1)
    test_data = pd.concat([sgx_data.loc[test, all_feature], ref_data.loc[test, ['REF-RH', 'REF-AMB_TEMP']], ref_data.loc[test , ['REF-' + target_gas]]] , axis=1)
    
    # Scaler
    scaler = MinMaxScaler()
    scaler.fit(train_data)
    train_data = np.transpose(scaler.transform(train_data))
    val_data = np.transpose(scaler.transform(val_data))
    test_data = np.transpose(scaler.transform(test_data))
    # print(train_data.shape, val_data.shape, test_data.shape)

    # create sequences
    win_len = 24 # Specify the window length
    X_train, y_train = create_sequences(train_data, win_len)
    X_val, y_val = create_sequences(val_data, win_len)
    X_test, y_test = create_sequences(test_data, win_len)
    # print(X_train.shape, y_train.shape, X_val.shape, y_val.shape, X_test.shape, y_test.shape)
    
    #  checking device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using {device} device")

    # model initialization
    kernel_width = 4
    input_features = X_train.shape[2]
    hidden_size = 64
    # model = Simple_CNN_GRU(kernel_width=kernel_width, input_features=input_features, hidden_size=hidden_size).to(device)
    model = CNN_GRU(kernel_width=kernel_width, input_features=input_features, hidden_size=hidden_size).to(device)
    # model = GRU(input_features=input_features, hidden_size=hidden_size).to(device)
    # loss & optimizer
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)


    #  training
    if args.train == True:
        num_epochs = 100
        batch_size = 16
        train_log = []
        val_log = []
        
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


        plt.figure()
        plt.plot(train_log, label = 'Train Loss')
        plt.plot(val_log, label = 'Val Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.grid()
        plt.legend()
        plt.show()
        torch.save(model.state_dict(), 'model/cnn_gru.pth')


    # Testing
    if args.test == True:

        model.load_state_dict(torch.load('model/cnn_gru.pth'))
        model.eval()
        y_train_cal = model(X_train.to(device))
        print('Train')
        score(y_train, y_train_cal.cpu().detach().numpy())
        visualize_result(y_train.numpy(), y_train_cal.cpu().detach().numpy(), train[win_len-1:], f'{target_gas} Train CNN_GRU')

        y_val_cal = model(X_val.to(device))
        print('Val')
        score(y_val, y_val_cal.cpu().detach().numpy())
        visualize_result(y_val.numpy(), y_val_cal.cpu().detach().numpy(), val[win_len-1:], f'{target_gas} Val CNN_GRU')

        y_test_cal = model(X_test.to(device))
        print('Test')
        score(y_test, y_test_cal.cpu().detach().numpy())
        visualize_result(y_test.numpy(), y_test_cal.cpu().detach().numpy(), test[win_len-1:], f'{target_gas} Test CNN_GRU')









