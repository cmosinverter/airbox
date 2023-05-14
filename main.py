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
import glob
import os
import datetime


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--train", help="start training", action="store_true")
    parser.add_argument("--test", help="start testing", action="store_true")
    parser.add_argument("--store_data", help="store the data to folder", action="store_true")
    parser.add_argument("--num_epochs", help="number of epochs", type=int, default=100)
    parser.add_argument("--batch_size", help="batch size", type=int, default=16)
    parser.add_argument("--win_len", help="window length", type=int, default=24)
    parser.add_argument("--kernel_width", help="kernel width", type=int, default=4)
    parser.add_argument("--hidden_size", help="hidden size", type=int, default=64)
    parser.add_argument("--lr", help="learning rate", type=float, default=1e-4)
    args = parser.parse_args()
    
    # set the hyperparameters
    num_epochs = args.num_epochs
    batch_size = args.batch_size
    win_len = args.win_len
    kernel_width = args.kernel_width
    hidden_size = args.hidden_size
    lr = args.lr

    # set random seed for reproducibility
    setSeed(0)

    # store the data to folder
    if args.store_data == True:
        storeData()

    # read the data
    sgx_data, ref_data = readData()

    # target & additional features
    target_gas = 'CO'
    external_gas_feature = []

    all_feature = ['SGX-' + target_gas] + external_gas_feature
    all_feature

    # Data Division
    dates = sgx_data.loc[sgx_data.index.isin(ref_data.index), :].index
    print('The total valid samples:', len(dates))
    # Add time feature
    diff_from_first = ref_data.index - ref_data.index[0]
    diff_in_hours = diff_from_first.total_seconds() / 3600
    ref_data['time'] = diff_in_hours


    
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
    X_train, y_train = create_sequences(train_data, win_len)
    X_val, y_val = create_sequences(val_data, win_len)
    X_test, y_test = create_sequences(test_data, win_len)
    # print(X_train.shape, y_train.shape, X_val.shape, y_val.shape, X_test.shape, y_test.shape)
    
    #  checking device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using {device} device")

    # model initialization
    input_features = X_train.shape[2]
    # model = Simple_CNN_GRU(kernel_width=kernel_width, input_features=input_features, hidden_size=hidden_size).to(device)
    model = CNN_GRU(kernel_width=kernel_width, input_features=input_features, hidden_size=hidden_size).to(device)
    # model = GRU(input_features=input_features, hidden_size=hidden_size).to(device)

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

        plt.figure()
        plt.plot(train_log, label = 'Train Loss')
        plt.plot(val_log, label = 'Val Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.grid()
        plt.legend()
        plt.savefig(f'model/{model_name}.png')
        plt.show()

        # Save the model with the new name
        torch.save(model.state_dict(), f'model/{model_name}.pth')

        # Save all the hyperparameters in a txt file
        with open(f'model/{model_name}.txt', 'w') as f:
            f.write(f'Train loss: {train_log}\n')
            f.write(f'Val loss: {val_log}\n')
            f.write(f'lr: {lr}\n')
            f.write(f'batch_size: {batch_size}\n')
            f.write(f'num_epochs: {num_epochs}\n')
            f.write(f'kernel_width: {kernel_width}\n')
            f.write(f'hidden_size: {hidden_size}\n')
            f.write(f'win_len: {win_len}\n')
            f.write(f'all_feature: {all_feature}\n')
            f.write(f'target_gas: {target_gas}\n')
        
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
        # y_train_cal = model(X_train.to(device))
        # print('Train')
        # score(y_train, y_train_cal.cpu().detach().numpy())
        # visualize_result(y_train.numpy(), y_train_cal.cpu().detach().numpy(), train[win_len-1:], f'{target_gas} Train CNN_GRU')

        # Calculate the loss for the val set
        # y_val_cal = model(X_val.to(device))
        # print('Val')
        # score(y_val, y_val_cal.cpu().detach().numpy())
        # visualize_result(y_val.numpy(), y_val_cal.cpu().detach().numpy(), val[win_len-1:], f'{target_gas} Val CNN_GRU')

        # Calculate the loss for the test set
        y_test_cal = model(X_test.to(device))
        print('Test')
        r2, rmse = score(y_test, y_test_cal.cpu().detach().numpy())
        m, inter = visualize_result(y_test.numpy(), y_test_cal.cpu().detach().numpy(), test[win_len-1:], f'{target_gas} Test CNN_GRU')


        








