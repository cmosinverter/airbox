import numpy as np
import torch
from datetime import datetime

def training(model, X_train, y_train, X_val, y_val, num_epochs, batch_size, optimizer, criterion, device, patience=8):

    train_log = []
    val_log = []

    # Generate a unique name for the model based on the current timestamp
    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

    # best_val_loss = np.inf
    # epochs_without_improvement = 0

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

        # Check if validation loss improved; if not, increment the counter
        # if avg_val_loss < best_val_loss:
        #     best_val_loss = avg_val_loss
        #     epochs_without_improvement = 0
        # else:
        #     epochs_without_improvement += 1

        # Check if we need to stop training based on patience
        # if epochs_without_improvement >= patience:
        #     print(f'Early stopping at epoch {epoch+1} as validation loss did not improve for {patience} consecutive epochs.')
            # break
    model_name = f'cnn_gru_{timestamp}'
    # Save the model with the new name
    torch.save(model.state_dict(), f'model/{model_name}.pth')


