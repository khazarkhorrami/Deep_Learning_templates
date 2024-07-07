#!/usr/bin/env python
# -*- coding: utf-8 -*-

from copy import deepcopy

import numpy as np
from torch import Tensor, rand, cuda, no_grad
from torch.nn import Module, Linear, Dropout, ReLU, MSELoss
from torch.optim import Adam

__docformat__ = 'reStructuredText'
__all__ = ['PyTorchExample']


class PyTorchExample(Module):

    def __init__(self):
       
        super().__init__()

        self.layer_1 = Linear(in_features=layer_1_input_dim,
                              out_features=layer_1_output_dim)
        self.layer_2 = Linear(in_features=layer_2_input_dim,
                              out_features=layer_1_output_dim)
        self.layer_3 = Linear(in_features=layer_1_output_dim,
                              out_features=layer_3_output_dim)

        self.dropout = Dropout(dropout)

        self.non_linearity_1 = ReLU()
        self.non_linearity_2 = ReLU()

    def forward(self,
                x_1: Tensor,
                x_2: Tensor) \
            -> Tensor:
        """Forward pass of the DNN.

        :param x_1: First input to the DNN.
        :type x_1: torch.Tensor
        :param x_2: Second input to the DNN.
        :type x_2: torch.Tensor
        :return: Output of the DNN.
        :rtype: torch.Tensor
        """

        # Use first input as input to layer 1
        h_1 = self.non_linearity_1(self.layer_1(x_1))

        # Use second input as input to layer 2
        h_2 = self.non_linearity_2(self.layer_2(x_2))

        # Apply dropout to both h1 and h2
        h_1 = self.dropout(h_1)
        h_2 = self.dropout(h_2)

        # Combine h1 and h2 and use them as an input ot layer 3
        y_hat = self.layer_3(h_1 + h_2)

        # Return the output
        return y_hat


def main():
    # Check if CUDA is available, else use CPU
    device = 'cuda' if cuda.is_available() else 'cpu'
    print(f'Process on {device}', end='\n\n')

    # Define hyper-parameters to be used.
    epochs = 300
    x_1_dim = 10
    x_2_dim = 5

    layer_1_output_dim = 3
    y_dim = 1

    dropout = 0.2
    nb_examples_training = 60
    nb_examples_validation = 20
    nb_examples_testing = 20
    batch_size = 2

    # Instantiate our DNN
    example_dnn = PyTorchExample(
        layer_1_input_dim=x_1_dim,
        layer_1_output_dim=layer_1_output_dim,
        layer_2_input_dim=x_2_dim,
        layer_3_output_dim=y_dim,
        dropout=dropout)

    # Pass DNN to the available device.
    # example_dnn = example_dnn.to(device)

    # Give the parameters of our DNN to an optimizer.
    optimizer = Adam(params=example_dnn.parameters(), lr=1e-3)

    # Instantiate our loss function as a class.
    loss_function = MSELoss()

    # Create our training dataset.
    x_1_training = rand(nb_examples_training, x_1_dim)
    x_2_training = rand(nb_examples_training, x_2_dim)
    y_training = rand(nb_examples_training, y_dim)

    # Create our validation dataset.
    x_1_validation = rand(nb_examples_validation, x_1_dim)
    x_2_validation = rand(nb_examples_validation, x_2_dim)
    y_validation = rand(nb_examples_validation, y_dim)

    # Create our testing dataset.
    x_1_testing = rand(nb_examples_testing, x_1_dim)
    x_2_testing = rand(nb_examples_testing, x_2_dim)
    y_testing = rand(nb_examples_testing, y_dim)

    # Variables for the early stopping
    lowest_validation_loss = 1e10
    best_validation_epoch = 0
    patience = 30
    patience_counter = 0

    best_model = None

    # Start training.
    for epoch in range(epochs):

        # Lists to hold the corresponding losses of each epoch.
        epoch_loss_training = []
        epoch_loss_validation = []

        # Indicate that we are in training mode, so (e.g.) dropout
        # will function
        example_dnn.train()

        # For each batch of our dataset.
        for i in range(0, nb_examples_training, batch_size):
            # Zero the gradient of the optimizer.
            optimizer.zero_grad()

            # Get the batches.
            x_1_input = x_1_training[i:i+batch_size, :]
            x_2_input = x_2_training[i:i+batch_size, :]
            y_output = y_training[i:i+batch_size, :]

            # Give them to the appropriate device.
            # x_1_input = x_1_input.to(device)
            # x_2_input = x_2_input.to(device)
            # y_output = y_output.to(device)

            # Get the predictions of our model.
            y_hat = example_dnn(x_1_input, x_2_input)

            # Calculate the loss of our model.
            loss = loss_function(input=y_hat, target=y_output)

            # Do the backward pass
            loss.backward()

            # Do an update of the weights (i.e. a step of the optimizer)
            optimizer.step()

            # Loss the loss of the batch
            epoch_loss_training.append(loss.item())

        # Indicate that we are in training mode, so (e.g.) dropout
        # will **not** function
        example_dnn.eval()

        # Say to PyTorch not to calculate gradients, so everything will
        # be faster.
        with no_grad():

            # For every batch of our validation data.
            for i in range(0, nb_examples_validation, batch_size):

                # Get the batch
                x_1_input = x_1_validation[i:i+batch_size, :]
                x_2_input = x_2_validation[i:i+batch_size, :]
                y_output = y_validation[i:i+batch_size, :]

                # Pass the data to the appropriate device.
                # x_1_input = x_1_input.to(device)
                # x_2_input = x_2_input.to(device)
                # y_output = y_output.to(device)

                # Get the predictions of the model.
                y_hat = example_dnn(x_1_input, x_2_input)

                # Calculate the loss.
                loss = loss_function(input=y_hat, target=y_output)

                # Log the validation loss.
                epoch_loss_validation.append(loss.item())

        # Calculate mean losses.
        epoch_loss_validation = np.array(epoch_loss_validation).mean()
        epoch_loss_training = np.array(epoch_loss_training).mean()

        # Check early stopping conditions.
        if epoch_loss_validation < lowest_validation_loss:
            lowest_validation_loss = epoch_loss_validation
            patience_counter = 0
            best_model = deepcopy(example_dnn.state_dict())
            best_validation_epoch = epoch
        else:
            patience_counter += 1

        # If we have to stop, do the testing.
        if patience_counter >= patience:
            print('\nExiting due to early stopping', end='\n\n')
            print(f'Best epoch {best_validation_epoch} with loss {lowest_validation_loss}', end='\n\n')
            if best_model is None:
                print('No best model. ')
            else:
                # Process similar to validation.
                print('Starting testing', end=' | ')
                testing_loss = []
                example_dnn.eval()
                with no_grad():
                    for i in range(0, nb_examples_testing, batch_size):
                        x_1_input = x_1_testing[i:i + batch_size, :]
                        x_2_input = x_2_testing[i:i + batch_size, :]
                        y_output = y_testing[i:i + batch_size, :]

                        # x_1_input = x_1_input.to(device)
                        # x_2_input = x_2_input.to(device)
                        # y_output = y_output.to(device)

                        y_hat = example_dnn(x_1_input, x_2_input)

                        loss = loss_function(input=y_hat, target=y_output)

                        testing_loss.append(loss.item())

                testing_loss = np.array(testing_loss).mean()
                print(f'Testing loss: {testing_loss:7.4f}')
                break
        print(f'Epoch: {epoch:03d} | '
              f'Mean training loss: {epoch_loss_training:7.4f} | '
              f'Mean validation loss {epoch_loss_validation:7.4f}')


if __name__ == '__main__':
    main()

# EOF
