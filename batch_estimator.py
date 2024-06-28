""" import torch
import torch.nn as nn
import torch.nn.functional as F

class Estimator(nn.Module):
    def __init__(self, classifier_state_length, is_target_dqn, bias_average):
        super(Estimator, self).__init__()

        # Define the layers.
        self.fc1 = nn.Linear(classifier_state_length, 10)
        self.fc2 = nn.Linear(10 + 1, 5)  # +1 because we'll concatenate action_placeholder.
        self.fc3 = nn.Linear(5, 1)

        # Initialize bias for the final layer
        nn.init.constant_(self.fc3.bias, bias_average)

        # If it's a target DQN, we freeze the layers' parameters to prevent training.
        if is_target_dqn:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, classifier_input, action_input):

        x = torch.sigmoid(self.fc1(classifier_input))
        x = torch.cat((x, action_input), dim=1)  # Concatenate along the feature dimension.
        x = torch.sigmoid(self.fc2(x))
        predictions = self.fc3(x)
        return predictions """

import torch
import torch.nn as nn
import torch.nn.functional as F

class Estimator(nn.Module):
    def __init__(self, classifier_state_length, is_target_dqn, bias_average):
        super(Estimator, self).__init__()

        # Define the layers.
        self.fc1 = nn.Linear(classifier_state_length, 10)
        self.fc2 = nn.Linear(10 + 1, 5)  # +1 because we'll concatenate action_input.
        self.fc3 = nn.Linear(5, 1)

        # Initialize bias for the final layer
        nn.init.constant_(self.fc3.bias, bias_average)

        # If it's a target DQN, we freeze the layers' parameters to prevent training.
        if is_target_dqn:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, classifier_input, action_input):
        # Ensure action_input is of the correct shape (batch_size, 1)
        if action_input.dim() == 1:
            action_input = action_input.unsqueeze(1)

        x = torch.sigmoid(self.fc1(classifier_input))
        
        # Debugging print to check shapes before concatenation
        print(f"x shape: {x.shape}, action_input shape: {action_input.shape}")
        
        # Concatenate along the feature dimension.
        x = torch.cat((x, action_input), dim=1)
        
        # Debugging print after concatenation
        print(f"x shape after concatenation: {x.shape}")
        
        x = torch.sigmoid(self.fc2(x))
        predictions = self.fc3(x)
        return predictions
