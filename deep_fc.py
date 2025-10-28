#     Contains the description of the deep_fc model, a CNN made of 4 convolutional layers and 2 FC layer
#     Highly inspired by https://onlinelibrary.wiley.com/doi/full/10.1002/hbm.23730

#     Copyright (C) 2022, Lucas Gruaz
#
#     This program is free software: you can redistribute it and/or modify
#     it under the terms of the GNU General Public License as published by
#     the Free Software Foundation, either version 3 of the License, or
#     (at your option) any later version.
# 
#     This program is distributed in the hope that it will be useful,
#     but WITHOUT ANY WARRANTY; without even the implied warranty of
#     MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#     GNU General Public License for more details.
# 
#     You should have received a copy of the GNU General Public License
#     along with this program.  If not, see <https://www.gnu.org/licenses/>.


##################### IMPORTS ########################

import torch
from torch import nn


###################### MODEL #########################

class DeepFcCNN(nn.Module):
    def __init__(
                self, 
                n_temp_filters = 25,    # Number of temporal filters
                temp_filter_length = 9, # Length of temporal filters
                n_spat_filters = 25,    # Number of spatial filters
                in_chans = 62,          # Number of input channels of the EEG
                max_pool_length = 3,    # Length of max pooling
                max_pool_stride = 3,    # Stride of max pooling
                in_length = 500,        # Length of input EEG
                n_conv_1 = 50,          # Number of filters for the first regular convolution
                conv_1_length = 12,     # Length of filters
                n_conv_2 = 100,         # Same for second convolution
                conv_2_length = 10,
                result_length = 14,     # Resulting length after all convolutions
                n_fc = 512,             # Number of hidden neurons between the two fc layers
                non_lin = nn.ReLU(),    # Non linearity function between the layers
                batch_norm = True,     # Whether to use batch normalization
                dropout = True,        # Whether to use dropout
                drop_perc = 0.5         # Percentage of dropping out
                ):
        
        super().__init__()
        # Temporal and spatial convolutions
        temp_conv = nn.Conv2d(1,n_temp_filters,(1,temp_filter_length), stride=1)
        spat_conv = nn.Conv2d(n_temp_filters, n_spat_filters, (in_chans, 1))
        
        # Batch normalization (if any)
        if batch_norm:
            norm_1 = nn.BatchNorm2d(n_spat_filters)
            norm_2 = nn.BatchNorm2d(n_conv_1)
            norm_3 = nn.BatchNorm2d(n_conv_2)
        else:
            norm_1 = nn.Identity()
            norm_2 = nn.Identity()
            norm_3 = nn.Identity()
        
        # Max pooling
        self.pool  = nn.MaxPool2d((1,max_pool_length), stride=(1,max_pool_stride))
        
        # Regular convolutions
        conv_1 = nn.Conv2d(n_spat_filters, n_conv_1, (1,conv_1_length), stride=1)
        conv_2 = nn.Conv2d(n_conv_1, n_conv_2, (1,conv_2_length), stride=1)
        
        # Fully connected layers
        fc1 = nn.Linear(result_length * n_conv_2, n_fc)
        fc2 = nn.Linear(n_fc, 1)
        
        # Dropout (if any)
        if dropout:
            dropout = nn.Dropout(p=drop_perc)
        else:
            dropout = nn.Identity()
        
        # Non linearity function
        non_lin = non_lin
        # Output activation function
        activation = nn.Sigmoid()
        
        # All layer until last convolution (separated to use grad-CAM visualization)
        self.until_last_conv = nn.Sequential(
            temp_conv,
            spat_conv,
            non_lin,
            norm_1,
            self.pool,
            conv_1,
            non_lin,
            norm_2,
            self.pool,
            conv_2,
            non_lin,
            norm_3
        )
        
        # Classification layers
        self.classifier = nn.Sequential(
            fc1,
            dropout,
            non_lin,
            fc2,
            activation
        )
            

    def forward(self, x):
        # Apply all layers to the input
        x = self.until_last_conv(x)
        x = self.pool(x)
        x = torch.flatten(x,1)
        x = self.classifier(x)
        return torch.squeeze(x)
        
    
    
    