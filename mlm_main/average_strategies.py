
import math
import torch
import torch.nn as nn

from Multilevel_LM.main_lm.neural_network_construction import FullyConnectedNN
#In this file, we simply average every m nodes, and contruct a new coarser neural network.
def split_tensor_into_blocks(tensor, block_size):
    n = tensor.shape[0]
    size_b = math.ceil(n/block_size)
    blocks = []
    block_average = torch.zeros((size_b,size_b))
    # Loop over the tensor with step size `block_size`
    for i in range(0, n, block_size):
        row_blocks = []
        for j in range(0, n, block_size):
            # Determine the size of the current block
            row_end = min(i + block_size, n)
            col_end = min(j + block_size, n)
            
            # Extract the block
            block = tensor[i:row_end, j:col_end]
            
            row_blocks.append(block)
        
        # Append all blocks of the current row
        blocks.append(row_blocks)
    for k in range(size_b):
        for l in range(size_b):
            block_average[k,l] = (blocks[k][l].sum())/(blocks[k][l].shape[0]*blocks[k][l].shape[1])
            
    
    return block_average

def average_nodes_model(model,m):
    #Define a new model with reduced number of r_nodes_per_layer
    input_dim = model.input_dim
    n_hidden_layers = model.n_hidden_layers
    r_nodes_per_layer = math.ceil(model.r_nodes_per_layer / m)
    output_dim = model.output_dim
    activation_function = model.activation_function
    new_model = FullyConnectedNN(input_dim, n_hidden_layers, r_nodes_per_layer, output_dim,activation_function)
    #print(new_model.hidden_layers[0].weight)
    previous_out_features = model.input_dim
    floor_m = math.floor(model.r_nodes_per_layer / m)
    floormm = floor_m*m
    remain = model.r_nodes_per_layer-floormm
    if math.floor(model.r_nodes_per_layer / m) == math.ceil(model.r_nodes_per_layer / m):
        for i,layer in enumerate(model.hidden_layers):
           
            weights = layer.weight.data.view(-1,m,previous_out_features).mean(dim=1)
            biases = layer.bias.data.view(-1,m).mean(dim=1)
            new_model.hidden_layers[i].weight.data =  weights
            new_model.hidden_layers[i].bias.data = biases
            previous_out_features = new_model.hidden_layers[i].out_features
                
                
    else:
        for i, layer in enumerate(model.hidden_layers):
            for j, new_layer in enumerate(new_model.hidden_layers):
                #if  layer.weight.shape[1] == 1:
                    
            

                    #weights = layer.weight.data[0:floormm].view(-1,m,previous_out_features).mean(dim=1)
                    #biases = layer.bias.data[0:floormm].view(-1,m).mean(dim=1)
                    
                    #remain = model.r_nodes_per_layer-floormm

                    #weights_end = layer.weight.data[floormm:].mean(dim=1)
                    #biases_end = layer.bias.data[floormm:].view(-1,remain).mean(dim=1)
                    #weights = torch.cat((weights,weights_end))
                    #biases = torch.cat((biases,biases_end))
                    #new_model.hidden_layers[i].weight.data =  weights
                    #new_model.hidden_layers[i].bias.data = biases
                    #previous_out_features = new_model.hidden_layers[i].out_features
                #else:
                   
                weights = split_tensor_into_blocks(layer.weight.data,m)
                mask = ~torch.isnan(weights).any(dim=0)

                # Filter out the columns with NaN
                weights = weights[:, mask]
                biases = layer.bias.data[0:floormm].view(-1,m).mean(dim=1)
                biases_end = layer.bias.data[floormm:].view(-1,remain).mean(dim=1)
                biases = torch.cat((biases,biases_end))
                new_model.hidden_layers[i].weight.data =  weights
                new_model.hidden_layers[i].bias.data = biases
                previous_out_features = new_model.hidden_layers[i].out_features
 

    
    return new_model
                
                
                    

                
                
                
    
