import numpy as np 
from src.experiment_MNIST_FCN.Lenet300_network_vanilla import NetSimple
from src.utils import get_device
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch
import torch.nn as nn


def count_zero_connections_per_neuron(weights, output_file=r"XAI_paper\output_files\neuron_weights_cnt.txt"):
    with open(output_file, 'w') as f:
        for key, weight in weights.items():
            if 'weight' in key:  # Only consider weights, skip biases
                f.write(f"\nProcessing layer: {key}\n")
                
                # Loop over each neuron (row in the weight matrix)
                for neuron_idx, neuron_weights in enumerate(weight):
                    # Outgoing connections (rows)
                    total_outgoing_connections = neuron_weights.numel()  # Total outgoing connections
                    zero_outgoing_connections = (neuron_weights == 0).sum().item()  # Zero outgoing connections

                    # Incoming connections (columns)
                    total_incoming_connections = weight[:, neuron_idx].numel()  # Total incoming connections
                    zero_incoming_connections = (weight[:, neuron_idx] == 0).sum().item()  # Zero incoming connections

                    # Write the results to the file
                    f.write(f"Neuron {neuron_idx}:\n")
                    f.write(f"  {total_outgoing_connections - zero_outgoing_connections} outgoing connections are not zero\n")
                    f.write(f"  {total_incoming_connections - zero_incoming_connections} incoming connections are not zero\n")
    
    print(f"Results written to {output_file}")

def cnt_zeros(weights):


    zero_count = 0
    total_count = 0

    for key, weight in weights.items():
            if 'weight' in key:  
                total_count += weight.numel()  
                zero_count += (weight == 0).sum().item() 

    print(f"Total weights: {total_count}")
    print(f"Zero weights: {zero_count}")
    print(f"Percentage of weights that are zero: {100 * zero_count / total_count:.2f}%")
    

def test(model, test_loader):
    model.eval()
    criterion = nn.CrossEntropyLoss(reduction='sum')
    test_loss = 0
    correct = 0
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(get_device()), target.to(get_device())
            output = model(data)
            test_loss += criterion(output, target).item()  # Sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)      # Get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()
    test_loss /= len(test_loader.dataset)
    print(f'\nTest set: Average loss: {test_loss:.4f}, '
          f'Accuracy: {correct}/{len(test_loader.dataset)}'
          f' ({100. * correct / len(test_loader.dataset):.0f}%)\n')


def validate():
    weights = torch.load(r"XAI_paper\nn_weights\model_v1_with_mask.pth", weights_only= True)
    model = NetSimple()
    
    
    
    activations = {}
    def get_activation(name):
        def hook(model, input, output):
            if name not in activations:
                activations[name] = []
            activations[name].append(output.detach().numpy())
        return hook
    

    model.fc1.register_forward_hook(get_activation('fc1'))
    model.fc2.register_forward_hook(get_activation('fc2'))
    model.fc3.register_forward_hook(get_activation('fc3'))
    
    def cnt_neuron_firing_rate(activations):
        cnt_per_layer = {}
        for name, list_activations in activations.items():

            for activation in list_activations:
                
                activation_transpose = activation.T
                
                if name not in cnt_per_layer:
                    cnt_per_layer[name] = [0 for _ in activation_transpose]

                #print(activation_transpose.shape)

                for i, neuron_activ in enumerate(activation_transpose):
                    cnt_per_layer[name][i] += np.sum(neuron_activ > 0)
                    
        return cnt_per_layer
    new_state_dict = {}
    for key, value in weights.items():
        new_key = key.replace('_weight', '.weight').replace('_bias', '.bias')
        new_state_dict[new_key] = value
    
    # Load the renamed weights into the model
    model.load_state_dict(new_state_dict)
    # Define transformations for the training and testing data
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    # Download and load the training data
    train_dataset = datasets.MNIST(root='./data', train=True,
                                   download=True, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    
    # Download and load the test data
    test_dataset = datasets.MNIST(root='./data', train=False,
                                  download=True, transform=transform)
    
    print("Test dataset size: ", len(test_dataset), "\n")
    
    test_loader = DataLoader(test_dataset, batch_size=100, shuffle=False)
    test(model, test_loader)
    
    
    
    cnt_zeros(weights)
    count_zero_connections_per_neuron(weights)
    cnt_per_layer = cnt_neuron_firing_rate(activations)
    with open(r"XAI_paper\output_files\fire_rate.txt", 'w') as f:
        for layer_name, neuron_counts in cnt_per_layer.items():
            f.write(f"Layer: {layer_name}\n")
            for neuron_idx, count in enumerate(neuron_counts):
                f.write(f"  Neuron {neuron_idx}: Active {count} times\n")

  
