import torch
import torch.nn as nn
import torch.nn.functional as F
import math 
from src.utils import get_device
import torchvision.models as models

class MaskedLinear(nn.Module):
    def __init__(self, in_features, out_features, bias=True, mask_enabled=True, freeze_weights=False, signs_enabled=True):
        super(MaskedLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.mask_enabled = mask_enabled
        self.signs_enabled = signs_enabled
        self.freeze_weights = freeze_weights
        print(f'Created masked with: signs {self.signs_enabled}, mask {self.mask_enabled}, weights {self.freeze_weights}')
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))

        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)

        if freeze_weights:
            self.weight.requires_grad = False
            self.bias.requires_grad = False


        self.mask_param = nn.Parameter(torch.Tensor(out_features, in_features))
        self.signs_mask_param = nn.Parameter(torch.Tensor(out_features, in_features))
        if mask_enabled == False:
            self.mask_param.requires_grad = False
        else:
            self.mask_param.requires_grad = True

        if signs_enabled == False:
            self.signs_mask_param.requires_grad = False
        else:
            self.signs_mask_param.requires_grad = True
        self.init_parameters()

    def init_parameters(self):
        nn.init.kaiming_normal_(self.weight,a= 0)
        #nn.init.uniform_(self.weight, a=-0.01, b=0.01)
        nn.init.uniform_(self.mask_param, a=0.3, b=0.3)  # Initialize mask_param, starts with all connections
        nn.init.uniform_(self.signs_mask_param, a=0.2, b=0.2)  # Initialize mask_param
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, input):
        if self.mask_enabled:
            # Constrain mask values between 0 and 1 using sigmoid
            mask_thresholded = BinaryMaskFunction.apply(self.mask_param)
            # Apply mask to weights
            masked_weight = self.weight * mask_thresholded
        else:
            masked_weight = self.weight

        if self.signs_enabled:
            # Constrain mask values between -1 and 1 using sigmoid
            mask_signs = SignMaskFunction.apply(self.signs_mask_param)
            masked_weight = masked_weight * mask_signs
        else:
            masked_weight = masked_weight


        return F.linear(input, masked_weight, self.bias)
class MaskedConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding =0, stride = 1, bias=True, mask_enabled=True, freeze_weights=False, signs_enabled=True):
        super(MaskedConv2d, self).__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.padding = padding
        self.stride = stride

        self.mask_enabled = mask_enabled
        self.signs_enabled = signs_enabled
        self.freeze_weights = freeze_weights
        print(f'Created masked with: signs {self.signs_enabled}, mask {self.mask_enabled}, weights {self.freeze_weights}')
        
        self.weight =  nn.Parameter(torch.Tensor(out_channels, in_channels, kernel_size, kernel_size)) 
        self.mask_param = nn.Parameter(torch.Tensor(out_channels, in_channels, kernel_size, kernel_size)) 
        self.signs_mask_param = nn.Parameter(torch.Tensor(out_channels, in_channels, kernel_size, kernel_size)) 

        if bias:
            self.bias = nn.Parameter(torch.randn(out_channels))
        else:
            self.register_parameter('bias', None)
        if freeze_weights:
            self.weight.requires_grad = False
            self.bias.requires_grad = False

        if mask_enabled == False:
            
            self.mask_param.requires_grad = False
        else:
            self.mask_param.requires_grad = True
        if signs_enabled == False:
            self.signs_mask_param.requires_grad = False
        else:
            self.signs_mask_param.requires_grad = True
    
        self.init_parameters()
        
    def init_parameters(self):
        nn.init.kaiming_normal_(self.weight, mode='fan_out', nonlinearity='relu')
        #nn.init.kaiming_normal_(self.weight, a = 0)
        nn.init.uniform_(self.mask_param, a = 0.2, b = 0.2)
        nn.init.uniform_(self.signs_mask_param, a=0.2, b=0.2)
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)
    
    def forward(self, input):
        if self.mask_enabled:
            mask_thresholded = BinaryMaskFunction.apply(self.mask_param)
            masked_weight = self.weight * mask_thresholded
        else:
            masked_weight = self.weight
            
        if self.signs_enabled:
            mask_signs = SignMaskFunction.apply(self.signs_mask_param)
            masked_weight = masked_weight * mask_signs    
        else:
            masked_weight = masked_weight
        return F.conv2d(input, masked_weight, self.bias, self.stride, self.padding)

class BasicBlock(nn.Module):

    def __init__(self, in_channels, out_channels, stride=1, downsample=None, mask_enabled=True, freeze_weights=False, signs_enabled=True):
        super(BasicBlock, self).__init__()
        self.mask_enabled = mask_enabled
        self.signs_enabled = signs_enabled
        self.freeze_weights = freeze_weights
        self.conv1 = MaskedConv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False,mask_enabled=self.mask_enabled, freeze_weights=self.freeze_weights, signs_enabled=self.signs_enabled)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = MaskedConv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False, mask_enabled=self.mask_enabled, freeze_weights=self.freeze_weights, signs_enabled=self.signs_enabled)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample  


    def get_masked_percentage_tensor(self) -> torch.Tensor:
        total = 0
        masked = torch.tensor(0, device=get_device(), dtype=torch.float)
        for layer in [self.conv1, self.conv2]:
            total += layer.weight.numel()
            mask = torch.sigmoid(layer.mask_param)
            masked += mask.sum()
        if self.downsample is not None:
            for sublayer in self.downsample:
                if hasattr(sublayer, 'mask_param'):  
                    total += sublayer.weight.numel()
                    mask = torch.sigmoid(sublayer.mask_param)
                    masked += mask.sum()
        return masked, total

    def get_true_masked_percentage_tensor(self) -> torch.Tensor:
        total = 0
        masked = torch.tensor(0, device=get_device(), dtype=torch.float)

        for layer in [self.conv1, self.conv2]:
            total += layer.weight.numel()
            mask = torch.sigmoid(layer.mask_param)
            mask_thresholded = (mask >= 0.5).float()
            masked += mask_thresholded.sum()

        if self.downsample is not None:
            for sublayer in self.downsample:
                if hasattr(sublayer, 'mask_param'):  
                    total += sublayer.weight.numel()
                    mask = torch.sigmoid(sublayer.mask_param)
                    mask_thresholded = (mask >= 0.5).float()
                    masked += mask_thresholded.sum()

        return masked, total

    def forward(self, x):
        identity = x 

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity  
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=1000, mask_enabled=True, freeze_weights=False, signs_enabled=False):
        super(ResNet, self).__init__()
        self.mask_enabled = mask_enabled
        self.signs_enabled = signs_enabled
        self.freeze_weights = freeze_weights
        self.in_channels = 64
        

        self.conv1 = MaskedConv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False, mask_enabled=self.mask_enabled, freeze_weights=self.freeze_weights, signs_enabled=self.signs_enabled)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # Residual block layers
        self.layer1 = self._make_layer(block, 64, layers[0], mask_enabled = self.mask_enabled, freeze_weights = self.freeze_weights, signs_enabled = self.signs_enabled)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, mask_enabled = self.mask_enabled, freeze_weights = self.freeze_weights, signs_enabled = self.signs_enabled)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, mask_enabled = self.mask_enabled, freeze_weights = self.freeze_weights, signs_enabled = self.signs_enabled)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, mask_enabled = self.mask_enabled, freeze_weights = self.freeze_weights, signs_enabled = self.signs_enabled)

      
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = MaskedLinear(512 * block.expansion, num_classes, mask_enabled=self.mask_enabled, freeze_weights=self.freeze_weights, signs_enabled=self.signs_enabled)
        #self.load_pretrained_weights()
    def _make_layer(self, block, out_channels, blocks, stride=1, mask_enabled=True, freeze_weights=False, signs_enabled=True):
        downsample = None
        if stride != 1 or self.in_channels != out_channels * block.expansion:
            downsample = nn.Sequential(
                MaskedConv2d(self.in_channels, out_channels * block.expansion, kernel_size=1, stride=stride, bias=False, mask_enabled=self.mask_enabled, freeze_weights=self.freeze_weights, signs_enabled=self.signs_enabled),
                nn.BatchNorm2d(out_channels * block.expansion),
            )

        layers = []
        layers.append(block(self.in_channels, out_channels, stride, downsample, mask_enabled = self.mask_enabled, signs_enabled = self.signs_enabled, freeze_weights = self.freeze_weights))
        self.in_channels = out_channels * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.in_channels, out_channels, mask_enabled = self.mask_enabled, signs_enabled = self.signs_enabled, freeze_weights = self.freeze_weights))

        return nn.Sequential(*layers)

    def get_masked_percentage_tensor(self) -> torch.Tensor:
        total = 0
        masked = torch.tensor(0, device=get_device(), dtype=torch.float)

        for layer in [self.fc, self.conv1]:
            total += layer.weight.numel()
            mask = torch.sigmoid(layer.mask_param)
            masked += mask.sum()

        for layer in [self.layer1, self.layer2, self.layer3, self.layer4]:
            for block in layer:
                if hasattr(block, 'mask_param'):
                    aux_masked, aux_total = block.get_masked_percentage_tensor()
                    masked += aux_masked
                    total += aux_total
            
        return masked / total

    def get_true_masked_percentage_tensor(self) -> torch.Tensor:
        total = 0
        masked = torch.tensor(0, device=get_device(), dtype=torch.float)
        for layer in [self.fc, self.conv1]:
            total += layer.weight.numel()
            mask = torch.sigmoid(layer.mask_param)
            mask_thresholded = (mask >= 0.5).float()
            masked += mask_thresholded.sum()

        for layer in [self.layer1, self.layer2, self.layer3, self.layer4]:
            for block in layer:
                if hasattr(block, 'mask_param'):
                    aux_masked, aux_total = block.get_true_masked_percentage_tensor()
                    masked += aux_masked
                    total += aux_total

        return masked / total
    def load_pretrained_weights(self, weight_path=r"C:\Users\Statia 1\Desktop\AlexoaieAntonio\XAI_paper\nn_weights\resnet18-f37072fd.pth"):
        # Load weights from the specified .pth file
        pretrained_state = torch.load(weight_path)  # Load the state dictionary from file

        # Get the current state dictionary of our model
        own_state = self.state_dict()

        for name, param in pretrained_state.items():
            if name not in own_state:
                print(f"Parameter '{name}' does not match any layer in the model's own_state.")
                continue  # Ignore parameters that don’t match our model structure

            if isinstance(param, nn.Parameter):
                param = param.data  # Convert parameters to tensors
            
            # Check for size compatibility
            if own_state[name].size() != param.size():
                print(f"Skipping '{name}' due to size mismatch: expected {own_state[name].size()}, got {param.size()}.")
                continue

            # Only load weights for the main layers (ignore mask/sign layers)
            if 'mask_param' not in name and 'signs_mask_param' not in name:
                own_state[name].copy_(param)

        print(f"Loaded pretrained weights from {weight_path}.")
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

