import torch
import numpy as np
import matplotlib.pyplot as plt




def sparsity_per_layer(state_dict_path):
    stateDict = torch.load(state_dict_path)
    layer_total = []
    layer_remaining = []

    for key in stateDict.keys():
        if "weight" not in key:
            continue  # Skip all layers but weights

        layer_total.append(stateDict[key].numel())  
        layer_remaining.append(int((stateDict[key] != 0).float().sum().item())) 

    return layer_total, layer_remaining

def plot_histogram_sparsity(layer_total, layer_remaining):
    layers = [f"Layer {i+1}" for i in range(len(layer_total))]
    x = np.arange(len(layers))


   # plt.bar(x, layer_total, color='lightblue', label='Total Parameters')
    plt.bar(x, layer_remaining, color='blue', label='Log10(Remaining Parameters)')

    plt.yscale("log")
    plt.xlabel("Layers")
    plt.ylabel("Log10(Number of Parameters)")
    plt.title("Logarithmic Sparsity per Layer")
    plt.xticks(x, layers)
    plt.legend()
    plt.show()

if __name__ == "__main__":
    state_dict_path = "networks_saved/lenet300_mnist_016_95"
    layer_total, layer_remaining = sparsity_per_layer(state_dict_path)
    print(layer_total, layer_remaining)
    for i, el  in enumerate(layer_remaining):
        layer_remaining[i] =  el / layer_total[i] * 100
    print(layer_total, layer_remaining)

    plot_histogram_sparsity(layer_total, layer_remaining)






# import os
# import torch
# import numpy as np
# import matplotlib.pyplot as plt
# from torchvision import datasets, transforms

# def extract_matrices():
#     """
#     Extracts binary masks from the first layer weights of multiple models.
#     Each mask indicates which input pixels are used by at least one neuron.
#     """
#     paths = [
#         "networks_saved/lenet300_mnist_016_95",
#         "networks_saved/lenet300_mnist_022_961",
#         "networks_saved/lenet300_mnist_029_97",
#         "networks_saved/lenet300_mnist_042_98", 
#         "networks_saved\lenet300_mnist_064_98",
#         "networks_saved\lenet300_mnist_092_98"
#     ]
#     matrices = []
#     for path in paths:
#         if not os.path.exists(path):
#             print(f"Weight file {path} does not exist. Skipping.")
#             continue
#         stateDict = torch.load(path, map_location=torch.device('cpu'))
#         if "fc1.weight" not in stateDict:
#             print(f"'fc1.weight' not found in {path}. Skipping.")
#             continue
#         layer = stateDict["fc1.weight"]  # Shape: (300, 784)
#         mask = (layer != 0).float()      # Binary mask: 1 if weight != 0, else 0
#         # Sum across neurons to identify used pixels
#         pixel_mask = torch.sum(mask, dim=0) > 0  # Shape: (784,)
#         matrices.append(pixel_mask.float())
#         used_pixels = pixel_mask.sum().item()
#         total_pixels = pixel_mask.numel()
#         print(f"Processed {os.path.basename(path)}: {used_pixels} pixels used out of {total_pixels}.")
#     return matrices



# def apply_mask(mask, image_tensor):
#     """
#     Applies the binary mask to the image tensor.
#     """
#     # Ensure the image is flattened
#     image_flat = image_tensor.view(-1)  # Shape: (784,)
#     # Apply the mask
#     masked_image = image_flat * mask
#     return masked_image

# def plot_image(original, masked, label):
#     """
#     Plots the original and masked images side by side.
#     """
#     fig, axs = plt.subplots(1, 2, figsize=(8, 4))
    
#     axs[0].imshow(original, cmap='gray')
#     axs[0].set_title(f'Original Image (Label: {label})')
#     axs[0].axis('off')
    
#     axs[1].imshow(masked, cmap='gray')
#     axs[1].set_title('Masked Image (Used Pixels)')
#     axs[1].axis('off')
    
#     plt.show()

# def load_images_per_label():
#     """
#     Loads one image per digit label (0-9) from the MNIST dataset.
#     Returns a dictionary mapping each digit to its corresponding image tensor.
#     """
#     transform = transforms.ToTensor()
#     mnist = datasets.MNIST('data', download=True, transform=transform)
#     images_per_label = {}
#     labels_found = set()
#     list_images = []
#     labels = []
#     for idx in range(len(mnist)):
#         image, label = mnist[idx]
#         if label not in images_per_label:
#             images_per_label[label] = image
#             labels_found.add(label)
#             labels.append(label)
#             list_images.append(image)
#             print(f"Loaded image index {idx} for digit {label}.")
#             if len(labels_found) == 10:
#                 break  # Stop after loading one image per digit
#     return labels, list_images

# def main1():
#     # Step 1: Extract masks from all models
#     masks = extract_matrices()
    
#     if not masks:
#         print("No masks were extracted. Exiting.")
#         return
    
#     # Step 2: Combine all masks using logical OR
#       # Shape: (784,)
    
#     # Step 3: Load an MNIST image
#     _, list_images = load_images_per_label()      # image_tensor shape: (1, 28, 28)

#     for idx, mask in enumerate(masks):
#         mask_2d = mask.view(28, 28).numpy()
#         plt.figure(figsize=(4,4))
#         plt.imshow(mask_2d, cmap='gray')
#         plt.title(f'Mask from Model {idx+1}')
#         plt.axis('off')
#         plt.show()
#     # Step 4: Apply the combined mask to the image
#     for image_tensor in list_images:
#         for mask in masks:
#             masked_image = apply_mask(mask, image_tensor)  # Shape: (784,)
            
#             # Step 5: Reshape images for visualization
#             original_image = image_tensor.view(28, 28).numpy()
#             masked_image_2d = masked_image.view(28, 28).detach().numpy()
#             plot_image(original_image, masked_image_2d,1)



# def main():
#     # Step 1: Extract masks from all models
#     masks = extract_matrices()
#     masks.reverse()
#     sparsities = [99.85,99.80,99.70,99.60,99.35,99.1]
#     sparsities.reverse()
#     if not masks:
#         print("No masks were extracted. Exiting.")
#         return
    
#     # Step 2: Load MNIST images
#     labels, list_images = load_images_per_label()  # labels: list of labels corresponding to images
    
#     if not list_images:
#         print("No images loaded. Exiting.")
#         return
    
#     # Determine the unique labels and organize images by label
#     unique_labels = sorted(list(set(labels)))
#     N = len(unique_labels)  # Number of labels (rows)
#     M = len(masks)          # Number of masks (columns)
    
#     # Organize images by label for easy access
#     # If multiple images per label, you might need to adjust this
#     label_to_image = {label: image for label, image in zip(labels, list_images)}
    
#     # Create a figure with a grid of subplots
#     fig, axes = plt.subplots(N + 1, M + 1, figsize=(3 * (M + 1), 3 * (N + 1)))
    
#     # Adjust spacing
#     plt.subplots_adjust(wspace=0.2, hspace=0.2)
    
#     # Iterate through the grid and populate the subplots
#     for i in range(N + 1):
#         for j in range(M + 1):
#             ax = axes[i, j]
#             ax.axis('off')  # Hide axes
            
#             if i == 0 and j == 0:
#                 # Top-left corner remains empty
#                 continue
#             elif i == 0 and j > 0:
#                 # First row: Display masks
#                 mask = masks[j - 1].view(28, 28).numpy()
#                 ax.imshow(mask, cmap='gray')
#                 ax.set_title(f'{sparsities[j-1]}%', fontsize=20)
#             elif j == 0 and i > 0:
#                 # First column: Display original images
#                 original_image = label_to_image[unique_labels[i - 1]]
#                 original_image_2d = original_image.view(28, 28).detach().numpy()
#                 ax.imshow(original_image_2d, cmap='gray')
#                 if i == 1:
#                     ax.set_title('Original Images', fontsize=20)

#             elif i > 0 and j > 0:
#                 # Display masked image
#                 label = unique_labels[i - 1]
#                 image_tensor = label_to_image[label]
                
#                 # Apply the current mask
#                 mask = masks[j - 1]
#                 masked_image = apply_mask(mask, image_tensor)  # Shape: (784,)
                
#                 # Reshape for visualization
#                 masked_image_2d = masked_image.view(28, 28).detach().numpy()
                
#                 ax.imshow(masked_image_2d, cmap='gray')
    
#     plt.savefig('masked_images_mnist.pdf', format='pdf', pad_inches=0.1, bbox_inches='tight')
#     plt.show()
# if __name__ == "__main__":
#     main()




# def main():
#     # Step 1: Extract masks from all models
#     masks = extract_matrices()
#     sparsities = [16,22,29,42,64,92]
#     if not masks:
#         print("No masks were extracted. Exiting.")
#         return
    
#     # Step 2: Combine all masks using logical OR
#       # Shape: (784,)
    
#     # Step 3: Load an MNIST image
#     labels, list_images = load_images_per_label()      # image_tensor shape: (1, 28, 28)

#     for idx, mask in enumerate(masks):
#         mask_2d = mask.view(28, 28).numpy()
#         plt.figure(figsize=(4,4))
#         plt.imshow(mask_2d, cmap='gray')
#         plt.title(f'Mask from Model {idx+1}')
#         plt.axis('off')
#         plt.show()
#         plt.savefig(f'mask_{sparsities[idx]}.pdf', format  = 'pdf')
#     # Step 4: Apply the combined mask to the image
#     for image_tensor in list_images:
#         original_image = image_tensor.view(28, 28).numpy()
#         # save the original image
#         plt.figure(figsize=(4,4))
#         plt.imshow(original_image, cmap='gray')
#         plt.savefig(f'original_image.pdf', format  = 'pdf')

#     for image_tensor in list_images:
#         for i, mask in enumerate(masks):
#             masked_image = apply_mask(mask, image_tensor)  # Shape: (784,)
            
#             # Step 5: Reshape images for visualization
#             original_image = image_tensor.view(28, 28).numpy()
#             masked_image_2d = masked_image.view(28, 28).detach().numpy()
#             plot_image(original_image, masked_image_2d,1)