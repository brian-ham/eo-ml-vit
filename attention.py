
from transformers import ViTForImageClassification, ViTConfig
import torch
from PIL import Image
import torchvision.transforms as transforms

import matplotlib.pyplot as plt
import numpy as np

from skimage.transform import resize

# Load the model
config = ViTConfig.from_pretrained('google/vit-base-patch16-224')
model = ViTForImageClassification(config)

# Replace the classifier with one that matches the saved weights
model.classifier = torch.nn.Linear(config.hidden_size, 1)
model.load_state_dict(torch.load('vit_model.pth', map_location=torch.device('cpu')))
model.config.output_attentions = True

print("Imported model")

# Load image
image = Image.open('earth_models/images-2014/31.265259000000004-32.258195.png').convert('RGB')

# Transform
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
input_tensor = transform(image).unsqueeze(0)  # Create a mini-batch as expected by the model

print("Ppened image")

# Foward Pass
outputs = model(input_tensor)
attention_weights = outputs.attentions 
print("Completed forward pass")

# Visualize attentino map
attention_map = attention_weights[8][0, 5].detach().numpy()
print("Plotting")

# Plot with colormap
plt.matshow(attention_map, cmap='hot')
plt.title("Normalized Attention Map - Layer 8, Head 5")
plt.savefig("attention_map.png")

# Attention Overlay
attention_matrix = attention_map[1:, 1:]  # Excluding class token
average_attention = np.mean(attention_matrix, axis=0)  # Average across rows to get a general attention vector
attention_grid = average_attention.reshape((14, 14))
resized_attention_grid = resize(attention_grid, (224, 224), order=3, mode='edge', anti_aliasing=True)

plt.figure(figsize=(10, 10))
plt.imshow(image)
plt.imshow(resized_attention_grid, cmap='hot', alpha=0.6)
plt.colorbar()
plt.axis('off')
plt.title('Patch-based Attention Overlay')
plt.savefig("attention_overlay.png")
plt.close()