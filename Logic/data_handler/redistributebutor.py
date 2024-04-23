import os
import shutil
from sklearn.model_selection import train_test_split

# Load the dataset
images = os.listdir('C:/Users/Gebruiker/Downloads/myvenv/data/v3.2/train')
labels = [image.split('.')[0] for image in images]  # assuming labels are the image names without the extension

# Split the dataset
train_images, val_images, train_labels, val_labels = train_test_split(images, labels, test_size=0.2, stratify=labels,
                                                                      random_state=42)

# Create directories
os.makedirs('train/images', exist_ok=True)
os.makedirs('train/labels', exist_ok=True)
os.makedirs('val/images', exist_ok=True)
os.makedirs('val/labels', exist_ok=True)

# Save the datasets
for image, label in zip(train_images, train_labels):
    shutil.copy(f'C:/Users/Gebruiker/Downloads/myvenv/data/v3.2/train/{image}', f'train/images/{image}')
    shutil.copy(f'C:/Users/Gebruiker/Downloads/myvenv/data/v3.2/train/{label}.txt', f'train/labels/{label}.txt')

for image, label in zip(val_images, val_labels):
    shutil.copy(f'C:/Users/Gebruiker/Downloads/myvenv/data/v3.2/train/{image}', f'val/images/{image}')
    shutil.copy(f'C:/Users/Gebruiker/Downloads/myvenv/data/v3.2/train/{label}.txt', f'val/labels/{label}.txt')