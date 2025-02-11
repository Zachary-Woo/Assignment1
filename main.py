import copy
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torchvision import datasets, models
from multiprocessing import freeze_support
import os  # For creating directories
import json  # For saving results
from datetime import datetime  # For unique filenames
import matplotlib.pyplot as plt  # For plotting
import numpy as np
from sklearn.model_selection import train_test_split  # For splitting dataset
import shutil  # For copying files
from collections import Counter  # For counting classes
import cv2  # For visualization
from PIL import Image

# Constants
TRAIN_PHASE = "train"
VAL_PHASE = "val"
TEST_PHASE = "test"
CLASS_NAMES = ['Normal', 'Pneumonia']

# Dataset paths
TRAIN_DIR = "./chest_xray/train"
VAL_DIR = "./chest_xray/val"
TEST_DIR = "./chest_xray/test"

# Training parameters
NUM_CLASSES = 2
BATCH_SIZE = 64
NUM_EPOCHS = 20
LEARNING_RATE = 0.0001
VAL_SPLIT = 0.2

# ImageNet mean and std for normalization/denormalization
# These values are the per-channel means and standard deviations computed on the ImageNet dataset
# They are used to normalize images to the same scale that the pretrained model was trained on
# mean: average pixel values for each channel (R,G,B) across ImageNet
# std: standard deviation of pixel values for each channel (R,G,B) across ImageNet
# Even for non-pretrained models, these values help standardize the input range and improve training
mean = np.array([0.485, 0.456, 0.406])  # RGB means
std = np.array([0.229, 0.224, 0.225])   # RGB standard deviations

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

################################################################################
# Helper Functions
################################################################################
def analyze_dataset(dataset_dir):
    """
    Analyzes the class distribution in a dataset directory
    """
    class_counts = {}
    for class_name in os.listdir(dataset_dir):
        class_path = os.path.join(dataset_dir, class_name)
        if os.path.isdir(class_path):
            class_counts[class_name] = len(os.listdir(class_path))
    return class_counts

def create_validation_split(train_dir, val_dir, val_split=0.2):
    """
    Creates a validation set from training data
    """
    # First, analyze what we have
    print("\nAnalyzing current dataset split...")
    train_counts = analyze_dataset(train_dir)
    val_counts = analyze_dataset(val_dir)
    
    print("Current training set distribution:")
    for class_name, count in train_counts.items():
        print(f"{class_name}: {count} images")
    
    print("\nCurrent validation set distribution:")
    for class_name, count in val_counts.items():
        print(f"{class_name}: {count} images")
    
    # If validation set is too small, create a new one from training data
    if sum(val_counts.values()) < 100:  # Arbitrary threshold
        print("\nValidation set is too small. Creating new split...")
        
        # Create temporary directories for new split
        temp_train_dir = train_dir + "_temp"
        temp_val_dir = val_dir + "_temp"
        os.makedirs(temp_train_dir, exist_ok=True)
        os.makedirs(temp_val_dir, exist_ok=True)
        
        # For each class
        for class_name in train_counts.keys():
            # Create class directories
            os.makedirs(os.path.join(temp_train_dir, class_name), exist_ok=True)
            os.makedirs(os.path.join(temp_val_dir, class_name), exist_ok=True)
            
            # Get list of files
            class_files = os.listdir(os.path.join(train_dir, class_name))
            
            # Split files
            train_files, val_files = train_test_split(
                class_files, 
                test_size=val_split,
                random_state=42  # For reproducibility
            )
            
            # Copy files to new locations
            for file in train_files:
                shutil.copy2(
                    os.path.join(train_dir, class_name, file),
                    os.path.join(temp_train_dir, class_name, file)
                )
            
            for file in val_files:
                shutil.copy2(
                    os.path.join(train_dir, class_name, file),
                    os.path.join(temp_val_dir, class_name, file)
                )
        
        # Replace original directories
        shutil.rmtree(val_dir)
        shutil.move(temp_val_dir, val_dir)
        shutil.rmtree(train_dir)
        shutil.move(temp_train_dir, train_dir)
        
        print("\nNew dataset split created!")
        print("\nNew training set distribution:")
        train_counts = analyze_dataset(train_dir)
        for class_name, count in train_counts.items():
            print(f"{class_name}: {count} images")
        
        print("\nNew validation set distribution:")
        val_counts = analyze_dataset(val_dir)
        for class_name, count in val_counts.items():
            print(f"{class_name}: {count} images")
    
    return train_counts, val_counts

def plot_training_history(history, title, filename):
    """
    Plots training and validation metrics
    """
    plt.figure(figsize=(12, 4))
    
    # Plot loss
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Training Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.title(f'{title} - Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    # Plot accuracy
    plt.subplot(1, 2, 2)
    plt.plot(history['train_acc'], label='Training Accuracy')
    plt.plot(history['val_acc'], label='Validation Accuracy')
    plt.title(f'{title} - Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join('plots', filename))
    plt.close()

def calculate_class_weights(dataset):
    """
    Calculates class weights for imbalanced dataset
    """
    # Count samples per class
    class_counts = Counter([label for _, label in dataset])
    total_samples = sum(class_counts.values())
    
    # Calculate weights
    class_weights = {
        class_idx: total_samples / (len(class_counts) * count)
        for class_idx, count in class_counts.items()
    }
    
    return torch.FloatTensor([class_weights[i] for i in range(len(class_counts))]).to(device)

def train_model(model, criterion, optimizer, dataloaders, num_epochs=10, model_name="model"):
    """
    This function handles the training process.
    I added model saving so we don't have to retrain every time!
    """
    print(f"\nStarting training process...")
    print(f"Training on device: {device}")
    
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    
    history = {
        'train_loss': [],
        'val_loss': [],
        'train_acc': [],
        'val_acc': []
    }
    
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")
        print("-" * 20)

        # Each epoch has a training and validation phase
        for phase in [TRAIN_PHASE, VAL_PHASE]:
            if phase == TRAIN_PHASE:
                model.train()
                print("Training phase:")
            else:
                model.eval()
                print("Validation phase:")

            running_loss = 0.0
            running_corrects = 0
            total_samples = 0
            
            batch_count = 0
            total_batches = len(dataloaders[phase])
            
            for inputs, labels in dataloaders[phase]:
                batch_count += 1
                if batch_count % 10 == 0:
                    print(f"Processing batch {batch_count}/{total_batches}")
                
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == TRAIN_PHASE):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    if phase == TRAIN_PHASE:
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
                total_samples += inputs.size(0)
                
            epoch_loss = running_loss / total_samples
            epoch_acc = running_corrects.double() / total_samples

            print(f"{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")

            if phase == TRAIN_PHASE:
                history['train_loss'].append(epoch_loss)
                history['train_acc'].append(epoch_acc.item())
            else:
                history['val_loss'].append(epoch_loss)
                history['val_acc'].append(epoch_acc.item())
            
            if phase == VAL_PHASE and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
                print(f"New best validation accuracy: {best_acc:.4f}")
                
                checkpoint = {
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'best_acc': best_acc,
                }
                torch.save(checkpoint, f'saved_models/{model_name}_best.pth')
    
    print(f"\nTraining complete!")
    print(f"Best validation accuracy: {best_acc:.4f}")

    model.load_state_dict(best_model_wts)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    with open(f'results/{model_name}_history_{timestamp}.json', 'w') as f:
        json.dump(history, f)
    
    return model, history

def evaluate_model(model, loader, phase=TEST_PHASE):
    """
    Evaluates the model and returns metrics
    """
    model.eval()

    correct = 0
    total = 0
    class_correct = [0, 0]
    class_total = [0, 0]

    with torch.no_grad():
        for inputs, labels in loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            correct += torch.sum(preds == labels).item()
            total += labels.size(0)

            for i in range(len(labels)):
                label = labels[i]
                pred = preds[i]
                if pred == label:
                    class_correct[label] += 1
                class_total[label] += 1

    overall_acc = correct / total
    normal_acc = class_correct[0] / class_total[0] if class_total[0] > 0 else 0
    pneumonia_acc = class_correct[1] / class_total[1] if class_total[1] > 0 else 0

    print(f"\n{phase.capitalize()} Set Results:")
    print(f"Overall accuracy: {overall_acc:.4f}")
    print(f"Normal accuracy: {normal_acc:.4f}")
    print(f"Pneumonia accuracy: {pneumonia_acc:.4f}")
    
    return {
        'overall_acc': overall_acc,
        'normal_acc': normal_acc,
        'pneumonia_acc': pneumonia_acc,
        'class_correct': class_correct,
        'class_total': class_total
    }

def visualize_model_predictions(model, test_loader, device, num_images=5, save_dir='failure_cases'):
    """
    Visualize and save model predictions, focusing on failure cases
    """
    os.makedirs(save_dir, exist_ok=True)
    model.eval()
    
    class_names = ['Normal', 'Pneumonia']
    failure_cases = []
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            
            # Find failure cases
            for j in range(inputs.size()[0]):
                if preds[j] != labels[j]:
                    failure_cases.append({
                        'input': inputs[j],
                        'pred': preds[j].item(),
                        'label': labels[j].item()
                    })
                    if len(failure_cases) >= num_images:
                        break
            if len(failure_cases) >= num_images:
                break
    
    # Plot failure cases
    fig, axes = plt.subplots(1, min(num_images, len(failure_cases)), figsize=(15, 3))
    if len(failure_cases) == 1:
        axes = [axes]
    
    for idx, case in enumerate(failure_cases[:num_images]):
        img = case['input'].cpu().numpy().transpose((1, 2, 0))
        img = std * img + mean  # Denormalize
        img = np.clip(img, 0, 1)
        
        axes[idx].imshow(img)
        axes[idx].set_title(f'Pred: {class_names[case["pred"]]}\nTrue: {class_names[case["label"]]}')
        axes[idx].axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'failure_cases.png'))
    plt.close()

def apply_gradcam(model, input_tensor, target_layer):
    """
    Apply Grad-CAM to visualize model's attention
    """
    # Get the feature maps from the target layer
    feature_maps = None
    gradients = None
    
    def save_fmaps(module, input, output):
        nonlocal feature_maps
        feature_maps = output.detach()
    
    def save_grads(module, grad_in, grad_out):
        nonlocal gradients
        gradients = grad_out[0].detach()
    
    # Register hooks
    handle_fwd = target_layer.register_forward_hook(save_fmaps)
    handle_back = target_layer.register_backward_hook(save_grads)
    
    # Forward pass
    model.eval()
    # Enable gradients for input tensor
    input_tensor.requires_grad = True
    
    # Forward pass
    outputs = model(input_tensor)
    score = outputs[:, outputs.argmax(dim=1)].squeeze()
    
    # Backward pass
    model.zero_grad()
    score.backward()
    
    # Calculate Grad-CAM
    with torch.no_grad():
        if gradients is None or feature_maps is None:
            print("Warning: gradients or feature maps are None")
            handle_fwd.remove()
            handle_back.remove()
            return torch.zeros((input_tensor.size(2), input_tensor.size(3)))
            
        pooled_gradients = torch.mean(gradients, dim=[0, 2, 3])
        for i in range(feature_maps.size()[1]):
            feature_maps[:, i, :, :] *= pooled_gradients[i]
        
        heatmap = torch.mean(feature_maps, dim=1).squeeze()
        heatmap = torch.relu(heatmap)
        heatmap = heatmap / (torch.max(heatmap) + 1e-10)  # Added small epsilon to prevent division by zero
    
    # Clean up
    handle_fwd.remove()
    handle_back.remove()
    
    return heatmap.cpu().numpy()

def visualize_gradcam(model, test_loader, device, num_images=5, save_dir='gradcam'):
    """
    Generate and save Grad-CAM visualizations
    """
    os.makedirs(save_dir, exist_ok=True)
    model.eval()
    
    # Get the last convolutional layer
    target_layer = None
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            target_layer = module
    
    if target_layer is None:
        print("Could not find a convolutional layer in the model")
        return
    
    images_processed = 0
    fig = plt.figure(figsize=(15, 3))
    
    for inputs, labels in test_loader:
        for j in range(inputs.size()[0]):
            if images_processed >= num_images:
                break
                
            input_tensor = inputs[j:j+1].to(device)
            label = labels[j].item()
            
            # Get model prediction
            with torch.no_grad():
                output = model(input_tensor)
                _, pred = torch.max(output, 1)
                pred = pred.item()
            
            # Generate Grad-CAM
            heatmap = apply_gradcam(model, input_tensor, target_layer)
            
            # Plot original image
            ax = fig.add_subplot(1, num_images, images_processed + 1)
            img = inputs[j].cpu().numpy().transpose((1, 2, 0))
            img = std * img + mean  # Denormalize
            img = np.clip(img, 0, 1)
            
            # Overlay heatmap
            heatmap = cv2.resize(heatmap, (224, 224))
            heatmap = np.uint8(255 * heatmap)
            heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
            heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
            overlay = 0.5 * img + 0.3 * heatmap / 255
            
            ax.imshow(overlay)
            ax.set_title(f'Pred: {CLASS_NAMES[pred]}\nTrue: {CLASS_NAMES[label]}')
            ax.axis('off')
            
            images_processed += 1
            
        if images_processed >= num_images:
            break
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'gradcam_visualization.png'))
    plt.close()

def freeze_layers(model, num_layers_to_freeze):
    """
    Freezes the first num_layers_to_freeze layers of the model
    """
    for i, (name, param) in enumerate(model.named_parameters()):
        if i < num_layers_to_freeze:
            param.requires_grad = False

# Create directories for saving models and results
print("Setting up directories for saving results...")
os.makedirs('saved_models', exist_ok=True)
os.makedirs('results', exist_ok=True)
os.makedirs('plots', exist_ok=True)

################################################################################
# Image Transformations
################################################################################
# Before feeding images to our network, we need to preprocess them
# I learned that data augmentation (like flipping and rotating) helps prevent overfitting
# These transformations are applied randomly during training

print("Setting up image transformations...")  # Added to track what's happening

# For training, we want to do data augmentation
train_transforms = transforms.Compose([
    # ResNet expects 224x224 images - had to look this up!
    transforms.Resize((224, 224)),
    
    # Randomly flip some images horizontally
    # This helps our model learn that the same features can appear on either side
    transforms.RandomHorizontalFlip(p=0.5),  # p=0.5 means 50% chance of flipping
    
    # Add some random rotation
    # Not too much though - we don't want to rotate chest X-rays too drastically
    transforms.RandomRotation(10),  # 10 degrees maximum
    
    # Add translation
    transforms.RandomAffine(degrees=0, translate=(0.05, 0.05)),
    
    # Add medical imaging specific adjustment
    transforms.RandomAdjustSharpness(sharpness_factor=2),
    
    # Convert to tensor (this is required for PyTorch)
    transforms.ToTensor(),
    
    # Normalize using ImageNet values
    # I'm honestly not 100% sure why these specific numbers are used
    # but they're standard for models pretrained on ImageNet
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],  # means for each color channel
        std=[0.229, 0.224, 0.225]     # standard deviations for each channel
    )
])

# For validation and testing, we don't want random augmentations
# We just want to resize and normalize the images
val_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# Using the same transforms for test as validation
test_transforms = val_transforms

################################################################################
# Loading the Dataset
################################################################################
print("Loading datasets...")  # Added to track progress

# ImageFolder is a really helpful class that automatically labels our images
# based on which subfolder they're in
try:
    train_dataset = datasets.ImageFolder(TRAIN_DIR, transform=train_transforms)
    val_dataset = datasets.ImageFolder(VAL_DIR, transform=val_transforms)
    test_dataset = datasets.ImageFolder(TEST_DIR, transform=test_transforms)
    print(f"Dataset sizes: Train={len(train_dataset)}, Val={len(val_dataset)}, Test={len(test_dataset)}")
except Exception as e:
    print(f"Error loading datasets: {e}")
    print("Make sure your dataset is in the correct location and format!")
    exit(1)

################################################################################
# Setting up Data Loaders
################################################################################
# DataLoader handles batching and shuffling for us
# num_workers helps load data faster by using multiple CPU cores
# I had some issues with num_workers on Windows, so I'm using a try-except block

print("Creating data loaders...")
try:
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,  # Shuffle training data to avoid learning order patterns
        num_workers=4  # Might need to adjust this based on your CPU
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,  # Don't need to shuffle validation data
        num_workers=4
    )

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,  # Don't shuffle test data either
        num_workers=4
    )
except Exception as e:
    print(f"Error setting up data loaders: {e}")
    print("Try reducing num_workers if you're having issues!")
    exit(1)

################################################################################
# Main execution
################################################################################
if __name__ == '__main__':
    freeze_support()
    
    # First, analyze and fix the dataset split
    print("\nAnalyzing and fixing dataset split...")
    train_counts, val_counts = create_validation_split(TRAIN_DIR, VAL_DIR, val_split=VAL_SPLIT)
    
    # Now load the datasets
    print("\nLoading datasets...")
    try:
        train_dataset = datasets.ImageFolder(TRAIN_DIR, transform=train_transforms)
        val_dataset = datasets.ImageFolder(VAL_DIR, transform=val_transforms)
        test_dataset = datasets.ImageFolder(TEST_DIR, transform=test_transforms)
        
        print("\nDataset sizes:")
        print(f"Training: {len(train_dataset)} images")
        print(f"Validation: {len(val_dataset)} images")
        print(f"Test: {len(test_dataset)} images")
        
        # Calculate class weights for handling imbalance
        print("\nCalculating class weights for handling class imbalance...")
        class_weights = calculate_class_weights(train_dataset)
        print(f"Class weights: {class_weights}")
        
    except Exception as e:
        print(f"Error loading datasets: {e}")
        print("Make sure your dataset is in the correct location and format!")
        exit(1)

    print("\nSetting up data loaders...")
    try:
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=BATCH_SIZE,
            shuffle=True,
            num_workers=4
        )

        val_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=4
        )

        test_loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=4
        )

        # Create dataloaders dictionary
        dataloaders_dict = {
            TRAIN_PHASE: train_loader,
            VAL_PHASE: val_loader
        }

    except Exception as e:
        print(f"Error setting up data loaders: {e}")
        print("Try reducing num_workers if you're having issues!")
        exit(1)

    ################################################################################
    # 1. Training from scratch
    ################################################################################
    print("\n1. Training ResNet50 from scratch...")

    model_scratch = models.resnet50(pretrained=False)
    model_scratch.fc = nn.Linear(model_scratch.fc.in_features, NUM_CLASSES)
    model_scratch = model_scratch.to(device)

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer_scratch = optim.Adam(model_scratch.parameters(), lr=LEARNING_RATE)

    # Add learning rate scheduling
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer_scratch, mode='max', patience=2)

    model_scratch, scratch_history = train_model(
        model_scratch,
        criterion,
        optimizer_scratch,
        dataloaders_dict,
        num_epochs=NUM_EPOCHS,
        model_name="resnet50_scratch"
    )

    # Plot training history
    plot_training_history(
        scratch_history,
        "ResNet50 from Scratch",
        "scratch_training_history.png"
    )

    # Evaluate on test set
    print("\nEvaluating model trained from scratch...")
    scratch_results = evaluate_model(model_scratch, test_loader)

    ################################################################################
    # 2. Fine tuning a pretrained ResNet 18
    ################################################################################
    print("\n2. Fine-tuning pretrained ResNet18...")

    model_pretrained = models.resnet18(pretrained=True)
    model_pretrained.fc = nn.Linear(model_pretrained.fc.in_features, NUM_CLASSES)
    model_pretrained = model_pretrained.to(device)

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer_pretrained = optim.Adam(model_pretrained.parameters(), lr=LEARNING_RATE)

    # Add learning rate scheduling
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer_pretrained, mode='max', patience=2)

    model_pretrained, pretrained_history = train_model(
        model_pretrained,
        criterion,
        optimizer_pretrained,
        dataloaders_dict,
        num_epochs=NUM_EPOCHS,
        model_name="resnet18_pretrained"
    )

    # Plot training history
    plot_training_history(
        pretrained_history,
        "Pretrained ResNet18",
        "pretrained_training_history.png"
    )

    # Evaluate on test set
    print("\nEvaluating pretrained model...")
    pretrained_results = evaluate_model(model_pretrained, test_loader)

    # Save final results
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    final_results = {
        'scratch_model': scratch_results,
        'pretrained_model': pretrained_results,
        'dataset_sizes': {
            TRAIN_PHASE: len(train_dataset),
            VAL_PHASE: len(val_dataset),
            TEST_PHASE: len(test_dataset)
        },
        'class_weights': class_weights.tolist(),
        'train_counts': train_counts,
        'val_counts': val_counts
    }

    with open(f'results/final_results_{timestamp}.json', 'w') as f:
        json.dump(final_results, f)

    print("\nTraining and evaluation complete! Results have been saved.")
    print("Check the 'plots' directory for training visualizations.")
    print("Check the 'results' directory for detailed metrics.")

    # After training and evaluation, add visualization
    print("\nGenerating failure case analysis...")
    visualize_model_predictions(model_scratch, test_loader, device, num_images=5)
    visualize_model_predictions(model_pretrained, test_loader, device, num_images=5)

    print("\nGenerating Grad-CAM visualizations...")
    visualize_gradcam(model_scratch, test_loader, device, num_images=5)
    visualize_gradcam(model_pretrained, test_loader, device, num_images=5)

    # For pretrained model, could freeze early layers
    def freeze_layers(model, num_layers_to_freeze):
        for i, (name, param) in enumerate(model.named_parameters()):
            if i < num_layers_to_freeze:
                param.requires_grad = False
