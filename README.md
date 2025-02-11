# Deep Learning-based Pneumonia Detection Using Chest X-Ray Images
## CAP 5516 - Medical Image Computing (Spring 2025)
### Zachary Wood
### Programming Assignment #1

## Implementation Details

### System Specifications
- Processor: Intel(R) Core(TM) i9-14900K 3.20 GHz
- RAM: 64.0 GB (63.8 GB usable)
- System Type: 64-bit operating system, x64-based processor
- GPU: CPU Only (PyTorch CPU)

### Network Architecture
- Task 1.1: ResNet50 (trained from scratch)
- Task 1.2: ResNet18 (pretrained)
- Output classes: 2 (Normal/Pneumonia)
- Final layer: Linear layer with 2 outputs

### Training Parameters
- Learning rate: 0.0001 (reduced from 0.001)
- Batch size: 64 (increased from 32)
- Training epochs: 20 (increased from 10)
- Optimizer: Adam
- Loss function: CrossEntropyLoss with class weights
- Learning rate scheduling: ReduceLROnPlateau
  - Mode: max (monitoring validation accuracy)
  - Patience: 2 epochs

### Data Augmentation
- Spatial transformations:
  - Resize to 224x224
  - Random horizontal flip (p=0.5)
  - Random rotation (±10 degrees)
  - Random translation (±5% in x and y)
- Image adjustments:
  - Random sharpness adjustment (factor=2)
  - Normalization using ImageNet statistics (mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

### Model-Specific Optimizations
- Task 1.1 (ResNet50 from scratch):
  - Larger model capacity for learning from scratch
  - No pretrained weights
  - Learning rate scheduling for better convergence

- Task 1.2 (ResNet18 pretrained):
  - ImageNet pretrained weights
  - Learning rate scheduling
  - Full fine-tuning of all layers

### Dataset Statistics
- Training set: 4,172 images
  - Normal: 1,072
  - Pneumonia: 3,100
- Validation set: 1,044 images
  - Normal: 269
  - Pneumonia: 775
- Test set: 624 images
  - Normal: 234
  - Pneumonia: 390

## Results

### Task 1.1 - Training from Scratch (ResNet50)
- Overall test accuracy: 83.01%
- Per-class accuracy:
  - Normal: 55.98%
  - Pneumonia: 99.23%
- Training characteristics:
  - Best validation accuracy: 96.17% (epoch 20)
  - Final training accuracy: 96.62%
  - Shows improved stability with longer training
  - Benefits from increased model capacity

### Task 1.2 - Fine-tuning Pretrained Model (ResNet18)
- Overall test accuracy: 85.58%
- Per-class accuracy:
  - Normal: 61.97%
  - Pneumonia: 99.74%
- Training characteristics:
  - Best validation accuracy: 98.47% (epoch 19)
  - Final training accuracy: 99.76%
  - Shows excellent convergence and stability
  - Significant improvement in Normal class detection

### Training Improvements from Parameter Updates
- Increased epochs (20 vs 10):
  - Allowed models to reach better convergence
  - Improved Normal class detection significantly
- Reduced learning rate (0.0001 vs 0.001):
  - More stable training curves
  - Better final accuracy
- Increased batch size (64 vs 32):
  - Improved training stability
  - Better generalization

### Dataset Distribution
- Training set: 4,172 images
  - Normal: 1,072 (25.7%)
  - Pneumonia: 3,100 (74.3%)
- Validation set: 1,044 images
  - Normal: 269 (25.8%)
  - Pneumonia: 775 (74.2%)
- Test set: 624 images
  - Normal: 234 (37.5%)
  - Pneumonia: 390 (62.5%)

## Analysis

### Model Performance
1. Class Imbalance:
   - Both models achieve excellent Pneumonia detection (>99%)
   - Normal case detection improved significantly:
     - ResNet50 (scratch): 55.98%
     - ResNet18 (pretrained): 61.97%
   - Class weights (Normal: 1.95, Pneumonia: 0.67) helped but imbalance effects still present

2. Model Comparison:
   - Pretrained ResNet18 performs best overall (85.58% vs 83.01%)
   - Key improvements:
     - Better Normal class detection (+5.99% absolute improvement)
     - Higher Pneumonia detection (+0.51%)
     - More stable training behavior

3. Learning Curves:
   - ResNet50 (scratch):
     - Steady improvement throughout extended training
     - Final validation accuracy: 96.17%
     - More stable with new hyperparameters
   - ResNet18 (pretrained):
     - Rapid initial convergence
     - Reached 98.47% validation accuracy
     - Maintained stability in later epochs

### Failure Case Analysis
The visualization of failure cases shows:
- Improved Normal case detection but still some systematic misclassification
- Both models show better discrimination ability
- Grad-CAM visualizations confirm appropriate feature attention

## Potential Improvements
1. Address class imbalance:
   - Collect more Normal cases
   - Try different sampling strategies
   - Experiment with other class weighting schemes

2. Model architecture:
   - Try other architectures designed for medical imaging
   - Experiment with different learning rates for different layers
   - Add more regularization to prevent overfitting

3. Training strategy:
   - Implement cross-validation
   - Try different optimizers
   - Experiment with learning rate scheduling

## Running the Code
1. Install requirements:
   ```bash
   pip install torch torchvision opencv-python numpy matplotlib sklearn
   ```

2. The dataset is already organized in the following structure:
   ```
   chest_xray/
   ├── train/
   │   ├── NORMAL/
   │   └── PNEUMONIA/
   ├── val/
   │   ├── NORMAL/
   │   └── PNEUMONIA/
   └── test/
       ├── NORMAL/
       └── PNEUMONIA/
   ```

3. Note about model files:
   - Due to size limitations, trained model files (.pth) are not included in the repository
   - Running the training script will generate new model files in the `saved_models` directory
   - Best models will be automatically saved during training
   - If you need the specific models used in our results, please contact me

4. Run the training script:
   ```bash
   python main.py
   ```

5. Check results in:
   - `plots/` for training curves
   - `failure_cases/` for misclassified examples
   - `gradcam/` for attention visualizations
   - `results/` for detailed metrics 