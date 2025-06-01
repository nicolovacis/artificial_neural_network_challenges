# artificial_neural_network_challenges

This repository contains two deep learning projects developed for the **Advanced Neural Network and Deep Learning (AN2DL)** course at Politecnico di Milano. Each subproject addresses a supervised learning task using artificial neural networks: one for image classification, the other for image segmentation.

---

## Project Structure

```bash
artificial_neural_network_challenges/
│
├── Classification/ # Blood cell classification task
│ ├── model.py/ # Model architectures (MobileNet, VGG19, ConvNeXt, etc.)
│ ├── preprocessing/ # Outlier removal, augmentation, dataset handling
│ ├── train/ # Training scripts, fine-tuning, Keras Tuner configs
│ ├── test/ # Inference and evaluation code
│ └── AN2DL_Report_.pdf # Detailed project report
│
├── Segmentation/ # Mars terrain segmentation task
│ ├── Models/ # U-Net++, custom blocks (SE, global context, etc.)
│ ├── PreProcessing/ # Augmentation, copy-paste balancing
│ ├── FineTuning/ # Learning rate schedulers, class weights, loss design
│ ├── Submission/ # Code for final output generation
│ └── AN2DL_Homework2_Report_.pdf # Detailed project report
```


---

## Summary of Work

### Blood Cell Classification

- **Task**: Classify 8 types of blood cells from ~14,000 labeled images.
- **Methods**:
  - Outlier detection using cosine similarity on VGG embeddings.
  - Extensive data augmentation (CutMix, MixUp, color degradation).
  - Pretrained networks (MobileNet, VGG19, EfficientNet, ConvNeXt) with custom dense heads.
  - Hyperparameter tuning using Keras Tuner.
  - Fine-tuning with layer unfreezing, class balancing, and dynamic learning rate.
- **Best Result**: 81% test accuracy with ConvNeXt.

### Mars Terrain Segmentation

- **Task**: Segment grayscale Mars images into 5 classes.
- **Methods**:
  - Class imbalance handled via image duplication and pixel-level copy-paste.
  - Architectures: U-Net++, SegNet, DeepLab, Mask R-CNN.
  - Custom U-Net++ with dilated convolutions, squeeze-and-excitation, and global context blocks.
  - Hybrid loss function combining CrossEntropy, Dice, Focal, and Boundary losses.
- **Best Result**: 0.563 mean IoU (Kaggle leaderboard).

---

## Tech Stack

- Python (TensorFlow/Keras)
- Albumentations, KerasCV
- Keras Tuner for hyperparameter optimization
- Custom attention modules and residual connections

---

## Reports

Each project folder contains a PDF report with problem analysis, architectural design, experimental results, and conclusions.

---

## Authors

- Nicolò Vacis  
- Giovanni Vaccarino  
- Vittorio Palladino  
- (Segmentation only) Maria Fernanda Molina Ron