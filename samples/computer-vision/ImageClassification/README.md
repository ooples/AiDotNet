# Image Classification with CNN

This sample demonstrates image classification using Convolutional Neural Networks (CNNs) like ResNet and EfficientNet, including transfer learning and data augmentation techniques.

## What You'll Learn

- How to configure CNN architectures for image classification
- How to use pre-trained models (transfer learning)
- How to apply data augmentation for better generalization
- How to get top-k predictions with confidence scores
- How to fine-tune models on custom datasets

## Supported Architectures

### ResNet Family

ResNet (Residual Networks) introduced skip connections that enable training of very deep networks:

| Model      | Parameters | Top-1 Acc | Top-5 Acc | Best For                   |
|------------|------------|-----------|-----------|----------------------------|
| ResNet-18  | 11.7M      | 69.8%     | 89.1%     | Fast inference, mobile     |
| ResNet-34  | 21.8M      | 73.3%     | 91.4%     | Balanced                   |
| ResNet-50  | 25.6M      | 76.1%     | 92.9%     | Popular choice             |
| ResNet-101 | 44.5M      | 77.4%     | 93.5%     | High accuracy              |
| ResNet-152 | 60.2M      | 78.3%     | 94.2%     | Highest ResNet accuracy    |

### EfficientNet Family

EfficientNet uses compound scaling to balance network depth, width, and resolution:

| Model          | Parameters | Top-1 Acc | Top-5 Acc | Best For                   |
|----------------|------------|-----------|-----------|----------------------------|
| EfficientNet-B0| 5.3M       | 77.1%     | 93.3%     | Most parameter efficient   |
| EfficientNet-B4| 19.3M      | 82.9%     | 96.4%     | Good accuracy/speed        |
| EfficientNet-B7| 66.3M      | 84.3%     | 97.0%     | Top accuracy               |

## Running the Sample

```bash
cd samples/computer-vision/ImageClassification
dotnet run
```

## Expected Output

```
=== AiDotNet Image Classification ===
CNN-based image classification with ResNet/EfficientNet

Available Classification Architectures:
  ResNet-18    -  11.7M params - Fast, good accuracy
  ResNet-50    -  25.6M params - Popular choice
  EfficientNet-B0 - 5.3M params - Most efficient

Configuration:
  Architecture: ResNet50
  Input Size: 224x224
  Number of Classes: 1000
  Top-K Predictions: 5

Creating image classifier...
  Classifier created successfully

Classifying: Animal-like image (warm colors)
------------------------------------------------------------
  Inference Time: 12.45ms

  Top-5 Predictions:
  -------------------------------------------------------
  | Rank | Class                      | Confidence |
  -------------------------------------------------------
  |    1 | lion                       | 42.35%     |
  |    2 | tiger                      | 28.12%     |
  |    3 | tabby cat                  | 15.67%     |
  |    4 | golden retriever           |  8.43%     |
  |    5 | Persian cat                |  5.43%     |
  -------------------------------------------------------

=== Data Augmentation Demo ===
Common augmentations for training image classifiers:

  | Augmentation         | Description                                    |
  -------------------------------------------------------------------------
  | RandomHorizontalFlip | Flip image horizontally with 50% probability   |
  | RandomRotation       | Rotate image by random angle (-30 to +30 deg)  |
  | RandomResizedCrop    | Crop random region and resize to target size   |
  | ColorJitter          | Randomly adjust brightness, contrast, sat      |
  | MixUp                | Blend two images with their labels             |
  -------------------------------------------------------------------------
```

## Code Highlights

### Creating a Classifier

```csharp
// Configure classification
var config = new ImageClassificationConfig
{
    Architecture = ClassificationArchitecture.ResNet50,
    NumClasses = 1000,  // ImageNet classes
    InputSize = 224,
    UsePretrained = true,
    TopK = 5
};

// Create classifier
var classifier = new ImageClassifier(config);
```

### Classifying Images

```csharp
// Classify single image
var result = classifier.Classify(image);

// Display top predictions
foreach (var prediction in result.TopPredictions)
{
    Console.WriteLine($"{prediction.ClassName}: {prediction.Confidence:P2}");
}
```

### Data Augmentation Pipeline

```csharp
// Training augmentation pipeline
var augmentedImage = image
    .RandomHorizontalFlip(probability: 0.5)
    .RandomRotation(degrees: 30)
    .RandomResizedCrop(size: 224, scale: (0.8, 1.0))
    .ColorJitter(brightness: 0.2, contrast: 0.2, saturation: 0.2)
    .RandomErasing(probability: 0.25)
    .Normalize(mean: ImageNetMean, std: ImageNetStd);
```

### Transfer Learning

```csharp
// Load pre-trained model and fine-tune
var model = new ResNet50<float>(numClasses: 10, pretrained: true);
model.FreezeBackbone();  // Only train classifier head

// Replace classifier for custom classes
model.ReplaceClassifier(new Sequential<float>(
    new Linear<float>(2048, 512),
    new ReLU<float>(),
    new Dropout<float>(0.5),
    new Linear<float>(512, numClasses)
));

// Train
var result = await new PredictionModelBuilder<float, Tensor<float>, int>()
    .ConfigureModel(model)
    .ConfigureOptimizer(new Adam<float>(learningRate: 0.001))
    .BuildAsync(trainImages, trainLabels);
```

## Data Augmentation Techniques

### Basic Augmentations

| Technique             | Description                                      | Use Case                    |
|-----------------------|--------------------------------------------------|------------------------------|
| RandomHorizontalFlip  | Flip horizontally with probability               | Most image tasks             |
| RandomVerticalFlip    | Flip vertically with probability                 | Satellite/medical imaging    |
| RandomRotation        | Rotate by random angle                           | Rotation-invariant tasks     |
| RandomResizedCrop     | Random crop and resize                           | Scale invariance             |
| CenterCrop            | Crop center of image                             | Inference preprocessing      |

### Color Augmentations

| Technique       | Parameters                            | Effect                           |
|-----------------|---------------------------------------|----------------------------------|
| ColorJitter     | brightness, contrast, saturation, hue | Simulates lighting variations    |
| GaussianBlur    | kernel_size, sigma                    | Reduces overfitting to textures  |
| RandomGrayscale | probability                           | Color invariance                 |
| Normalize       | mean, std                             | Standardizes input distribution  |

### Advanced Augmentations

| Technique   | Description                                          | Benefit                     |
|-------------|------------------------------------------------------|------------------------------|
| MixUp       | Blend two images and their labels                    | Smoother decision boundaries |
| CutMix      | Replace image region with another image's patch      | Better localization          |
| AutoAugment | Learned augmentation policy                          | Optimal for specific tasks   |
| RandAugment | Random augmentation with magnitude                   | Simple and effective         |

## ImageNet Preprocessing

Standard ImageNet preprocessing:

```csharp
// ImageNet normalization values
var ImageNetMean = new[] { 0.485f, 0.456f, 0.406f };
var ImageNetStd = new[] { 0.229f, 0.224f, 0.225f };

// Preprocessing pipeline
var preprocessed = image
    .Resize(256)                    // Resize shorter side to 256
    .CenterCrop(224)                // Crop center 224x224
    .ToTensor()                     // Convert to tensor [0, 1]
    .Normalize(ImageNetMean, ImageNetStd);  // Normalize
```

## Fine-Tuning Strategies

### Strategy 1: Feature Extraction (Frozen Backbone)

Best when: Small dataset, similar to ImageNet

```csharp
model.FreezeBackbone();
// Only classifier head is trained
// Fast training, lower risk of overfitting
```

### Strategy 2: Fine-Tuning (Gradual Unfreezing)

Best when: Medium dataset, somewhat different from ImageNet

```csharp
// Phase 1: Train classifier head
model.FreezeBackbone();
Train(epochs: 5, lr: 0.001);

// Phase 2: Unfreeze and fine-tune
model.UnfreezeBackbone();
Train(epochs: 10, lr: 0.0001);  // Lower learning rate
```

### Strategy 3: Full Training

Best when: Large dataset, very different from ImageNet

```csharp
var model = new ResNet50<float>(pretrained: false);
// Train from scratch with full learning rate
```

## Performance Optimization

### Inference Speed

1. **Use smaller models**: EfficientNet-B0 or ResNet-18 for real-time
2. **Reduce input size**: 160x160 instead of 224x224
3. **Use GPU**: Enable CUDA/cuDNN acceleration
4. **Batch processing**: Process multiple images together

### Training Efficiency

1. **Mixed precision**: Use FP16 for 2x speedup
2. **Gradient checkpointing**: Trade compute for memory
3. **Learning rate scheduling**: Use warmup + cosine decay
4. **Early stopping**: Stop when validation loss plateaus

## Common Issues and Solutions

### Overfitting

- Add more augmentation (MixUp, CutMix)
- Use dropout and weight decay
- Reduce model size
- Get more training data

### Underfitting

- Increase model capacity
- Train longer
- Reduce regularization
- Check data quality

### Class Imbalance

- Use weighted loss function
- Oversample minority classes
- Use focal loss

## Next Steps

- [YOLOv8Detection](../ObjectDetection/YOLOv8Detection/) - Detect multiple objects
- [OCR](../OCR/) - Extract text from images
- [Instance Segmentation](../Segmentation/) - Pixel-level object masks

## Resources

- [ResNet Paper](https://arxiv.org/abs/1512.03385)
- [EfficientNet Paper](https://arxiv.org/abs/1905.11946)
- [ImageNet Dataset](https://www.image-net.org/)
- [Data Augmentation Survey](https://arxiv.org/abs/2106.07085)
