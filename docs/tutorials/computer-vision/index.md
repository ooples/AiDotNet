---
layout: default
title: Computer Vision
parent: Tutorials
nav_order: 4
has_children: true
permalink: /tutorials/computer-vision/
---

# Computer Vision Tutorial
{: .no_toc }

Build powerful image and video understanding models with AiDotNet.
{: .fs-6 .fw-300 }

---

## Table of contents
{: .no_toc .text-delta }

1. TOC
{:toc}

---

## Overview

AiDotNet provides 50+ computer vision models for:
- Object Detection (YOLO, DETR, Faster R-CNN)
- Image Classification (ResNet, EfficientNet, ViT)
- Instance Segmentation (Mask R-CNN, SAM)
- OCR (CRNN, TrOCR)
- And more!

---

## Image Classification

### Using Pre-trained Models

```csharp
using AiDotNet.ComputerVision;

// Load a pre-trained ResNet-50
var classifier = await ImageClassifier.LoadAsync<float>("resnet50");

// Classify an image
var image = await Image.LoadAsync("cat.jpg");
var prediction = classifier.Classify(image);

Console.WriteLine($"Prediction: {prediction.Label}");
Console.WriteLine($"Confidence: {prediction.Confidence:P1}");
```

### Training a Custom Classifier

```csharp
using AiDotNet;
using AiDotNet.NeuralNetworks.Architectures;

// Configure a CNN
var model = new ResNet<float>(new ResNetConfig<float>
{
    Variant = ResNetVariant.ResNet18,
    NumClasses = 10,
    InputChannels = 3,
    InputHeight = 224,
    InputWidth = 224
});

// Build and train
var result = await new AiModelBuilder<float, Tensor<float>, Tensor<float>>()
    .ConfigureModel(model)
    .ConfigureOptimizer(new AdamOptimizer<float>(learningRate: 1e-4f))
    .ConfigureDataAugmentation(new ImageAugmentationConfig
    {
        RandomHorizontalFlip = true,
        RandomRotation = 15,
        ColorJitter = true
    })
    .ConfigureGpuAcceleration()
    .BuildAsync(trainImages, trainLabels);
```

---

## Object Detection

### YOLOv8 Detection

```csharp
using AiDotNet.ComputerVision;

// Create detector
var detector = new YOLOv8Detector<float>(
    modelPath: "yolov8n.onnx",
    confidenceThreshold: 0.5f,
    nmsThreshold: 0.45f);

// Detect objects
var image = await Image.LoadAsync("street.jpg");
var detections = detector.Detect(image);

foreach (var det in detections)
{
    Console.WriteLine($"{det.Label}: {det.Confidence:P0}");
    Console.WriteLine($"  Box: {det.BoundingBox}");
}
```

### Available Detection Models

| Model | Description | Speed | Accuracy |
|:------|:------------|:------|:---------|
| YOLOv8n | Nano - fastest | ⚡⚡⚡⚡ | ⭐⭐ |
| YOLOv8s | Small | ⚡⚡⚡ | ⭐⭐⭐ |
| YOLOv8m | Medium | ⚡⚡ | ⭐⭐⭐⭐ |
| YOLOv8l | Large | ⚡ | ⭐⭐⭐⭐⭐ |
| DETR | Transformer-based | ⚡⚡ | ⭐⭐⭐⭐⭐ |

### Training Custom Object Detection

```csharp
var detector = new YOLOv8<float>(new YOLOConfig<float>
{
    NumClasses = 5,
    ImageSize = 640
});

await detector.TrainAsync(
    trainDataset,
    epochs: 100,
    batchSize: 16,
    learningRate: 0.01f);
```

---

## Instance Segmentation

### Using Mask R-CNN

```csharp
using AiDotNet.ComputerVision;

var segmenter = new MaskRCNN<float>(numClasses: 80);

var image = await Image.LoadAsync("people.jpg");
var instances = segmenter.Segment(image);

foreach (var instance in instances)
{
    Console.WriteLine($"{instance.Label}: {instance.Confidence:P0}");
    Console.WriteLine($"  Mask pixels: {instance.Mask.Sum()}");
}
```

### Segment Anything Model (SAM)

```csharp
var sam = await SAM.LoadAsync<float>("sam_vit_h");

// Segment with point prompts
var masks = sam.Segment(image, points: [(512, 384)]);

// Segment with box prompt
var masks2 = sam.Segment(image, box: new Box(100, 100, 400, 400));
```

---

## OCR (Text Recognition)

### Basic OCR

```csharp
using AiDotNet.ComputerVision;

var ocr = new OCREngine<float>();

var image = await Image.LoadAsync("document.png");
var result = ocr.Recognize(image);

Console.WriteLine("Extracted text:");
Console.WriteLine(result.Text);

// With bounding boxes
foreach (var line in result.Lines)
{
    Console.WriteLine($"[{line.BoundingBox}] {line.Text}");
}
```

### Scene Text Recognition

```csharp
var sceneOCR = new SceneTextRecognizer<float>();

var image = await Image.LoadAsync("street_sign.jpg");
var texts = sceneOCR.Detect(image);

foreach (var text in texts)
{
    Console.WriteLine($"'{text.Content}' at {text.Location}");
}
```

---

## Data Augmentation

```csharp
.ConfigureDataAugmentation(new ImageAugmentationConfig
{
    // Geometric transforms
    RandomHorizontalFlip = true,
    RandomVerticalFlip = false,
    RandomRotation = 15,  // degrees
    RandomCrop = 0.8f,    // min scale

    // Color transforms
    ColorJitter = true,
    Brightness = 0.2f,
    Contrast = 0.2f,
    Saturation = 0.2f,
    Hue = 0.1f,

    // Other
    RandomErasing = true,
    Mixup = 0.2f,
    CutMix = 0.2f
})
```

---

## Transfer Learning

Use pre-trained weights and fine-tune on your data:

```csharp
var model = await ResNet.LoadPretrainedAsync<float>(
    "resnet50",
    weights: "imagenet");

// Freeze backbone
model.FreezeBackbone();

// Replace classification head
model.SetNumClasses(5);

// Train
await model.TrainAsync(myData, epochs: 10);
```

---

## GPU Acceleration

```csharp
.ConfigureGpuAcceleration(new GpuAccelerationConfig
{
    Enabled = true,
    DeviceId = 0,
    MixedPrecision = true  // FP16 for faster training
})
```

---

## Batch Processing

```csharp
// Process multiple images efficiently
var images = await Task.WhenAll(
    imagePaths.Select(p => Image.LoadAsync(p)));

var results = detector.DetectBatch(images, batchSize: 32);
```

---

## Best Practices

1. **Resize consistently**: Use the same input size as training
2. **Normalize correctly**: Match the preprocessing of pre-trained models
3. **Use data augmentation**: Prevents overfitting, improves generalization
4. **Start with pre-trained**: Fine-tuning is usually faster than training from scratch
5. **Monitor GPU memory**: Reduce batch size if OOM errors occur

---

## Next Steps

- [YOLOv8 Sample](/samples/computer-vision/ObjectDetection/YOLOv8Detection/)
- [Image Classification Sample](/samples/computer-vision/ImageClassification/)
- [OCR Sample](/samples/computer-vision/OCR/)
- [Computer Vision API Reference](/api/AiDotNet.ComputerVision/)
