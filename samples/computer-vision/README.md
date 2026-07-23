# Computer Vision Samples

This directory contains examples of computer vision models in AiDotNet.

## Available Samples

| Sample | Description |
|--------|-------------|
| [YOLOv8Detection](./ObjectDetection/YOLOv8Detection/) | Object detection with YOLOv8 |
| [ImageClassification](./ImageClassification/) | CNN-based image classification |
| [OCR](./OCR/) | Optical character recognition |

## Quick Start

```csharp
using AiDotNet;
using AiDotNet.ComputerVision;

var detector = new YOLOv8Detector<float>(
    modelPath: "yolov8n.onnx",
    confidenceThreshold: 0.5f);

var image = LoadImage("photo.jpg");
var detections = detector.Detect(image);

foreach (var det in detections)
{
    Console.WriteLine($"{det.Label}: {det.Confidence:P0} at {det.BoundingBox}");
}
```

## Computer Vision Models (50+)

### Object Detection
- YOLO v8, v9, v10, v11
- DETR (Detection Transformer)
- Faster R-CNN
- RetinaNet
- SSD

### Image Classification
- ResNet (18, 34, 50, 101, 152)
- EfficientNet
- Vision Transformer (ViT)
- ConvNeXt

### Segmentation
- Mask R-CNN
- U-Net
- DeepLab v3+
- SAM (Segment Anything)

### OCR
- CRNN
- TrOCR
- EasyOCR compatible

## Learn More

- [Computer Vision Tutorial](/docs/tutorials/computer-vision/)
- [API Reference](/api/AiDotNet.ComputerVision/)
