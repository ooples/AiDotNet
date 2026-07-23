# YOLOv8 Object Detection

This sample demonstrates real-time object detection using YOLOv8, one of the most popular and efficient object detection architectures.

## What You'll Learn

- How to configure YOLOv8 for object detection
- How to use the `ComputerVisionPipelineFactory` facade pattern
- How to process detection results with bounding boxes
- How to tune confidence thresholds for different use cases
- How to perform batch detection for multiple images

## YOLOv8 Overview

YOLOv8 (You Only Look Once v8) is a state-of-the-art, real-time object detection model that:

- Processes images in a single forward pass (hence "You Only Look Once")
- Achieves excellent speed-accuracy trade-offs
- Supports multiple model sizes (Nano to XLarge)
- Pre-trained on COCO dataset (80 object classes)

### Model Size Comparison

| Size   | Parameters | Speed (ms)* | mAP50-95 | Best For                    |
|--------|------------|-------------|----------|------------------------------|
| Nano   | 3.2M       | 1.2         | 37.3     | Mobile, edge devices         |
| Small  | 11.2M      | 2.1         | 44.9     | Real-time applications       |
| Medium | 25.9M      | 4.0         | 50.2     | Balanced (recommended)       |
| Large  | 43.7M      | 6.8         | 52.9     | High accuracy requirements   |
| XLarge | 68.2M      | 12.3        | 53.9     | Maximum accuracy             |

*Inference time on NVIDIA V100 GPU

## Running the Sample

```bash
cd samples/computer-vision/ObjectDetection/YOLOv8Detection
dotnet run
```

## Expected Output

```
=== AiDotNet YOLOv8 Object Detection ===
Real-time object detection with bounding boxes

Available YOLOv8 model sizes:
  Nano   - 3.2M params  - Fastest inference, mobile deployment
  Small  - 11.2M params - Fast with good accuracy
  Medium - 25.9M params - Balanced speed and accuracy (recommended)
  Large  - 43.7M params - High accuracy, slower inference
  XLarge - 68.2M params - Highest accuracy, production use

Configuration:
  Architecture: YOLOv8
  Model Size: Medium
  Input Size: 640x640
  Confidence Threshold: 0.25
  NMS Threshold: 0.45
  Number of Classes: 80

Creating YOLOv8 detection pipeline...
  Pipeline created successfully

Processing Image 1: Urban scene simulation
--------------------------------------------------
  Image Size: 640x640
  Inference Time: 45.23ms
  Objects Detected: 3

  Detected Objects:
  ------------------------------------------------------------
  | Class               | Confidence | Bounding Box (x1,y1,x2,y2) |
  ------------------------------------------------------------
  | person              | 92.3%      | (100,100,200,200)          |
  | car                 | 87.5%      | (300,150,450,350)          |
  | dog                 | 78.2%      | (500,400,600,550)          |
  ------------------------------------------------------------

=== Batch Detection Demo ===
Processing multiple images in a single batch...

Batch Size: 3 images
Total Batch Inference Time: 89.45ms
Average per Image: 29.82ms
Total Objects Detected: 9

=== Confidence Threshold Tuning ===
Effect of different confidence thresholds:

  | Threshold | Detections | Notes                          |
  ------------------------------------------------------------
  | 10%       |         12 | More detections, higher FPs    |
  | 25%       |          6 | Balanced (recommended)         |
  | 50%       |          4 | Fewer but more confident       |
  | 75%       |          2 | Only high-confidence           |
  | 90%       |          1 | Very few, very confident       |
  ------------------------------------------------------------
```

## Code Highlights

### Creating a Detection Pipeline

```csharp
// Configure detection options
var options = new ObjectDetectionOptions<float>
{
    Architecture = DetectionArchitecture.YOLOv8,
    Size = ModelSize.Medium,
    ConfidenceThreshold = 0.25,
    NmsThreshold = 0.45,
    InputSize = new[] { 640, 640 }
};

// Create pipeline using factory (facade pattern)
var pipeline = ComputerVisionPipelineFactory.CreateYOLOv8Pipeline<float>(options);
```

### Detecting Objects

```csharp
// Single image detection
var result = pipeline.DetectObjects(image);

foreach (var detection in result.Detections)
{
    Console.WriteLine($"Found: {detection.ClassName}");
    Console.WriteLine($"  Confidence: {detection.Confidence:P1}");
    Console.WriteLine($"  Location: ({detection.BoundingBox.X1}, {detection.BoundingBox.Y1})");
}
```

### Batch Detection

```csharp
// Process multiple images efficiently
var batchResult = pipeline.ObjectDetector.DetectBatch(batchTensor);

foreach (var result in batchResult.Results)
{
    Console.WriteLine($"Detected {result.Detections.Count} objects");
}
```

### Visualizing Results

```csharp
// Draw bounding boxes on the image
var visualizedImage = pipeline.VisualizeDetections(image, result);
```

## Key Concepts

### Confidence Threshold

The confidence threshold controls how certain the model must be to report a detection:
- **Low (0.1-0.25)**: More detections, but more false positives
- **Medium (0.25-0.5)**: Balanced trade-off (recommended)
- **High (0.5-0.9)**: Fewer but more reliable detections

### Non-Maximum Suppression (NMS)

NMS removes duplicate detections of the same object:
- **NMS Threshold 0.45**: Standard setting, removes overlapping boxes
- **Lower threshold**: More aggressive duplicate removal
- **Higher threshold**: Keeps more overlapping detections

### COCO Classes

YOLOv8 is trained on 80 COCO object classes including:
- **People**: person
- **Vehicles**: bicycle, car, motorcycle, airplane, bus, train, truck, boat
- **Animals**: bird, cat, dog, horse, sheep, cow, elephant, bear, zebra, giraffe
- **Furniture**: chair, couch, bed, dining table
- **Electronics**: tv, laptop, mouse, remote, keyboard, cell phone
- **Food**: banana, apple, sandwich, orange, pizza, donut, cake
- And many more...

## Advanced Configuration

### Custom Classes

For custom object detection, you can fine-tune or train on your own dataset:

```csharp
var options = new ObjectDetectionOptions<float>
{
    NumClasses = 10,  // Your custom class count
    ClassNames = new[] { "class1", "class2", ... },
    UsePretrained = false  // Train from scratch
};
```

### Multi-Scale Detection

For better accuracy on images with objects at various scales:

```csharp
var options = new ObjectDetectionOptions<float>
{
    UseMultiScale = true,  // 2-3x slower but more accurate
    InputSize = new[] { 1280, 1280 }  // Higher resolution
};
```

### Different Architectures

AiDotNet supports multiple detection architectures:

```csharp
// DETR (Transformer-based)
var options = new ObjectDetectionOptions<float>
{
    Architecture = DetectionArchitecture.DETR
};

// RT-DETR (Real-Time DETR)
var options = new ObjectDetectionOptions<float>
{
    Architecture = DetectionArchitecture.RTDETR
};
```

## Performance Tips

1. **Use GPU**: Enable GPU acceleration for 10-50x faster inference
2. **Batch Processing**: Process multiple images together for better throughput
3. **Choose Right Size**: Use Nano/Small for real-time, Large/XLarge for accuracy
4. **Optimize Input**: Use 640x640 for speed, 1280x1280 for accuracy

## Next Steps

- [ImageClassification](../../ImageClassification/) - Classify entire images
- [OCR](../../OCR/) - Extract text from images
- [Instance Segmentation](../../Segmentation/) - Pixel-level object masks

## Resources

- [YOLOv8 Paper](https://docs.ultralytics.com/)
- [COCO Dataset](https://cocodataset.org/)
- [AiDotNet Computer Vision Documentation](../../../../docs/computer-vision/)
