# Image Classification Web Application

A complete end-to-end web application demonstrating image classification with AiDotNet.

## Overview

This sample is a full-stack web application that:
1. Provides a web UI for uploading images
2. Classifies images using a trained CNN model
3. Displays predictions with confidence scores
4. Shows model training history and metrics

## Features

- **Drag & Drop Upload**: Easy image upload interface
- **Real-time Classification**: Instant predictions as images are uploaded
- **Multiple Image Support**: Batch classify multiple images
- **Confidence Visualization**: Bar charts showing class probabilities
- **Training Dashboard**: View training metrics and loss curves
- **API Endpoints**: REST API for programmatic access

## Prerequisites

- .NET 8.0 SDK or later
- AiDotNet NuGet package
- Modern web browser

## Running the Sample

```bash
cd samples/end-to-end/ImageClassificationWebApp
dotnet run
```

Open your browser to `http://localhost:5200`

## Project Structure

```
ImageClassificationWebApp/
├── Program.cs              # Application entry point
├── ImageClassificationWebApp.csproj
├── Pages/
│   ├── Index.razor        # Home page with upload UI
│   └── Training.razor     # Training dashboard
├── Components/
│   ├── ImageUploader.razor
│   ├── PredictionResult.razor
│   └── ConfidenceChart.razor
├── Services/
│   └── ClassificationService.cs
├── Models/
│   └── ImageClassificationModel.cs
└── wwwroot/
    ├── css/
    └── images/
```

## How It Works

### 1. Model Training
The application uses a pre-trained CNN model or trains one on startup:
- ResNet-18 architecture
- Trained on CIFAR-10 dataset
- 10 classes: airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck

### 2. Image Processing
Uploaded images are:
- Resized to 32x32 pixels
- Normalized using ImageNet statistics
- Converted to tensor format

### 3. Classification
The model outputs:
- Predicted class label
- Confidence score for each class
- Top-5 predictions

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/classify` | POST | Classify a single image |
| `/api/classify/batch` | POST | Classify multiple images |
| `/api/model/info` | GET | Get model information |
| `/api/model/classes` | GET | Get list of classes |

## Example API Usage

```bash
# Classify an image
curl -X POST http://localhost:5200/api/classify \
  -F "image=@cat.jpg"

# Response:
# {
#   "prediction": "cat",
#   "confidence": 0.92,
#   "probabilities": {
#     "cat": 0.92,
#     "dog": 0.05,
#     "bird": 0.02,
#     ...
#   }
# }
```

## Customization

### Using Your Own Model
```csharp
// Load a custom trained model
var modelPath = "path/to/your/model.aidotnet";
var model = PredictionModelResult<float, Tensor<float>, Tensor<float>>
    .LoadFromFile(modelPath);
```

### Adding New Classes
Modify the `ClassLabels` in `ImageClassificationModel.cs`:
```csharp
public static readonly string[] ClassLabels =
{
    "class1", "class2", "class3", ...
};
```

## Code Structure

- `Program.cs` - Complete application with embedded services
- Demonstrates Blazor Server with AiDotNet integration
- Uses DI for model lifecycle management
- Includes both UI and API endpoints

## Related Samples

- [ImageClassification](../../computer-vision/ImageClassification/) - Basic image classification
- [ModelServing](../../deployment/ModelServing/) - Production model serving
- [YOLOv8Detection](../../computer-vision/ObjectDetection/YOLOv8Detection/) - Object detection

## Learn More

- [Computer Vision Tutorial](/docs/tutorials/computer-vision/)
- [Deployment Guide](/docs/tutorials/deployment/)
- [Blazor Documentation](https://docs.microsoft.com/aspnet/core/blazor)
