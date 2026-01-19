# Optical Character Recognition (OCR)

This sample demonstrates OCR capabilities for extracting text from images, including text localization, recognition, and multi-language support.

## What You'll Learn

- How to configure OCR for scene text and documents
- How to detect and localize text regions in images
- How to recognize text with confidence scores
- How to handle multiple languages
- How to process rotated and curved text

## OCR Pipeline Components

### Text Detection Models

Text detection finds regions in an image that contain text:

| Model  | Description                               | Best For                     |
|--------|-------------------------------------------|------------------------------|
| DBNet  | Differentiable Binarization Network       | Fast, general purpose        |
| CRAFT  | Character Region Awareness                | Curved/artistic text         |
| EAST   | Efficient and Accurate Scene Text         | Real-time applications       |

### Text Recognition Models

Text recognition reads the text within detected regions:

| Model | Description                              | Accuracy | Speed  |
|-------|------------------------------------------|----------|--------|
| CRNN  | Convolutional Recurrent Neural Network   | Good     | Fast   |
| TrOCR | Transformer-based OCR                    | Higher   | Slower |

## Running the Sample

```bash
cd samples/computer-vision/OCR
dotnet run
```

## Expected Output

```
=== AiDotNet Optical Character Recognition (OCR) ===
Extract text from images with localization and multi-language support

Available Text Detection Models:
  DBNet    - Differentiable Binarization Network (fast, accurate)
  CRAFT    - Character Region Awareness (good for curved text)
  EAST     - Efficient and Accurate Scene Text detector

Configuration:
  Mode: SceneText
  Detection Model: DBNet
  Recognition Model: CRNN
  Confidence Threshold: 0.5
  Supported Languages: en, es, fr, de, zh, ja

Processing: Simple sign (single line)
======================================================================
  Image Size: 640x480
  Inference Time: 45.23ms
  Text Regions Found: 1

  Detected Text Regions:
  -----------------------------------------------------------------
  | # | Text                         | Conf  | Location (x,y,w,h)  |
  -----------------------------------------------------------------
  | 1 | WELCOME TO AIDOTNET          | 95%   | (100,200,440,80)    |
  -----------------------------------------------------------------

  Full Extracted Text:
  "WELCOME TO AIDOTNET"

Processing: Multi-line document
======================================================================
  Text Regions Found: 4

  Full Extracted Text:
  "Chapter 1: Introduction Machine learning is transforming..."
```

## Code Highlights

### Creating an OCR Pipeline

```csharp
// Configure OCR options
var options = new OCROptions<float>
{
    Mode = OCRMode.SceneText,
    DetectionModel = TextDetectionModel.DBNet,
    RecognitionModel = TextRecognitionModel.CRNN,
    SupportedLanguages = new[] { "en", "es", "fr", "de" },
    ConfidenceThreshold = 0.5f,
    DetectOrientation = true,
    GroupTextLines = true
};

// Create OCR pipeline using facade pattern
var pipeline = ComputerVisionPipelineFactory.CreateSceneTextPipeline<float>(options);
```

### Recognizing Text

```csharp
// Recognize text in image
var result = pipeline.RecognizeText(imageTensor);

// Get full extracted text
Console.WriteLine($"Full text: {result.FullText}");

// Process individual text regions
foreach (var region in result.TextRegions)
{
    Console.WriteLine($"Text: {region.Text}");
    Console.WriteLine($"  Confidence: {region.Confidence:P1}");
    Console.WriteLine($"  Box: ({region.Box.X1}, {region.Box.Y1}, " +
                     $"{region.Box.Width}, {region.Box.Height})");
}
```

### Visualizing Results

```csharp
// Draw bounding boxes and text on image
var visualized = pipeline.VisualizeOCR(image, result);
```

## OCR Modes

### Scene Text OCR

Optimized for text in natural images (signs, billboards, product labels):

```csharp
var options = new OCROptions<float>
{
    Mode = OCRMode.SceneText,
    DetectOrientation = true,  // Handle rotated text
    CorrectSkew = true         // Fix slightly tilted text
};
```

### Document OCR

Optimized for scanned documents and printed text:

```csharp
var options = new OCROptions<float>
{
    Mode = OCRMode.Document,
    GroupTextLines = true,     // Merge into paragraphs
    MaxSequenceLength = 200    // Longer text lines
};
```

## Multi-Language Support

AiDotNet OCR supports multiple languages:

| Language   | Code | Script      | Notes                    |
|------------|------|-------------|--------------------------|
| English    | en   | Latin       | Default                  |
| Spanish    | es   | Latin       | Includes accents         |
| French     | fr   | Latin       | Includes accents         |
| German     | de   | Latin       | Includes umlauts         |
| Chinese    | zh   | Simplified  | Character-based          |
| Japanese   | ja   | Mixed       | Hiragana/Katakana/Kanji  |
| Korean     | ko   | Hangul      | Syllabic blocks          |
| Arabic     | ar   | Arabic      | Right-to-left            |

### Configuring Languages

```csharp
var options = new OCROptions<float>
{
    SupportedLanguages = new[] { "en", "es", "fr" }
};
```

## Text Localization

### Bounding Boxes

For regular text, bounding boxes are returned:

```
(x1, y1) -------- (x2, y1)
    |                |
    |     TEXT       |
    |                |
(x1, y2) -------- (x2, y2)
```

### Polygons

For rotated or curved text, polygon coordinates are returned:

```csharp
// 4-point polygon for rotated text
region.Polygon = new List<(float X, float Y)>
{
    (topLeftX, topLeftY),
    (topRightX, topRightY),
    (bottomRightX, bottomRightY),
    (bottomLeftX, bottomLeftY)
};
```

## Post-Processing Options

| Option              | Description                                      |
|---------------------|--------------------------------------------------|
| Spell Correction    | Fix common OCR errors using dictionary           |
| Text Merging        | Combine nearby regions into sentences/paragraphs |
| Layout Analysis     | Detect columns, tables, headers                  |
| Language Detection  | Auto-detect text language per region             |
| Reading Order       | Sort text in natural reading order               |

### Example: Spell Correction

```csharp
// Enable spell correction post-processing
options.PostProcessing = new PostProcessingOptions
{
    SpellCorrection = true,
    Dictionary = "en-US",
    ConfidenceThreshold = 0.7  // Only correct low-confidence text
};
```

## Performance Optimization

### Speed vs Accuracy Trade-offs

| Configuration                    | Speed    | Accuracy |
|----------------------------------|----------|----------|
| DBNet + CRNN (default)           | Fast     | Good     |
| DBNet + TrOCR                    | Medium   | Higher   |
| CRAFT + TrOCR                    | Slower   | Highest  |

### Batch Processing

Process multiple images efficiently:

```csharp
// Batch OCR
var results = ocr.RecognizeBatch(listOfImages);

// Results maintain input order
for (int i = 0; i < results.Count; i++)
{
    Console.WriteLine($"Image {i}: {results[i].FullText}");
}
```

### GPU Acceleration

```csharp
// Enable GPU for faster inference
options.UseGpu = true;
options.GpuDeviceId = 0;
```

## Common Use Cases

### 1. Document Digitization

```csharp
// Configure for scanned documents
var options = new OCROptions<float>
{
    Mode = OCRMode.Document,
    RecognitionModel = TextRecognitionModel.TrOCR,
    GroupTextLines = true
};

var result = ocr.RecognizeText(scannedDocument);
File.WriteAllText("output.txt", result.FullText);
```

### 2. License Plate Recognition

```csharp
// Configure for license plates
var options = new OCROptions<float>
{
    CharacterSet = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789",
    ConfidenceThreshold = 0.8  // High confidence for plates
};
```

### 3. Receipt/Invoice Processing

```csharp
// Extract structured data from receipts
var result = ocr.RecognizeText(receiptImage);

// Parse extracted text for amounts, dates, etc.
var lines = result.TextRegions
    .OrderBy(r => r.Box?.Y1)
    .Select(r => r.Text);
```

## Error Handling

### Low Confidence Results

```csharp
foreach (var region in result.TextRegions)
{
    if (region.Confidence < 0.7)
    {
        Console.WriteLine($"Low confidence ({region.Confidence:P0}): {region.Text}");
        // Consider manual review or spell correction
    }
}
```

### Handling Different Orientations

```csharp
// Auto-detect and correct orientation
options.DetectOrientation = true;
options.CorrectSkew = true;

// For specific rotation
if (result.DetectedRotation != 0)
{
    Console.WriteLine($"Image was rotated {result.DetectedRotation} degrees");
}
```

## Comparison with Other OCR Tools

| Feature              | AiDotNet OCR | Tesseract | Google Vision |
|----------------------|--------------|-----------|---------------|
| Scene Text           | Excellent    | Limited   | Excellent     |
| Document OCR         | Good         | Excellent | Excellent     |
| Multi-language       | Yes          | Yes       | Yes           |
| Curved Text          | Yes          | No        | Yes           |
| Local/Offline        | Yes          | Yes       | No            |
| GPU Acceleration     | Yes          | Limited   | N/A (Cloud)   |

## Next Steps

- [YOLOv8Detection](../ObjectDetection/YOLOv8Detection/) - Detect objects
- [ImageClassification](../ImageClassification/) - Classify images
- [Document AI](../../document/) - Advanced document processing

## Resources

- [DBNet Paper](https://arxiv.org/abs/1911.08947)
- [CRNN Paper](https://arxiv.org/abs/1507.05717)
- [TrOCR Paper](https://arxiv.org/abs/2109.10282)
- [CRAFT Paper](https://arxiv.org/abs/1904.01941)
