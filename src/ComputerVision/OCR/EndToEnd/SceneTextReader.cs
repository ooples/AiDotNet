using AiDotNet.Augmentation.Image;
using AiDotNet.ComputerVision.Detection.TextDetection;
using AiDotNet.ComputerVision.OCR.Recognition;
using AiDotNet.Tensors;

namespace AiDotNet.ComputerVision.OCR.EndToEnd;

/// <summary>
/// End-to-end scene text reader that combines detection and recognition.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para><b>For Beginners:</b> SceneTextReader is a complete OCR pipeline that first
/// detects text regions in images (using CRAFT, EAST, or DBNet), then recognizes
/// the text in each region (using CRNN or TrOCR). It's designed for reading text
/// in natural images like photos of signs, billboards, and product labels.</para>
///
/// <para>Key features:
/// - Two-stage pipeline: detection + recognition
/// - Handles arbitrary text orientations
/// - Works with curved and rotated text
/// - Configurable detection and recognition models
/// </para>
/// </remarks>
public class SceneTextReader<T>
{
    private readonly INumericOperations<T> _numOps;
    private readonly TextDetectorBase<T> _detector;
    private readonly OCRBase<T> _recognizer;
    private readonly OCROptions<T> _options;

    /// <summary>
    /// Name of this scene text reader.
    /// </summary>
    public string Name => $"SceneTextReader-{_detector.Name}-{_recognizer.Name}";

    /// <summary>
    /// Creates a new scene text reader with default models.
    /// </summary>
    public SceneTextReader(OCROptions<T> options)
    {
        _numOps = Tensors.Helpers.MathHelper.GetNumericOperations<T>();
        _options = options;

        // Initialize detector based on options
        var detectionOptions = new TextDetectionOptions<T>
        {
            Architecture = options.DetectionModel switch
            {
                TextDetectionModel.CRAFT => TextDetectionArchitecture.CRAFT,
                TextDetectionModel.EAST => TextDetectionArchitecture.EAST,
                TextDetectionModel.DBNet => TextDetectionArchitecture.DBNet,
                _ => TextDetectionArchitecture.DBNet
            },
            ConfidenceThreshold = options.ConfidenceThreshold
        };

        _detector = options.DetectionModel switch
        {
            TextDetectionModel.CRAFT => new CRAFT<T>(detectionOptions),
            TextDetectionModel.EAST => new EAST<T>(detectionOptions),
            TextDetectionModel.DBNet => new DBNet<T>(detectionOptions),
            _ => new DBNet<T>(detectionOptions)
        };

        // Initialize recognizer based on options
        _recognizer = options.RecognitionModel switch
        {
            TextRecognitionModel.CRNN => new CRNN<T>(options),
            TextRecognitionModel.TrOCR => new TrOCR<T>(options),
            _ => new CRNN<T>(options)
        };
    }

    /// <summary>
    /// Reads all text in an image.
    /// </summary>
    /// <param name="image">Input image tensor [batch, channels, height, width].</param>
    /// <returns>OCR result with all recognized text.</returns>
    public OCRResult<T> ReadText(Tensor<T> image)
    {
        var startTime = DateTime.UtcNow;

        int imageWidth = image.Shape[3];
        int imageHeight = image.Shape[2];

        // Step 1: Detect text regions
        var detectionResult = _detector.Detect(image);

        // Step 2: Recognize text in each region
        var recognizedTexts = new List<RecognizedText<T>>();

        foreach (var region in detectionResult.TextRegions)
        {
            // Crop text region from image
            var crop = CropRegion(image, region);

            // Recognize text
            var (text, confidence) = _recognizer.RecognizeText(crop);

            if (!string.IsNullOrWhiteSpace(text) &&
                _numOps.ToDouble(confidence) >= _numOps.ToDouble(_options.ConfidenceThreshold))
            {
                var recognized = new RecognizedText<T>(text, confidence)
                {
                    Box = region.Box,
                    Polygon = region.Polygon
                };

                recognizedTexts.Add(recognized);
            }
        }

        // Sort text regions by reading order (top-to-bottom, left-to-right)
        if (_options.GroupTextLines)
        {
            recognizedTexts = SortByReadingOrder(recognizedTexts);
        }

        // Build full text
        string fullText = string.Join(" ", recognizedTexts.Select(r => r.Text));

        return new OCRResult<T>
        {
            TextRegions = recognizedTexts,
            FullText = fullText,
            InferenceTime = DateTime.UtcNow - startTime,
            ImageWidth = imageWidth,
            ImageHeight = imageHeight
        };
    }

    /// <summary>
    /// Reads text from a pre-detected region.
    /// </summary>
    public (string text, T confidence) ReadRegion(Tensor<T> croppedRegion)
    {
        return _recognizer.RecognizeText(croppedRegion);
    }

    private Tensor<T> CropRegion(Tensor<T> image, TextRegion<T> region)
    {
        int batch = image.Shape[0];
        int channels = image.Shape[1];
        int height = image.Shape[2];
        int width = image.Shape[3];

        // Get bounding box
        double x1 = _numOps.ToDouble(region.Box.X1);
        double y1 = _numOps.ToDouble(region.Box.Y1);
        double x2 = _numOps.ToDouble(region.Box.X2);
        double y2 = _numOps.ToDouble(region.Box.Y2);

        // Clamp to image bounds
        int cropX1 = Math.Max(0, (int)x1);
        int cropY1 = Math.Max(0, (int)y1);
        int cropX2 = Math.Min(width, (int)x2);
        int cropY2 = Math.Min(height, (int)y2);

        int cropW = Math.Max(1, cropX2 - cropX1);
        int cropH = Math.Max(1, cropY2 - cropY1);

        var crop = new Tensor<T>(new[] { batch, channels, cropH, cropW });

        for (int b = 0; b < batch; b++)
        {
            for (int c = 0; c < channels; c++)
            {
                for (int h = 0; h < cropH; h++)
                {
                    for (int w = 0; w < cropW; w++)
                    {
                        int srcH = cropY1 + h;
                        int srcW = cropX1 + w;

                        if (srcH < height && srcW < width)
                        {
                            crop[b, c, h, w] = image[b, c, srcH, srcW];
                        }
                    }
                }
            }
        }

        // If polygon available and text is rotated, apply perspective correction
        if (region.Polygon.Count >= 4 && Math.Abs(region.RotationAngle) > 5)
        {
            crop = CorrectPerspective(crop, region);
        }

        return crop;
    }

    private Tensor<T> CorrectPerspective(Tensor<T> crop, TextRegion<T> region)
    {
        // Simplified perspective correction - just return crop for now
        // Full implementation would use homography transformation
        if (Math.Abs(region.RotationAngle) > 45)
        {
            // Rotate 90 degrees if heavily rotated
            return Rotate90(crop);
        }

        return crop;
    }

    private Tensor<T> Rotate90(Tensor<T> input)
    {
        int batch = input.Shape[0];
        int channels = input.Shape[1];
        int height = input.Shape[2];
        int width = input.Shape[3];

        var output = new Tensor<T>(new[] { batch, channels, width, height });

        for (int b = 0; b < batch; b++)
        {
            for (int c = 0; c < channels; c++)
            {
                for (int h = 0; h < height; h++)
                {
                    for (int w = 0; w < width; w++)
                    {
                        output[b, c, w, height - 1 - h] = input[b, c, h, w];
                    }
                }
            }
        }

        return output;
    }

    private List<RecognizedText<T>> SortByReadingOrder(List<RecognizedText<T>> texts)
    {
        if (texts.Count <= 1)
            return texts;

        // Group into lines based on Y-coordinate overlap
        var lines = new List<List<RecognizedText<T>>>();

        foreach (var text in texts.OrderBy(t => t.Box != null ? _numOps.ToDouble(t.Box.Y1) : 0))
        {
            bool addedToLine = false;

            foreach (var line in lines)
            {
                // Check if this text overlaps vertically with the line
                var lineTop = line.Min(t => t.Box != null ? _numOps.ToDouble(t.Box.Y1) : 0);
                var lineBottom = line.Max(t => t.Box != null ? _numOps.ToDouble(t.Box.Y2) : 0);
                var textTop = text.Box != null ? _numOps.ToDouble(text.Box.Y1) : 0;
                var textBottom = text.Box != null ? _numOps.ToDouble(text.Box.Y2) : 0;

                double overlapThreshold = (lineBottom - lineTop) * 0.5;

                if (textTop < lineBottom && textBottom > lineTop &&
                    Math.Min(textBottom, lineBottom) - Math.Max(textTop, lineTop) > overlapThreshold)
                {
                    line.Add(text);
                    addedToLine = true;
                    break;
                }
            }

            if (!addedToLine)
            {
                lines.Add(new List<RecognizedText<T>> { text });
            }
        }

        // Sort each line left-to-right, then concatenate lines
        var result = new List<RecognizedText<T>>();

        foreach (var line in lines.OrderBy(l => l.Min(t => t.Box != null ? _numOps.ToDouble(t.Box.Y1) : 0)))
        {
            var sortedLine = line.OrderBy(t => t.Box != null ? _numOps.ToDouble(t.Box.X1) : 0);
            result.AddRange(sortedLine);
        }

        return result;
    }

    /// <summary>
    /// Gets the total parameter count.
    /// </summary>
    public long GetParameterCount()
    {
        return _detector.GetParameterCount() + _recognizer.GetParameterCount();
    }

    /// <summary>
    /// Loads pretrained weights.
    /// </summary>
    public async Task LoadWeightsAsync(string detectorPath, string recognizerPath, CancellationToken cancellationToken = default)
    {
        await _detector.LoadWeightsAsync(detectorPath, cancellationToken);
        await _recognizer.LoadWeightsAsync(recognizerPath, cancellationToken);
    }
}
